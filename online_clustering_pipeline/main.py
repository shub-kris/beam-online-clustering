import argparse
import sys

import apache_beam as beam
import config as cfg
from apache_beam.io.gcp.pubsub import ReadFromPubSub
from apache_beam.ml.inference.base import KeyedModelHandler, RunInference
from apache_beam.ml.inference.pytorch_inference import PytorchModelHandlerKeyedTensor
from pipeline.options import get_pipeline_options
from pipeline.transformations import (
    Decode,
    GetUpdates,
    ModelWrapper,
    NormalizeEmbedding,
    StatefulOnlineClustering,
    tokenize_sentence,
)
from transformers import AutoConfig


def parse_arguments(argv):
    parser = argparse.ArgumentParser(description="catalog-data")

    parser.add_argument(
        "-m",
        "--mode",
        help="Mode to run pipeline in.",
        choices=["local", "cloud"],
        default="local",
    )
    parser.add_argument(
        "-p",
        "--project",
        help="GCP project to run pipeline on.",
        default=cfg.PROJECT_ID,
    )

    args, _ = parser.parse_known_args(args=argv)
    return args


def run():
    args = parse_arguments(sys.argv)
    pipeline_options = get_pipeline_options(
        job_name=cfg.JOB_NAME,
        num_workers=cfg.NUM_WORKERS,
        max_num_workers=cfg.MAX_NUM_WORKERS,
        project=args.project,
        mode=args.mode,
    )

    model_handler = PytorchModelHandlerKeyedTensor(
        state_dict_path=cfg.MODEL_STATE_DICT_PATH,
        model_class=ModelWrapper,
        model_params={"config": AutoConfig.from_pretrained(cfg.MODEL_CONFIG_PATH)},
        device="cpu",
    )

    with beam.Pipeline(options=pipeline_options) as pipeline:
        docs = (
            pipeline
            | "Read from PubSub"
            >> ReadFromPubSub(subscription=cfg.SUBSCRIPTION_ID, with_attributes=True)
            | "Decode PubSubMessage" >> beam.ParDo(Decode())
        )
        normalized_embedding = (
            docs
            | "Tokenize Text" >> beam.Map(tokenize_sentence)
            | "Get Embedding" >> RunInference(KeyedModelHandler(model_handler))
            | "Normalize Embedding" >> beam.ParDo(NormalizeEmbedding())
        )
        clustering = (
            normalized_embedding
            | "Map doc to key" >> beam.Map(lambda x: (1, x))
            | "StatefulClustering using Birch" >> beam.ParDo(StatefulOnlineClustering())
        )

        updated_clusters = clustering | "Format Update" >> beam.ParDo(GetUpdates())


if __name__ == "__main__":
    run()
