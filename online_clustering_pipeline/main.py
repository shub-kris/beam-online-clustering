import argparse
import logging
import sys
from pprint import pprint

import apache_beam as beam
from apache_beam.io.gcp.pubsub import ReadFromPubSub
from apache_beam.transforms import trigger, window

import config as cfg
from pipeline.options import get_pipeline_options
from pipeline.transformations import (
    Decode,
    FormatClusterUpdate,
    GetEmbedding,
    NormalizeEmbedding,
    StatefulOnlineClustering,
)


def parse_arguments(argv):
    parser = argparse.ArgumentParser(description="catalog-data")

    parser.add_argument(
        "-m",
        "--mode",
        help="Mode to run pipeline in.",
        choices=["local", "cloud", "template"],
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
    # example_docs = [
    #     {
    #         "uuid": "Movies (Star Wars 1)",
    #         "text": "Star Wars is my favourite movie!",
    #     },
    #     {
    #         "uuid": "Movies (Star Wars 2)",
    #         "text": "I reject the later edits. Clearly, Han Solo shot first!",
    #     },
    #     {
    #         "uuid": "Turtles",
    #         "text": "I like turtles.",
    #     },
    #     {
    #         "uuid": "Weather 1",
    #         "text": (
    #             "The weather next week will be cold and dry with bouts of fog "
    #             "but there are signs rain and possibly snow could arrive by Christmas Day."
    #         ),
    #     },
    #     {
    #         "uuid": "Movies (Star Trek)",
    #         "text": "Star Trek is an awesome series.",
    #     },
    #     {
    #         "uuid": "Weather 2",
    #         "text": (
    #             "After a stormy week, strong winds will blow and it will rain intermittently."
    #         ),
    #     },
    # ]

    with beam.Pipeline(options=pipeline_options) as pipeline:
        docs = (
            pipeline
            | "Read from PubSub"
            >> ReadFromPubSub(subscription=cfg.SUBSCRIPTION_ID, with_attributes=True)
            | "Decode PubSubMessage" >> beam.ParDo(Decode())
        )
        # docs = pipeline | "Load Documents" >> beam.Create(example_docs)
        embedding = (
            docs
            | "Get Text Embedding" >> beam.ParDo(GetEmbedding())
            | "Normalize Embedding" >> beam.ParDo(NormalizeEmbedding())
        )
        clustering = (
            embedding
            | "Map doc to key" >> beam.Map(lambda x: (1, x))
            | "StatefulClustering using Birch" >> beam.ParDo(StatefulOnlineClustering())
        )

        _ = clustering | "Format Update" >> beam.ParDo(FormatClusterUpdate())
        # _ = (
        #     embedding
        #     | "Print" >> beam.Map(pprint)
        # )


if __name__ == "__main__":
    run()
