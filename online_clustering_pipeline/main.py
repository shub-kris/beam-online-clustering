import argparse
import sys

import apache_beam as beam
from apache_beam.io.gcp.pubsub import ReadFromPubSub

import config as cfg
from pipeline.options import get_pipeline_options
from pipeline.transformations import (
    Decode,
    GetEmbedding,
    GetUpdates,
    NormalizeEmbedding,
    StatefulOnlineClustering,
)


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

    with beam.Pipeline(options=pipeline_options) as pipeline:
        docs = (
            pipeline
            | "Read from PubSub"
            >> ReadFromPubSub(subscription=cfg.SUBSCRIPTION_ID, with_attributes=True)
            | "Decode PubSubMessage" >> beam.ParDo(Decode())
        )
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

        updated_clusters = clustering | "Format Update" >> beam.ParDo(GetUpdates())

if __name__ == "__main__":
    run()

        # _ = (
        #     updated_clusters
        #     | "Write to BQ" >> WriteToBigQuery(
        #         method='STREAMING_INSERTS',
        #         schema=cfg.TABLE_SCHEMA,
        #         write_disposition = beam.io.BigQueryDisposition.WRITE_APPEND,
        #         create_disposition=beam.io.BigQueryDisposition.CREATE_IF_NEEDED)
        # )
