import argparse
import logging
from pprint import pprint
import sys

import apache_beam as beam
from pipeline.options import get_pipeline_options
import config as cfg
from pipeline.utils import get_dataset, ConvertToPubSubMessage
from apache_beam.io.gcp.pubsub import WriteToPubSub


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
    train_categories = [
        "talk.politics.guns",
        "rec.sport.hockey",
        "alt.atheism",
        "sci.med",
    ]
    test_categories = train_categories + ["comp.graphics"]
    train_data, train_labels = get_dataset(train_categories)
    train_data = train_data[:10]

    with beam.Pipeline(options=pipeline_options) as pipeline:
        _ = (
            pipeline
            | "Load Documents" >> beam.Create(train_data)
            | "Take only first few words" >> beam.Map(lambda x: x[:140])
            | "Convert to PubSub Message" >> beam.ParDo(ConvertToPubSubMessage())
            | "Write to PubSub"
            >> WriteToPubSub(topic=cfg.TOPIC_ID, with_attributes=True)
        )


if __name__ == "__main__":
    run()
