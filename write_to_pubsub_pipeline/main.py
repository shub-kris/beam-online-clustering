import argparse
import sys
import uuid

import apache_beam as beam
from apache_beam.io import WriteToBigQuery
from apache_beam.io.gcp.pubsub import PubsubMessage, WriteToPubSub

import config as cfg
from pipeline.options import get_pipeline_options
from pipeline.utils import get_dataset


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
        docs = (
            pipeline
            | "Load Documents" >> beam.Create(train_data)
            | "Take only first few words" >> beam.Map(lambda x: x[:1024])
            | "Replace new lines with spaces"
            >> beam.Map(lambda x: x.replace("\n", " "))
            | "Assign unique key"
            >> beam.Map(lambda x: {"id": str(uuid.uuid4()), "text": x})
        )

        # Write to PubSub for streaming
        _ = (
            docs
            | "Convert to PubSub Message"
            >> beam.Map(
                lambda x: PubsubMessage(
                    data=x.get("text").encode("utf-8"), attributes={"id": x.get("id")}
                )
            )
            | "Write to PubSub"
            >> WriteToPubSub(topic=cfg.TOPIC_ID, with_attributes=True)
        )

        # # Write to BigQuery for tracking
        # _ = (
        #     docs
        #     | "Write to BQ" >> WriteToBigQuery(
        #         method='STREAMING_INSERTS',
        #         schema=cfg.TABLE_SCHEMA,
        #         write_disposition = beam.io.BigQueryDisposition.WRITE_APPEND,
        #         create_disposition=beam.io.BigQueryDisposition.CREATE_IF_NEEDED)
        #     )


if __name__ == "__main__":
    run()
