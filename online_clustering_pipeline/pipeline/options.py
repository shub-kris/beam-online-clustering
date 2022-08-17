"""This file contains the pipeline options to configure the Dataflow pipeline."""
import os
from datetime import datetime
from typing import Any

from apache_beam.options.pipeline_options import PipelineOptions

import config as cfg


def get_pipeline_options(
    project: str,
    job_name: str,
    mode: str,
    num_workers: int = cfg.NUM_WORKERS,
    max_num_workers: int = cfg.MAX_NUM_WORKERS,
    streaming: bool = True,
    **kwargs: Any,
) -> PipelineOptions:
    """Function to retrieve the pipeline options.
    Args:
        project: GCP project to run on
        mode: Indicator to run local, cloud or template
        num_workers: Number of Workers for running the job parallely
        max_num_workers: Maximum number of workers running the job parallely
    Returns:
        Dataflow pipeline options
    """
    job_name = f'{job_name}-{datetime.now().strftime("%Y%m%d%H%M%S")}'

    staging_bucket = f"gs://{cfg.PROJECT_ID}-ml-examples"

    # For a list of available options, check:
    # https://cloud.google.com/dataflow/docs/guides/specifying-exec-params#setting-other-cloud-dataflow-pipeline-options
    dataflow_options = {
        "runner": "DirectRunner" if mode == "local" else "DataflowRunner",
        "job_name": job_name,
        "project": project,
        "region": "us-central1",
        "staging_location": f"{staging_bucket}/dflow-staging",
        "temp_location": f"{staging_bucket}/dflow-temp",
        "autoscaling_algorithm": "THROUGHPUT_BASED",
        "save_main_session": False,
        "setup_file": "./setup.py",
        "max_num_workers": cfg.MAX_NUM_WORKERS,
        "streaming": streaming,
    }

    # Optional parameters
    if num_workers:
        dataflow_options.update({"num_workers": num_workers})

    if max_num_workers:
        dataflow_options.update({"max_num_workers": max_num_workers})

    if mode == "template":
        dataflow_options["template_location"] = f"{staging_bucket}/templates/{job_name}"

    commit_hash = os.environ.get("commit_hash")
    if commit_hash:
        dataflow_options["commit_hash"] = commit_hash

    return PipelineOptions(flags=[], **dataflow_options)
