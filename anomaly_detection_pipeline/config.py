PROJECT_ID = "apache-beam-testing"
SUBSCRIPTION_ID = f"projects/{PROJECT_ID}/subscriptions/newsgroup-dataset-subscription"
JOB_NAME = "anomaly-detection-hdbscan"
NUM_WORKERS = 1


TABLE_SCHEMA = {
    "fields": [
        {"name": "text", "type": "STRING", "mode": "NULLABLE"},
        {"name": "id", "type": "STRING", "mode": "NULLABLE"},
        {"name": "cluster", "type": "INTEGER", "mode": "NULLABLE"},
    ]
}
TABLE_URI = f"{PROJECT_ID}:deliverables_ml6.anomaly-detection"

TOKENIZER_NAME = "sentence-transformers/stsb-distilbert-base"
MODEL_STATE_DICT_PATH = f"gs://{PROJECT_ID}-ml-examples/sentence-transformers-stsb-distilbert-base/pytorch_model.bin"
MODEL_CONFIG_PATH = TOKENIZER_NAME
CLUSTERING_MODEL_PATH = (
    f"gs://{PROJECT_ID}-ml-examples/anomaly-detection/clustering.joblib"
)
