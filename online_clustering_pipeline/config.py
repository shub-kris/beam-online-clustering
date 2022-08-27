PROJECT_ID = "apache-beam-testing"
# Subscription for PubSub Topic
SUBSCRIPTION_ID = f"projects/{PROJECT_ID}/subscriptions/newsgroup-dataset-subscription"
JOB_NAME = "online-clustering-birch"
NUM_WORKERS = 1
MAX_NUM_WORKERS = 15


TABLE_SCHEMA = None

TOKENIZER_NAME = "sentence-transformers/stsb-distilbert-base"
MODEL_STATE_DICT_PATH = "./model_weights/pytorch_model.bin"
MODEL_CONFIG_PATH = "sentence-transformers/stsb-distilbert-base"
