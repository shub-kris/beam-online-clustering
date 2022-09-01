import json

import apache_beam as beam
import config as cfg
import hdbscan
import numpy as np
import torch
import yagmail
from apache_beam.ml.inference.base import PredictionResult
from apache_beam.ml.inference.sklearn_inference import (
    SklearnModelHandlerNumpy,
    _validate_inference_args,
)
from transformers import AutoTokenizer, DistilBertModel

Tokenizer = AutoTokenizer.from_pretrained(cfg.TOKENIZER_NAME)


def tokenize_sentence(input_dict):
    """
    It takes a dictionary with a text and an id, tokenizes the text, and returns a tuple of the text and
    id and the tokenized text

    Args:
      input_dict: a dictionary with the text and id of the sentence

    Returns:
      A tuple of the text and id, and a dictionary of the tokens.
    """
    text, id = input_dict["text"], input_dict["id"]
    tokens = Tokenizer([text], padding=True, truncation=True, return_tensors="pt")
    tokens = {key: torch.squeeze(val) for key, val in tokens.items()}
    return (text, id), tokens


class ModelWrapper(DistilBertModel):
    def forward(self, **kwargs):
        output = super().forward(**kwargs)
        sentence_embedding = (
            self.mean_pooling(output, kwargs["attention_mask"]).detach().cpu().numpy()
        )
        return sentence_embedding

    # Mean Pooling - Take attention mask into account for correct averaging
    def mean_pooling(self, model_output, attention_mask):
        token_embeddings = model_output[
            0
        ]  # First element of model_output contains all token embeddings
        input_mask_expanded = (
            attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        )
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(
            input_mask_expanded.sum(1), min=1e-9
        )


class CustomSklearnModelHandlerNumpy(SklearnModelHandlerNumpy):
    def batch_elements_kwargs(self):
        """Limit batch size to 1 for inference"""
        return {"max_batch_size": 1}

    def run_inference(self, batch, model, inference_args=None):
        """Runs inferences on a batch of numpy arrays.

        Args:
          batch: A sequence of examples as numpy arrays. They should
            be single examples.
          model: A numpy model or pipeline. Must implement predict(X).
            Where the parameter X is a numpy array.
          inference_args: Any additional arguments for an inference.

        Returns:
          An Iterable of type PredictionResult.
        """
        _validate_inference_args(inference_args)
        vectorized_batch = np.vstack(batch)
        predictions = hdbscan.approximate_predict(model, vectorized_batch)
        return [PredictionResult(x, y) for x, y in zip(batch, predictions)]


class NormalizeEmbedding(beam.DoFn):
    def process(self, element, *args, **kwargs):
        """
        For each element in the input PCollection, normalize the embedding vector, and
        yield a new element with the normalized embedding added
        Args:
          element: The element to be processed.
        """
        (text, id), prediction = element
        embedding = prediction.inference
        l2_norm = np.linalg.norm(embedding)
        yield (text, id), np.expand_dims(embedding / l2_norm, axis=0)


class Decode(beam.DoFn):
    def process(self, element, *args, **kwargs):
        """
        For each element in the input PCollection, retrieve the id and decode the bytes into string

        Args:
          element: The element that is being processed.
        """
        yield {
            "text": element.data.decode("utf-8"),
            "id": element.attributes["id"],
        }


class DecodePrediction(beam.DoFn):
    def process(self, element, *args, **kwargs):
        (text, id), prediction = element
        cluster = prediction.inference.item()
        bq_dict = {"text": text, "id": id, "cluster": cluster}
        yield bq_dict


def trigger_email_alert(receiver: str = "shubham.krishna@ml6.eu"):
    """
    It sends an email to the specified receiver with the specified body
    Args:
      receiver (str): The email address of the person who will receive the alert. Defaults to
    shubham.krishna@ml6.eu
    """
    with open("./cred.json") as json_file:
        cred = json.load(json_file)
    yag = yagmail.SMTP(**cred)
    body = "A new cluster has been created"
    yag.send(to=receiver, subject="New Cluster Alert", contents=body)
