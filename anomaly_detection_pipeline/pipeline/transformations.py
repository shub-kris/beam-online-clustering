import apache_beam as beam
import hdbscan
import numpy as np
from joblib import load
from sentence_transformers import SentenceTransformer


class SentenceEmbedder:
    def __init__(self):
        self.model = SentenceTransformer("all-MiniLM-L6-v2")

    def get_sentence_embedding(self, text: str, size_limit: int = 500):
        """
        It takes a string of text, truncates it to 500 characters, and then returns the embedding of that
        truncated text
        Args:
          text (str): The text to be embedded.
          size_limit (int): The maximum number of characters to use from the text. Defaults to 500
        Returns:
          An embedding vector
        """
        truncated_text = text[:size_limit]
        return self.model.encode([truncated_text])[0]


## Initialize it here so that you don't have to invoke a new class object all the time
sentence_embedder = SentenceEmbedder()
# Load the trained clustering model
clustering_model = load("clustering.joblib")


class GetEmbedding(beam.DoFn):
    def process(self, element, *args, **kwargs):
        """
        > For each element in the input PCollection, get the sentence embedding for the text field, and
        yield a new element with the embedding added
        Args:
          element: the element that is being processed
        """
        sentence_embedding = sentence_embedder.get_sentence_embedding(element["text"])
        yield {**element, "embedding": sentence_embedding}


class NormalizeEmbedding(beam.DoFn):
    def process(self, element, *args, **kwargs):
        """
        For each element in the input PCollection, normalize the embedding vector, and
        yield a new element with the normalized embedding added
        Args:
          element: The element to be processed.
        """
        embedding = element.get("embedding")
        l2_norm = np.linalg.norm(embedding)
        yield {**element, "normalized_text_embedding": embedding / l2_norm}


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


class Inference(beam.DoFn):
    def process(self, element, *args, **kwargs):
        """
        It takes in a dictionary of data that contains text and the embeddings,
        and returns a dictionary of data with cluster label added

        Args:
          element: The element that is being processed.
        """
        normalized_embedding = np.expand_dims(
            element.get("normalized_text_embedding"), axis=0
        )
        label, strength = hdbscan.approximate_predict(
            clustering_model, normalized_embedding
        )
        yield {**element, "cluster": label[0]}
