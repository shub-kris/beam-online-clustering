import json
from collections import Counter, defaultdict

import apache_beam as beam
import numpy as np
import yagmail
from apache_beam.coders import PickleCoder
from apache_beam.transforms.userstate import ReadModifyWriteStateSpec
from sentence_transformers import SentenceTransformer
from sklearn.cluster import Birch


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


class StatefulOnlineClustering(beam.DoFn):

    BIRCH_MODEL_SPEC = ReadModifyWriteStateSpec("clustering_model", PickleCoder())
    DATA_ITEMS_SPEC = ReadModifyWriteStateSpec("data_items", PickleCoder())
    EMBEDDINGS_SPEC = ReadModifyWriteStateSpec("embeddings", PickleCoder())
    UPDATE_COUNTER_SPEC = ReadModifyWriteStateSpec("update_counter", PickleCoder())
    CLUSTERS_COUNTER_SPEC = ReadModifyWriteStateSpec("num_clusters", PickleCoder())

    def process(
        self,
        element,
        model_state=beam.DoFn.StateParam(BIRCH_MODEL_SPEC),
        collected_docs_state=beam.DoFn.StateParam(DATA_ITEMS_SPEC),
        collected_embeddings_state=beam.DoFn.StateParam(EMBEDDINGS_SPEC),
        update_counter_state=beam.DoFn.StateParam(UPDATE_COUNTER_SPEC),
        num_clusters_state=beam.DoFn.StateParam(CLUSTERS_COUNTER_SPEC),
        *args,
        **kwargs,
    ):
        """
        Takes the embedding of a document and updates the clustering model

        Args:
          element: The input element to be processed.
          model_state: This is the state of the clustering model. It is a stateful parameter, which means
        that it will be updated after each call to the process function.
          collected_docs_state: This is a stateful dictionary that stores the documents that have been
        processed so far.
          collected_embeddings_state: This is a dictionary of document IDs and their embeddings.
          update_counter_state: This is a counter that keeps track of how many documents have been
        processed.
        """
        # 1. Initialise or load states
        clustering = model_state.read() or Birch(n_clusters=None, threshold=0.7)
        collected_documents = collected_docs_state.read() or dict()
        collected_embeddings = collected_embeddings_state.read() or dict()
        update_counter = update_counter_state.read() or Counter()
        prev_num_clusters = num_clusters_state.read() or 0

        # 2. Extract document, add to state, and add to clustering model
        _, doc = element
        doc_id = doc["id"]
        embedding_vector = doc["normalized_text_embedding"]
        collected_embeddings[doc_id] = embedding_vector
        collected_documents[doc_id] = {"id": doc_id, "text": doc["text"]}
        update_counter = len(collected_documents)

        clustering.partial_fit(np.atleast_2d(embedding_vector))

        # 3. Predict cluster labels of collected documents
        cluster_labels = clustering.predict(
            np.array(list(collected_embeddings.values()))
        )
        num_clusters = len(set(cluster_labels))

        # trigger email alert if new clusters are formed
        if num_clusters > prev_num_clusters:
            trigger_email_alert()

        # 4. Write states
        model_state.write(clustering)
        collected_docs_state.write(collected_documents)
        collected_embeddings_state.write(collected_embeddings)
        update_counter_state.write(update_counter)
        num_clusters_state.write(num_clusters)

        yield {
            "labels": cluster_labels,
            "docs": collected_documents,
            "id": collected_embeddings.keys(),
            "counter": update_counter,
        }


def trigger_email_alert(receiver: str = "shubham.krishna@ml6.eu"):
    with open("./cred.json") as json_file:
        cred = json.load(json_file)
    yag = yagmail.SMTP(**cred)
    body = "A new cluster has been created"
    yag.send(to=receiver, subject="New Cluster Alert", contents=body)


class GetUpdates(beam.DoFn):
    def process(self, element, *args, **kwargs):
        """
        Prints and returns clusters with items contained in it
        """
        cluster_labels = element.get("labels")
        doc_ids = element.get("id")
        docs = element.get("docs")
        print(f"Update Number: {element.get('counter')}:::\n")
        label_items_map = defaultdict(list)
        for doc_id, cluster_label in zip(doc_ids, cluster_labels):
            label_items_map[cluster_label].append(docs[doc_id])
            # print(f"Doc-Text: {docs[doc_id]}, cluster_label: {cluster_label}")
        print(label_items_map)
        print("\n\n\n\n")
        yield label_items_map
