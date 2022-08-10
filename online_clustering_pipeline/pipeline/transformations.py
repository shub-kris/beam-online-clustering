import logging
import uuid
from bisect import insort
from collections import Counter, defaultdict
from typing import Callable, DefaultDict, List, Tuple

import apache_beam as beam
import numpy as np
from apache_beam.coders import PickleCoder
from apache_beam.transforms.userstate import ReadModifyWriteStateSpec

# import tensorflow_hub as hub
from sentence_transformers import SentenceTransformer
from sklearn.cluster import Birch


class SentenceEmbedder:
    def __init__(self):
        # self.model = hub.load("https://tfhub.dev/google/universal-sentence-encoder-large/5")
        self.model = SentenceTransformer("all-MiniLM-L6-v2")

    def get_sentence_embedding(self, text: str, size_limit: int = 500):
        truncated_text = text[:size_limit]
        return self.model.encode([truncated_text])[0]


## Initialize it here so that you don't have to invoke a new class object all the time
sentence_embedder = SentenceEmbedder()


class GetEmbedding(beam.DoFn):
    def process(self, element, *args, **kwargs):
        sentence_embedding = sentence_embedder.get_sentence_embedding(element["text"])
        yield {**element, "embedding": sentence_embedding}


class NormalizeEmbedding(beam.DoFn):
    def process(self, element, *args, **kwargs):
        embedding = element.get("embedding")
        l2_norm = np.linalg.norm(embedding)
        yield {**element, "normalized_text_embedding": embedding / l2_norm}


class Decode(beam.DoFn):
    def process(self, element, *args, **kwargs):
        attributes = element.attributes
        # print(attributes)
        yield {
            "text": attributes["text"].replace("\n", " "),
            "uuid": attributes["uuid"],
        }


class StatefulOnlineClustering(beam.DoFn):

    BIRCH_MODEL_SPEC = ReadModifyWriteStateSpec("clustering_model", PickleCoder())
    LABEL_MAP_SPEC = ReadModifyWriteStateSpec("label_map", PickleCoder())
    NEWS_ITEMS_SPEC = ReadModifyWriteStateSpec("news_items", PickleCoder())
    EMBEDDINGS_SPEC = ReadModifyWriteStateSpec("embeddings", PickleCoder())
    PREVIOUS_ASSIGNMENT_SPEC = ReadModifyWriteStateSpec(
        "previous_event_assignment", PickleCoder()
    )
    UPDATE_COUNTER_SPEC = ReadModifyWriteStateSpec("update_counter", PickleCoder())

    @staticmethod
    def assign_unique_labels(
        cluster_labels: list,
        document_ids: list,
        label_map: DefaultDict[Tuple[str], str],
        unique_label_fun: Callable = uuid.uuid4,
    ):
        # Rename cluster labels as sorted tuple of document ids
        membership_labels = defaultdict(list)
        for cluster_label, item_id in zip(cluster_labels, document_ids):
            insort(membership_labels[cluster_label], item_id)

        labels = [
            tuple(membership_labels[cluster_label]) for cluster_label in cluster_labels
        ]

        def find_subset_label(label: tuple, label_map: DefaultDict[Tuple[str], str]):
            """
            It finds the previous label that is a subset of the current label

            Args:
              label (tuple): the label we're trying to find a subset for
              label_map (DefaultDict[Tuple[str], str]): A dictionary of labels to their corresponding label
            names.

            Returns:
              The previous label that is a subset of the current label.
            """
            label_differences = [
                (previous_label, set(previous_label) - set(label))
                for previous_label in label_map.keys()
            ]
            previous_label = next(
                (
                    previous_label
                    for (previous_label, difference) in label_differences
                    if len(difference) == 0
                ),
                None,
            )
            return previous_label

        # Update label map
        for label in labels:
            if label not in label_map:
                previous_label = find_subset_label(label, label_map)
                if previous_label is None:
                    label_map[label] = str(unique_label_fun())
                else:
                    label_map[label] = label_map.pop(previous_label)

        # Collect unique labels for docs
        unique_labels = [label_map[label] for label in labels]

        return unique_labels, label_map

    @staticmethod
    def collect_updated_clusters(
        cluster_labels,
        label_map,
        collected_documents,
        collected_embeddings,
        previous_assignments,
        update_counter,
    ):
        # Assign unique cluster labels to documents
        unique_labels, label_map = StatefulOnlineClustering.assign_unique_labels(
            cluster_labels, collected_embeddings.keys(), label_map
        )
        new_assignments = dict(zip(collected_embeddings.keys(), unique_labels))

        # Collect updated documents
        changed_clusters = set(
            dict(
                set(new_assignments.items()) - set(previous_assignments.items())
            ).values()
        )
        clusters_awaiting_update = defaultdict(dict)
        for item_id, cluster_label in new_assignments.items():
            if cluster_label in changed_clusters:
                clusters_awaiting_update[cluster_label].update(
                    {item_id: collected_documents[item_id]}
                )
        update_counter.update(clusters_awaiting_update.keys())

        return clusters_awaiting_update, label_map, new_assignments, update_counter

    def process(
        self,
        element,
        model_state=beam.DoFn.StateParam(BIRCH_MODEL_SPEC),
        label_map_state=beam.DoFn.StateParam(LABEL_MAP_SPEC),
        collected_docs_state=beam.DoFn.StateParam(NEWS_ITEMS_SPEC),
        collected_embeddings_state=beam.DoFn.StateParam(EMBEDDINGS_SPEC),
        previous_assignments_state=beam.DoFn.StateParam(PREVIOUS_ASSIGNMENT_SPEC),
        update_counter_state=beam.DoFn.StateParam(UPDATE_COUNTER_SPEC),
        *args,
        **kwargs,
    ):
        # 1. Initialise or load states
        clustering = model_state.read() or Birch(n_clusters=None, threshold=0.7)
        label_map = label_map_state.read() or dict()
        collected_documents = collected_docs_state.read() or dict()
        collected_embeddings = collected_embeddings_state.read() or dict()
        previous_assignments = previous_assignments_state.read() or dict()
        update_counter = update_counter_state.read() or Counter()

        # 2. Extract document, add to state, and add to clustering model
        _, doc = element
        doc_uuid = doc["uuid"]
        embedding_vector = doc["normalized_text_embedding"]
        collected_embeddings[doc_uuid] = embedding_vector
        collected_documents[doc_uuid] = {"uuid": doc_uuid, "text": doc["text"]}
        # collected_documents[doc_uuid] = {"uuid": doc["uuid"]}
        # collected_documents[doc_uuid] = {"text": doc["text"]}

        clustering.partial_fit(np.atleast_2d(embedding_vector))

        # 3. Predict cluster labels of collected documents
        cluster_labels = clustering.predict(
            np.array(list(collected_embeddings.values()))
        )
        (
            clusters_awaiting_update,
            label_map,
            new_assignments,
            update_counter,
        ) = StatefulOnlineClustering.collect_updated_clusters(
            cluster_labels,
            label_map,
            collected_documents,
            collected_embeddings,
            previous_assignments,
            update_counter,
        )

        # 4. Write states
        model_state.write(clustering)
        label_map_state.write(label_map)
        collected_docs_state.write(collected_documents)
        collected_embeddings_state.write(collected_embeddings)
        previous_assignments_state.write(new_assignments)
        update_counter_state.write(update_counter)

        # 5. Yield updated clusters
        for cluster_id, items in clusters_awaiting_update.items():
            yield {
                "cluster_id": cluster_id,
                "updates": update_counter[cluster_id],
                "documents": items,
            }


class FormatClusterUpdate(beam.DoFn):
    def process(self, element, *args, **kwargs):
        if element["updates"] > 1:
            instruction = f"Update:      {element['cluster_id']}"
        else:
            instruction = f"New cluster: {element['cluster_id']}"
        print(instruction)
        print(f"{element['documents']}\n\n")
        # print(f"Documents": )
        # print("Documents:", *element["documents"], sep="\n\t", end="\n\n")
        yield {
            **element,
        }
