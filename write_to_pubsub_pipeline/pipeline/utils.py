from sklearn.datasets import fetch_20newsgroups
import apache_beam as beam
from apache_beam.io.gcp.pubsub import PubsubMessage
import uuid


def get_dataset(categories: list, subset: str = "train"):
    """
    It fetches the 20 newsgroups dataset, removes headers, footers, and quotes, and returns the data and
    targets as lists

    Args:
      categories (list): list of categories to load.
      subset (str): train or test. Defaults to train

    Returns:
      A list of data and a list of targets.
    """
    newsgroups_subset = fetch_20newsgroups(
        subset=subset, remove=("headers", "footers", "quotes"), categories=categories
    )
    list_subset_data = list(newsgroups_subset.data)
    list_subset_targets = list(newsgroups_subset.target)
    return list_subset_data, list_subset_targets


class ConvertToPubSubMessage(beam.DoFn):
    def process(self, element):
        processed_string = element.replace("\n", " ")
        encoded_string =  processed_string.encode("utf-8")
        element = PubsubMessage(
            data=encoded_string, attributes={"text": processed_string, "uuid": str(uuid.uuid4())}
        )
        # element = PubsubMessage(
        #     data=encoded_string, attributes={"text": processed_string, "uuid": processed_string}
        # )
        yield element
