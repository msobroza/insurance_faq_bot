# This files contains your custom actions which can be used to run
# custom Python code.
#
# See this guide on how to implement these action:
# https://rasa.com/docs/rasa/custom-actions


from typing import Any, Text, Dict, List


from rasa_sdk import Action, Tracker
from rasa_sdk.executor import CollectingDispatcher
try:
    from .retrieval_pipeline import FaqTransformersPipeline
except ImportError:
    from retrieval_pipeline import FaqTransformersPipeline
# Defines the retrieval pipeline used
RETRIEVAL_PIPELINE = FaqTransformersPipeline()


class ActionFaqRetrieval(Action):
    """Implements the action of passage retriever

    Args:
        Action (_type_): action
    """

    def name(self) -> Text:
        """Unique identifier of the action"""

        return "call_haystack"

    def run(self, dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
        query = str(tracker.latest_message["text"])
        result = RETRIEVAL_PIPELINE.retrieve_faq(query)
        dispatcher.utter_message(text=result["answer"])