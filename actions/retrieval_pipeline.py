from abc import ABC, abstractclassmethod
from typing import Any, Text, Dict
import requests
import yaml
from yaml.loader import SafeLoader
import pandas as pd
import numpy as np
from transformers import (AutoTokenizer,
                          CamembertForSequenceClassification,
                          TextClassificationPipeline)
try:
    from .config import (FAQ_DATA_PATH,
                         MODEL_PATH,
                         FAQ_SELECTION_PATH)
except ImportError:
    from config import (FAQ_DATA_PATH,
                        MODEL_PATH,
                        FAQ_SELECTION_PATH)


class FaqRetrievalPipeline(ABC):
    """General class of faq retrievers"""

    NOT_FOUND_ANSWER = "Aucune réponse trouvée !"
    NOT_FOUND_SCORE = -1.0

    def __init__(self, threshold_value: float) -> None:
        super().__init__()
        self._threshold = threshold_value

    @property
    def threshold_value(self) -> float:
        """Get threshold value

        Returns:
            float: Value between 0-1
        """
        return self._threshold

    @threshold_value.setter
    def threshold_value(self, value: float) -> None:
        if not isinstance(value, float):
            raise TypeError(f"Incorrect type of threshold value {value}.")
        elif 0.0 <= value <= 1.0:
            raise ValueError(f"Threshold value {value} isn't between 0 and 1.")
        else:
            self._threshold = value

    @abstractclassmethod
    def retrieve_faq(self, query: Text) -> Dict[Text, Any]:
        pass


class FaqTransformersPipeline(FaqRetrievalPipeline):
    """Class that makes use of a transformers pipeline to get faq results
    """

    def __init__(self, threshold_value=0.5,
                 model_type=CamembertForSequenceClassification,
                 pipeline_type=TextClassificationPipeline,
                 device=0,
                 padding=True,
                 truncation=True,
                 batch_size=32,
                 **kwargs) -> None:
        super(FaqTransformersPipeline, self).__init__(threshold_value)
        self.model = model_type.from_pretrained(MODEL_PATH)
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
        self.pipeline = pipeline_type(model=self.model,
                                      tokenizer=self.tokenizer,
                                      padding=padding,
                                      truncation=truncation,
                                      device=device,
                                      batch_size=batch_size,
                                      **kwargs
                                      )
        self.preprocessing_fcn = PreprocessingInputs()
        self._read_faq_selection()
        self._initialize_local_faq()

    def _read_faq_selection(self):
        """Get the rules of faq selection
        """
        with open(FAQ_SELECTION_PATH) as f:
            self.rules = yaml.load(f, Loader=SafeLoader)

    def _initialize_local_faq(self):
        """Load the faq dataframe
        """
        # Loads the faq data
        df = pd.read_csv(FAQ_DATA_PATH)
        # Use all available entries
        if self.rules["use_all_faq_entries"]:
            self.df = df
        else:
            # Filter only possible ids
            ids = [id_s for id_s in self.rules['ids_of_selection']
                   if id_s < len(df)]
            self.df = df.iloc[ids].reset_index(drop=True)

    def retrieve_faq(self, query: Text) -> str:
        query_results = self.pipeline([self.preprocessing_fcn(
            query, ans) for ans in self.df.answer.values])
        query_results = [r_faq["score"] if r_faq["label"] ==
                         "LABEL_1" else 1.0-r_faq["score"] for r_faq in query_results]
        faq_index_max = np.argmax(query_results)
        score_max = query_results[faq_index_max]
        # If threshold is greater than the score than we filter the results
        if score_max < self._threshold:
            return {"answer": self.NOT_FOUND_ANSWER, "score": self.NOT_FOUND_SCORE}
        return {"answer": self.df.iloc[faq_index_max].answer, "score": score_max}


class PreprocessingInputs():
    """It perform the concatenation of query and answer strings with a separator

    Returns:
        str: result of concatenation
    """
    BEGIN_SENTENCE = "<s>"
    END_SENTENCE = "</s>"

    def __call__(self, query: str, answer: str) -> str:
        return self.BEGIN_SENTENCE+query+self.END_SENTENCE+self.BEGIN_SENTENCE+answer+self.END_SENTENCE


class FaqHaystackPipeline(FaqRetrievalPipeline):
    """This class implements the functions to retrieve FAQ entries from a running Haystack API REST
    For more details please see the documentation of Haystack:
    https://haystack.deepset.ai/guides/rest-api
    """
    # URL of Haystack API REST
    URL = "http://localhost:8000/query"

    def __init__(self, threshold_value=0.5) -> None:
        super(FaqHaystackPipeline, self).__init__(threshold_value)

    def retrieve_faq(self, query: Text) -> str:
        payload = {"query": query}
        headers = {
            'Content-Type': 'application/json'
        }
        response = requests.request(
            "POST", self.URL, headers=headers, json=payload).json()

        if response["answers"] and response["answers"][0]["score"] > self._threshold:
            score = response["answers"][0]["score"]
            answer = response["answers"][0]["answer"]
        else:
            answer = self.NOT_FOUND_ANSWER
            score = self.NOT_FOUND_SCORE
        return {"answer": answer, "score": score}