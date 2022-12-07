import os
import pickle
from typing import Any, List, Optional
import numpy


cwd = os.path.dirname(__file__)


class simple_popular_model():

    def __init__(self):
        self.top_popular_list: List = pickle.load(
            open(
                os.path.join(
                    cwd,
                    'top_popular_list.pickle',
                ),
                'rb'
            )
        )

    def get_popular_reco(
        self,
        user_id: int,
        k_recs: int
    ) -> List:
        return list(self.top_popular_list[:k_recs])


class KnnModel:
    def __init__(self, path: str = f"{cwd}/hot_reco_dict.pickle"):
        with open(path, "rb") as f:
            self.model = pickle.load(f)

    def predict(self, user_id: int) -> Optional[List[int]]:
        if user_id in self.model.keys():
            return self.model[user_id]
        return None
