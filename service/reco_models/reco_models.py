import os
import dill
import pickle
from abc import ABC, abstractmethod
from typing import Dict, List, Optional

import pandas as pd


cwd = os.path.dirname(__file__)


class simple_popular_model():

    def __init__(self):
        self.users_dictionary = pickle.load(
            open(
                os.path.join(
                    cwd,
                    'users_dictionary.pickle'
                ),
                'rb'
            )
        )
        self.popular_dictionary: Dict = pickle.load(
            open(
                os.path.join(
                    cwd,
                    'popular_dictionary.pickle'
                ),
                'rb'
            )
        )

    def get_popular_reco(
        self,
        user_id: int,
        k_recs: int
    ) -> List:
        # проверяю юзер в датасете или нет
        try:
            category = self.users_dictionary[user_id]
            reco = self.popular_dictionary[category][:k_recs]
        except KeyError:
            # если юзер не принадлежит никакой категории, рекомендуем
            # ему популярное в среднем
            reco = self.popular_dictionary['popular_for_all'][:k_recs]
        return reco


class KnnModel(ABC):
    def __init__(self, name: str):
        with open(f"{cwd}/{name}", "rb") as f:
            self.model = pickle.load(f)

    @abstractmethod
    def predict(self, user_id: int) -> Optional[List[int]]:
        pass


class OfflineKnnModel(KnnModel):
    def predict(self, user_id: int) -> Optional[List[int]]:
        if user_id in self.model.keys():
            return self.model[user_id]
        return None


class OnlineKnnModel(KnnModel):

    def __init__(self, name: str):
        with open(f"{cwd}/{name}", "rb") as f:
            self.model = dill.load(f)

    def predict(self, user_id: int) -> Optional[List[int]]:
        recs = self.model.\
            predict(pd.DataFrame(data={"user_id": [user_id]}))["item_id"]\
            .tolist()
        return recs
