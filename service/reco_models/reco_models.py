import os
import pickle
from abc import ABC, abstractmethod
from typing import Dict, List, Optional
import dill

from pandas import DataFrame
from scipy.sparse import coo_matrix
from numpy import log, ones, float32, array
from collections import Counter
from implicit.nearest_neighbours import ItemItemRecommender
from typing import Dict, Optional


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
            self.model = dill.load(f)

    @abstractmethod
    def predict(self, user_id: int) -> Optional[List[int]]:
        pass


class OfflineKnnModel(KnnModel):
    def predict(self, user_id: int) -> Optional[List[int]]:
        if user_id in self.model.keys():
            return self.model[user_id]
        return None


class OnlineKnnModel(KnnModel):
    def predict(self, user_id: int) -> Optional[List[int]]:
        return self.model.predict(user_id)
