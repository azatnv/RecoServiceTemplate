import dill
import pickle
from abc import ABC, abstractmethod
from typing import List, Optional


class SimplePopularModel:
    def __init__(self, users_path: str, recs_path: str):
        self.users_dictionary = pickle.load(open(users_path, 'rb'))
        self.popular_dictionary = pickle.load(open(recs_path, 'rb'))

    def predict(self, user_id: int, k_recs: int) -> List[int]:
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
        with open(f"{name}", "rb") as f:
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
