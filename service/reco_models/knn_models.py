from abc import abstractmethod
from typing import List, Optional

import dill

from service.reco_models.model import RecommendationModel


class KnnModel(RecommendationModel):
    def __init__(self, name: str):
        with open(f"{name}", "rb") as f:
            self.model = dill.load(f)

    @abstractmethod
    def predict(self, user_id: int, k_recs) -> Optional[List[int]]:
        pass


class OfflineKnnModel(KnnModel):
    def predict(self, user_id: int, k_recs) -> Optional[List[int]]:
        if user_id in self.model.keys():
            return self.model[user_id]
        return None


class OnlineKnnModel(KnnModel):
    def predict(self, user_id: int, k_recs) -> Optional[List[int]]:
        return self.model.predict(user_id)
