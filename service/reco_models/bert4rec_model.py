import pickle
from typing import List, Optional

from service.reco_models.model import RecommendationModel


class BERT4Rec(RecommendationModel):
    def __init__(self, model_path) -> None:
        with open(model_path, "rb") as pickle_fo:
            self.model = pickle.load(pickle_fo)

    def predict(self, user_id: int, k_recs: int = 10) -> Optional[List[int]]:
        if user_id not in self.model:
            return None

        return self.model[user_id]
