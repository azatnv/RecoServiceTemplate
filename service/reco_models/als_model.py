import pickle
from math import exp
from typing import Dict, List, Optional, Tuple

from service.reco_models.model import RecommendationModel


class ALS(RecommendationModel):
    def __init__(self, paths: Dict[str, str]) -> None:
        with open(paths["ui_csr"], "rb") as f:
            self.ui_csr = pickle.load(f)

        with open(paths["user_ext_to_int"], "rb") as f:
            self.user_ext_to_int = pickle.load(f)
        with open(paths["item_int_to_ext"], "rb") as f:
            self.item_int_to_ext = pickle.load(f)
        with open(paths["item_ext_to_int"], "rb") as f:
            self.item_ext_to_int = pickle.load(f)

        with open(paths["als_model"], "rb") as f:
            self.model = pickle.load(f)

        with open(paths["item_to_title"], "rb") as f:
            self.item_to_title = pickle.load(f)

    def predict(self, user_id: int, k_recs: int = 10) -> Optional[List[int]]:
        int_user_id = self.user_ext_to_int[user_id]
        rec = self.model.recommend(int_user_id, user_items=self.ui_csr, N=k_recs, filter_already_liked_items=True)
        return [self.item_int_to_ext[item_int_id] for (item_int_id, _) in rec]

    def explain(self, user_id: int, item_id: int, threshold: float = 0.05) -> Tuple[Optional[int], Optional[str]]:
        if user_id not in self.user_ext_to_int or item_id not in self.item_ext_to_int:
            return None, None

        internal_userid = self.user_ext_to_int[user_id]
        internal_itemid = self.item_ext_to_int[item_id]

        total_score, top_contributions, _ = self.model.explain(
            userid=internal_userid, user_items=self.ui_csr, itemid=internal_itemid, N=2
        )
        if total_score < threshold:
            return None, None

        p = int((0.5 / (1 + exp(-(total_score * 5 - 1))) + 0.5) * 100)

        title_1 = self.item_to_title[self.item_int_to_ext[top_contributions[0][0]]]
        explanation = f"Рекомендуем тем, кому нравится «{title_1}»"

        if top_contributions[1][1] >= threshold:
            title_2 = self.item_to_title[self.item_int_to_ext[top_contributions[1][0]]]
            explanation += f" и «{title_2}»"

        return p, explanation
