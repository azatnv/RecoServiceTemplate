import json
import pickle

import hnswlib

from service.reco_models.model import RecommendationModel


class DSSM(RecommendationModel):
    def __init__(
        self,
        index_path: str,
        user_vectors_path: str,
        user_id_to_uid_path: str,
        iid_to_item_id_path: str,
        uid_to_watched_iids_path: str,
        dim: int = 128,
        ef_s: int = 50,
    ):
        self.index = hnswlib.Index("cosine", dim)
        self.index.load_index(index_path)
        self.index.set_ef(ef_s)

        with open(user_vectors_path, "rb") as pickle_fo:
            self.dssm_user_vectors = pickle.load(pickle_fo)

        with open(user_id_to_uid_path, "r") as json_fo:
            self.user_id_to_uid = json.load(json_fo)
        with open(iid_to_item_id_path, "r") as json_fo:
            self.iid_to_item_id = json.load(json_fo)

        with open(uid_to_watched_iids_path, "r") as json_fo:
            self.uid_to_watched_iids = json.load(json_fo)

    def predict(self, user_id: int, k_recs: int = 10):
        if str(user_id) not in self.user_id_to_uid:
            return None

        uid = self.user_id_to_uid[str(user_id)]

        watched = self.uid_to_watched_iids[str(uid)]

        pred_iids, _ = self.index.knn_query(self.dssm_user_vectors[uid], k=k_recs + len(watched))
        answer_iids = []
        for iid in pred_iids[0]:
            if iid not in watched:
                answer_iids.append(iid)
            if len(answer_iids) == k_recs:
                break

        return [self.iid_to_item_id[str(iid)] for iid in answer_iids]
