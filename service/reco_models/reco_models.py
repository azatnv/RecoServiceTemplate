import pickle
from abc import ABC, abstractmethod
from typing import Dict, List, Optional

import dill
import numpy as np
from lightfm import LightFM
from numpy.typing import NDArray
from scipy import sparse


class SimplePopularModel:
    def __init__(self, users_path: str, recs_path: str):
        self.users_dictionary: Dict[int, str] = pickle.load(
            open(users_path, "rb")
        )
        self.popular_dictionary: Dict[str, List[int]] = pickle.load(
            open(recs_path, "rb")
        )

    def predict(self, user_id: int, k_recs: int) -> List[int]:
        try:
            # Check if user is suitable for category reco
            category = self.users_dictionary.get(user_id, None)
            if category:
                return self.popular_dictionary[category][:k_recs]
            # If not the case, give him popular on average
            return self.popular_dictionary["popular_for_all"][:k_recs]
        except TypeError:
            return list(range(k_recs))


class Model(ABC):
    def __init__(self, name: str):
        with open(f"{name}", "rb") as f:
            self.model = dill.load(f)

    @abstractmethod
    def predict(self, user_id: int) -> Optional[List[int]]:
        pass


class OfflineKnnModel(Model):
    def predict(self, user_id: int) -> Optional[List[int]]:
        if user_id in self.model.keys():
            return self.model[user_id]
        return None


class OnlineKnnModel(Model):
    def predict(self, user_id: int) -> Optional[List[int]]:
        return self.model.predict(user_id)


class FactorizationMachine(Model):
    def __init__(
        self,
        name: str,
        USER_MAPPING: str,
        ITEM_MAPPING: str,
        USERS_FEATURES: str,
        UNIQUE_FEATURES: str,
    ):
        super().__init__(name)
        with open(f"{name}", "rb") as f:
            self.model: LightFM = dill.load(f)
        with open(USER_MAPPING, "rb") as f:
            self.user_mapping: Dict[int, int] = dill.load(f)
        with open(ITEM_MAPPING, "rb") as f:
            self.item_mapping: Dict[int, int] = dill.load(f)
        with open(USERS_FEATURES, "rb") as f:
            self.users_features: Dict[int, Dict[str, str]] = dill.load(f)
        with open(UNIQUE_FEATURES, "rb") as f:
            self.features: NDArray[np.unicode_] = dill.load(f)

        self.items_internal_ids = np.arange(
            len(self.item_mapping.keys()), dtype=int
        )

    def _get_hot_reco(self, iternal_user_id: int) -> List[int]:
        hot_scores: NDArray[np.float32] = self.model.predict(
            iternal_user_id, item_ids=self.items_internal_ids
        )
        idxs = np.argsort(hot_scores)[::-1]
        recs = self.items_internal_ids[idxs][:10]
        recs = [self.item_mapping[reco] for reco in recs]
        return recs

    def _get_cold_reco(self, user_id: int) -> Optional[List[int]]:
        user_feature = self.users_features.get(user_id, None)
        if not user_feature:
            return None

        user_feature_list = list(user_feature.values())

        # If no features then we'll use another model
        if len(user_feature_list) == 0:
            return None

        feature_mask = np.isin(self.features, user_feature_list)
        feature_row = sparse.csr_matrix(feature_mask)

        cold_scores: NDArray[np.float32] = self.model.predict(
            0, self.items_internal_ids, user_features=feature_row
        )

        idxs = np.argsort(cold_scores)[::-1]
        recs = self.items_internal_ids[idxs][:10]
        recs = [self.item_mapping[reco] for reco in recs]
        return recs

    def predict(self, user_id: int) -> Optional[List[int]]:
        # Check if user is hot or not
        iternal_user_id = self.user_mapping.get(user_id, None)
        if iternal_user_id:
            return self._get_hot_reco(iternal_user_id=iternal_user_id)
        return None
        # return self._get_cold_reco(user_id=user_id)
