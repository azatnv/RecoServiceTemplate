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


class OnlineFM:
    """This class is implementation of recommendations generation with LightFM.

    LightFM library realization is utilized. We use built-in method predict()
    to generate recos both for hot users — i.e. who has interactions — and
    cold users — i.e. who could possibly have only features. If cold user
    has no features at all then popular model is the best option to make
    recommendation.

    Attributes:
        name: The model name to load
        model: The LightFM model itself
        user_mapping: The dictionary to make the transition
            internal (generated during the model fitting) -> external
        item_mapping: The dictionary to make the transition
            external -> internal (generated during the model fitting)
        features_for_cold: The features values for every known cold user
        features: The all possible features values set
        items_internal_ids:
        cold_with_fm: The flag to use LightFM model to to generate recos
            for cold users having features or not (use popular instead)

    """

    def __init__(
        self,
        name: str,
        USER_MAPPING: str,
        ITEM_MAPPING: str,
        FEATURES_FOR_COLD: str,
        UNIQUE_FEATURES: str,
        cold_with_fm: bool = True,
    ):
        with open(f"{name}", "rb") as f:
            self.model: LightFM = dill.load(f)
        with open(USER_MAPPING, "rb") as f:
            self.user_mapping: Dict[int, int] = dill.load(f)
        with open(ITEM_MAPPING, "rb") as f:
            self.item_mapping: Dict[int, int] = dill.load(f)
        with open(FEATURES_FOR_COLD, "rb") as f:
            self.features_for_cold: Dict[int, Dict[str, str]] = dill.load(f)
        with open(UNIQUE_FEATURES, "rb") as f:
            self.features: NDArray[np.unicode_] = dill.load(f)

        self.items_internal_ids = np.arange(
            len(self.item_mapping.keys()), dtype=int
        )
        self.cold_with_fm: bool = cold_with_fm

    def _get_hot_reco(self, iternal_user_id: int, k_recs: int) -> List[int]:
        hot_scores: NDArray[np.float32] = self.model.predict(
            iternal_user_id, item_ids=self.items_internal_ids
        )
        idxs = np.argsort(hot_scores)[::-1]
        recs = self.items_internal_ids[idxs][:k_recs]
        recs = [self.item_mapping[reco] for reco in recs]
        return recs

    def _get_cold_reco(
        self, user_feature: Dict[str, str], k_recs: int
    ) -> List[int]:
        user_feature_list = list(user_feature.values())
        feature_mask = np.isin(self.features, user_feature_list)
        feature_row = sparse.csr_matrix(feature_mask)

        cold_scores: NDArray[np.float32] = self.model.predict(
            0, self.items_internal_ids, user_features=feature_row
        )
        idxs = np.argsort(cold_scores)[::-1]
        recs = self.items_internal_ids[idxs][:k_recs]
        recs = [self.item_mapping[reco] for reco in recs]
        return recs

    def predict(self, user_id: int, k_recs: int) -> Optional[List[int]]:
        # Check if user is hot or not
        iternal_user_id = self.user_mapping.get(user_id, None)
        if iternal_user_id:
            return self._get_hot_reco(
                iternal_user_id=iternal_user_id, k_recs=k_recs
            )

        if self.cold_with_fm:
            # Check if cold user have any features
            user_feature = self.features_for_cold.get(user_id, None)
            if user_feature:
                return self._get_cold_reco(
                    user_feature=user_feature, k_recs=k_recs
                )
        # If not the case, let the popular model to make recos
        return None
