import pickle
from abc import ABC, abstractmethod
from typing import Dict, List, Optional
import numpy as np
from scipy import sparse

import dill


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
    def __init__(self, name, USER_MAPPING, ITEM_MAPPING, 
                 USERS_FEATURES, UNIQUE_FEATURES):
        super().__init__(name)
        with open(f"{name}", "rb") as f:
            self.model = dill.load(f)
        with open(USER_MAPPING, "rb") as f:
            self.user_mapping = dill.load(f)
        with open(ITEM_MAPPING, "rb") as f:
            self.item_mapping = dill.load(f)
        with open(USERS_FEATURES, "rb") as f:
            self.users_features: Dict[Dict[str, str]] = dill.load(f)
        with open(UNIQUE_FEATURES, "rb") as f:
            self.features = dill.load(f)
        
        self.items_internal_ids = np.arange(len(self.item_mapping.keys()))
    
    def _get_hot_reco(self, user_id) -> List[int]:
        iternal_user_id = self.user_mapping[user_id]
        hot_scores = self.model.predict(iternal_user_id, item_ids=self.items_internal_ids)
        idxs = np.argsort(hot_scores)[::-1]
        recs = self.items_internal_ids[idxs][:10]
        recs = [self.item_mapping[reco] for reco in recs]
        return recs
    
    def _get_cold_reco(self, user_id) -> Optional[List[int]]:
        
        if user_id not in self.users_features:
            return None
        user_feature = list(self.users_features[user_id].values())
        
        # If no features then
        if len(user_feature) == 0:
            return None

        feature_mask = np.isin(self.features, user_feature)
        feature_row = sparse.csr_matrix(feature_mask)
        
        cold_scores = self.model.predict(0, self.items_internal_ids, user_features=feature_row)
        
        idxs = np.argsort(cold_scores)[::-1]
        recs = self.items_internal_ids[idxs][:10]
        recs = [self.item_mapping[reco] for reco in recs]
        return recs
    
    def predict(self, user_id: int) -> Optional[List[int]]:
        # Check if user is hot or not
        if user_id in self.user_mapping:
            return self._get_hot_reco(user_id=user_id)
        else:
            return self._get_cold_reco(user_id=user_id)


