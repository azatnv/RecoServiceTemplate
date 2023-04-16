from abc import ABC, abstractmethod
from typing import List, Optional


class RecommendationModel(ABC):
    @abstractmethod
    def predict(self, user_id: int, k_recs: int) -> Optional[List[int]]:
        pass
