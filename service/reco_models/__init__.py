from typing import Dict

from ..configuration import (
    OFFLINE_KNN_MODEL_PATH,
    ONLINE_KNN_MODEL_PATH,
    POPULAR_IN_CATEGORY,
    POPULAR_MODEL_RECS,
    POPULAR_MODEL_USERS,
)
from .knn_models import OfflineKnnModel, OnlineKnnModel
from .model import RecommendationModel
# from .lightfm_models import ANNLightFM, OnlineFM
from .popular_models import PopularInCategory, SimplePopularModel, TestModel

__all__ = ("models",)

test_model = TestModel()

baseline_model = PopularInCategory(POPULAR_IN_CATEGORY)
popular_model = SimplePopularModel(
    POPULAR_MODEL_USERS,
    POPULAR_MODEL_RECS,
)

offline_knn_model = OfflineKnnModel(OFFLINE_KNN_MODEL_PATH)
online_knn_model = OnlineKnnModel(ONLINE_KNN_MODEL_PATH)

models: Dict[str, RecommendationModel] = {
    "test_model": test_model,
    "baseline": baseline_model,
    "popular_model": popular_model,
    "offline_knn": offline_knn_model,
    "online_knn": online_knn_model,
    # "light_fm_1"
    # "light_fm_2"
    # "ann_lightfm"
}

# ----------------------------------------------------------
#                Temporary commented out
# ----------------------------------------------------------
# ANNLightFM,; OnlineFM,

# ANN_PATHS,; FEATURES_FOR_COLD,; ITEM_MAPPING,; LIGHT_FM,; UNIQUE_FEATURES,; USER_MAPPING,

# Use LightFM model to predict recos for cold with features,
# popular for others
# online_fm_part_popular = OnlineFM(
#     name=LIGHT_FM,
#     USER_MAPPING=USER_MAPPING,
#     ITEM_MAPPING=ITEM_MAPPING,
#     FEATURES_FOR_COLD=FEATURES_FOR_COLD,
#     UNIQUE_FEATURES=UNIQUE_FEATURES,
# )
# #  Use popular model to predict recos for all cold
# online_fm_all_popular = OnlineFM(
#     name=LIGHT_FM,
#     cold_with_fm=False,
#     USER_MAPPING=USER_MAPPING,
#     ITEM_MAPPING=ITEM_MAPPING,
#     FEATURES_FOR_COLD=FEATURES_FOR_COLD,
#     UNIQUE_FEATURES=UNIQUE_FEATURES,
# )
# ann_lightfm = ANNLightFM(ANN_PATHS, popular_model)
