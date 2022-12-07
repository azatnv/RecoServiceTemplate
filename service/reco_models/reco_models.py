import os
import pickle
from typing import Any, List

cwd = os.path.dirname(__file__)


class simple_popular_model():

    def __init__(self):
        self.dataset = pickle.load(
            open(
                os.path.join(
                    cwd,
                    'popular_dataset.pickle'
                ),
                'rb'
            )
        )
        self.model: Any = pickle.load(
            open(
                os.path.join(
                    cwd,
                    'popular_model.pickle'
                ),
                'rb'
            )
        )
        self.top_popular_list: List = pickle.load(
            open(
                os.path.join(
                    cwd,
                    'top_popular_list.pickle',
                ),
                'rb'
            )
        )

    def get_popular_reco(
        self,
        user_id: int,
        k_recs: int
    ) -> List:
        # проверяю юзер в датасете или нет
        if user_id in self.dataset.user_id_map.external_ids:
            recos = list(
                self.model.recommend(
                    [user_id],
                    dataset=self.dataset,
                    k=k_recs,
                    filter_viewed=False
                )['item_id'].values
            )
        else:
            recos = list(
                self.top_popular_list[:k_recs]
            )
        return recos
