import os
import pickle
from typing import List, Dict

cwd = os.path.dirname(__file__)


class simple_popular_model():

    def __init__(self):
        self.users_dictionary = pickle.load(
            open(
                os.path.join(
                    cwd,
                    'users_dictionary.pickle'
                ),
                'rb'
            )
        )
        self.popular_dictionary: Dict = pickle.load(
            open(
                os.path.join(
                    cwd,
                    'popular_dictionary.pickle'
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
        
        try:
            category = self.users_dictionary[user_id]
            reco = self.popular_dictionary[category][:k_recs]
        except KeyError:
            # если юзер не принадлежит никакой категории, рекомендуем
            # ему популярное в среднем
            reco = self.popular_dictionary['popular_for_all'][:k_recs]
        return reco
