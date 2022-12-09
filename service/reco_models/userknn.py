from pandas import DataFrame
from scipy.sparse import coo_matrix
from numpy import log, ones, float32, array
from collections import Counter
from implicit.nearest_neighbours import ItemItemRecommender
from typing import Dict, Optional

class UserKnn_light:
    """Class for fit-perdict UserKNN model
       based on ItemKNN model from implicit.nearest_neighbours
    """
    from pandas import DataFrame
    from scipy.sparse import coo_matrix
    from numpy import log, ones, float32, array
    from collections import Counter
    from implicit.nearest_neighbours import ItemItemRecommender
    from typing import Dict, Optional

    def __init__(self, model: ItemItemRecommender, N_users: int = 50):
        self.N_users = N_users
        self.model = model
        self.is_fitted = False

    def get_mappings(self, train):
        self.users_inv_mapping = dict(enumerate(train['user_id'].unique()))
        self.users_mapping = {v: k for k, v in self.users_inv_mapping.items()}

        self.items_inv_mapping = dict(enumerate(train['item_id'].unique()))
        self.items_mapping = {v: k for k, v in self.items_inv_mapping.items()}

    def get_matrix(self, df: DataFrame,
                   user_col: str = 'user_id',
                   item_col: str = 'item_id',
                   weight_col: str = None,
                   users_mapping: Dict[int, int] = None,
                   items_mapping: Dict[int, int] = None):
        from scipy.sparse import coo_matrix
        from numpy import ones, float32

        if weight_col:
            weights = df[weight_col].astype(float32)
        else:
            weights = ones(len(df), dtype=float32)

        interaction_matrix = coo_matrix((
            weights,
            (
                df[user_col].map(self.users_mapping.get),
                df[item_col].map(self.items_mapping.get)
            )
        ))

        self.watched = df.groupby(user_col).agg({item_col: tuple})
        return interaction_matrix

    def idf(self, n: int, x: float):
        from numpy import log
        return log((1 + n) / (1 + x) + 1)

    def _count_item_idf(self, df: DataFrame):
        from collections import Counter
        from pandas import DataFrame
        item_cnt = Counter(df['item_id'].values)
        item_idf = DataFrame.from_dict(item_cnt, orient='index',
                                       columns=['doc_freq']).reset_index()
        item_idf['idf'] = item_idf['doc_freq'].apply(
            lambda x: self.idf(self.n, x))
        self.item_idf = item_idf

    def fit(self, train: DataFrame, weight_col: Optional[
        str] = None):  # добавил аргумент weight_col = None, теперь можно выбирать что есть вес
        self.user_knn = self.model
        self.get_mappings(train)
        # 1) Тут не был задан параметр weight_col=
        self.weights_matrix = self.get_matrix(train, weight_col=weight_col,
                                              users_mapping=self.users_mapping,
                                              items_mapping=self.items_mapping)

        self.n = train.shape[0]

        # 2) (не получилось) Можно добавить bm25 или придумать свой способ переранжирования
        self._count_item_idf(train)

        self.user_knn.fit(self.weights_matrix)
        self.is_fitted = True

    def _get_similar_users(self, user_id: int, model: ItemItemRecommender,
                           user_mapping: Dict[int, int],
                           user_inv_mapping: Dict[int, int],
                           N_users: int) -> array:
        from numpy import array
        internal_user_id = user_mapping[user_id]
        sim_users = array(model.similar_items(internal_user_id, N=N_users))[:,
                    0]
        return sim_users

    def predict(self, user_id: int, N_recs: int = 10):
        if not self.is_fitted:
            raise ValueError("Please call fit before predict")

        from pandas import DataFrame

        sim_users = self._get_similar_users(
            user_id=user_id,
            model=self.user_knn,
            user_mapping=self.users_mapping,
            user_inv_mapping=self.users_inv_mapping,
            N_users=self.N_users
        )

        recs = DataFrame({
            "user_id": user_id,
            "sim_user_id": sim_users,
        })

        recs = recs.merge(self.watched, left_on=['sim_user_id'],
                          right_on=['user_id'], how='left') \
            .drop(["sim_user_id"], axis=1) \
            .explode('item_id') \
            .drop_duplicates(['item_id'], keep='first') \
            .merge(self.watched, left_on=["user_id"], right_on=["user_id"],
                   how="left")

        # исключаем уже просмотренное тестовым пользователем из рекомендаций
        recs = recs[
            recs.apply(lambda x: x["item_id_x"] not in x["item_id_y"], axis=1)] \
            .drop(["item_id_y"], axis=1) \
            .merge(self.item_idf, left_on='item_id_x', right_on='index',
                   how='left')

        recs = recs.sort_values(['idf'], ascending=True)
        return recs["item_id_x"][:N_recs].tolist()
