import random
from typing import Any, Dict, List

from fastapi import APIRouter, Depends, FastAPI, Request
from fastapi.security import HTTPBearer
from fastapi.security.http import HTTPAuthorizationCredentials
from pydantic import BaseModel

from service.api.exceptions import (
    BearerAccessTokenError,
    ModelNotFoundError,
    UserNotFoundError,
)
from service.api.responses import (
    AuthorizationResponse,
    ForbiddenResponse,
    NotFoundError,
)
from service.log import app_logger
from service.reco_models import als_model, models


class RecoResponse(BaseModel):
    user_id: int
    items: List[int]


class ExplainResponse(BaseModel):
    p: int
    explanation: str


bearer_scheme = HTTPBearer()

router = APIRouter()

responses: Dict[str, Any] = {
    "401": AuthorizationResponse().get_response(),
    "403": ForbiddenResponse().get_response(),
    "404": NotFoundError().get_response(),
}


@router.get(
    path="/health",
    tags=["Health"],
)
async def health() -> str:
    return "I am alive"


@router.get(
    path="/reco/{model_name}/{user_id}",
    tags=["Recommendations"],
    response_model=RecoResponse,
    responses=responses,  # type: ignore
)
async def get_reco(
    request: Request,
    model_name: str,
    user_id: int,
    token: HTTPAuthorizationCredentials = Depends(bearer_scheme),
) -> RecoResponse:
    app_logger.info(f"Request for model: {model_name}, user_id: {user_id}")

    if token.credentials != "Team_5":
        raise BearerAccessTokenError()
    if user_id > 10**9:
        raise UserNotFoundError(error_message=f"User {user_id} not found")

    k_recs = request.app.state.k_recs

    if model_name in models.keys():
        reco = models[model_name].predict(user_id, k_recs=k_recs)
    else:
        raise ModelNotFoundError(error_message=f"Model {model_name} not found")

    if not reco:
        reco = models["baseline"].predict(user_id, k_recs=k_recs)
    return RecoResponse(user_id=user_id, items=reco)


@router.get(
    path="/explain/{model_name}/{user_id}/{item_id}",
    tags=["Explanations"],
    response_model=ExplainResponse,
)
async def explain(request: Request, model_name: str, user_id: int, item_id: int) -> ExplainResponse:
    """
    Пользователь переходит на карточку контента, на которой нужно показать
    процент релевантности этого контента зашедшему пользователю,
    а также текстовое объяснение почему ему может понравится этот контент.

    :param request: запрос.
    :param model_name: название модели, для которой нужно получить объяснения.
    :param user_id: id пользователя, для которого нужны объяснения.
    :param item_id: id контента, для которого нужны объяснения.
    :return: Response со значением процента релевантности и текстовым объяснением, понятным пользователю.
    - "p": "процент релевантности контента item_id для пользователя user_id"
    - "explanation": "текстовое объяснение почему рекомендован item_id"
    """
    p, explanation = als_model.explain(user_id, item_id)
    if p is None:
        return ExplainResponse(p=random.randint(50, 80), explanation="Вам может понравится")
    return ExplainResponse(p=p, explanation=explanation)


def add_views(app: FastAPI) -> None:
    app.include_router(router)
