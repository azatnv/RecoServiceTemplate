from typing import List

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
from service.reco_models.reco_models import simple_popular_model

popular_model = simple_popular_model()  # type: ignore


class RecoResponse(BaseModel):
    user_id: int
    items: List[int]


bearer_scheme = HTTPBearer()

router = APIRouter()

responses = {
    '401': AuthorizationResponse().get_response(),   # type: ignore
    '403': ForbiddenResponse().get_response(),       # type: ignore
    '404': NotFoundError().get_response()            # type: ignore
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
    if model_name != "test_model":
        raise ModelNotFoundError(error_message=f"Model {model_name} not found")

    k_recs = request.app.state.k_recs

    try:
        reco = popular_model.get_popular_reco(user_id, k_recs)
        # reco = list(range(k_recs))
    except TypeError:
        reco = list(range(k_recs))
    return RecoResponse(user_id=user_id, items=reco)


def add_views(app: FastAPI) -> None:
    app.include_router(router)
