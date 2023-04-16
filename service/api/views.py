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
from service.reco_models import models


class RecoResponse(BaseModel):
    user_id: int
    items: List[int]


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

    reco = None
    if model_name in models.keys():
        reco = models["model_name"].predict(user_id, k_recs=k_recs)
    else:
        raise ModelNotFoundError(error_message=f"Model {model_name} not found")

    # if model_name in ("light_fm_1", "light_fm_2"):
    #     reco = (
    #         online_fm_all_popular.predict(user_id, k_recs)
    #         if model_name == "light_fm_1"
    #         else online_fm_part_popular.predict(user_id, k_recs)
    #     )
    # if model_name == "ann_lightfm":
    #     reco = ann_lightfm.predict(user_id)

    if not reco:
        reco = models["baseline"].predict(user_id, k_recs=k_recs)
    return RecoResponse(user_id=user_id, items=reco)


def add_views(app: FastAPI) -> None:
    app.include_router(router)
