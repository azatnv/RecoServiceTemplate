from typing import Any, Dict

from service.models import Error, ErrorResponse


class BasicErrorResponse:

    def __init__(self) -> None:
        self.model = ErrorResponse
        self.description: str = "None"
        self.content: Dict[Any, Any] = {
            "application/json": {}
        }

    def get_response(self) -> Dict[Any, Any]:
        "Return response of the Error"
        response: Dict[Any, Any] = {
            "model": self.model,
            "description": self.description,
            "content": self.content,
        }
        return response


class AuthorizationResponse(BasicErrorResponse):

    def __init__(self):
        super().__init__()
        self.description: str = "Error: Unauthorized"
        self.content: Dict[Any, Any] = {
            "application/json": {
                "example": ErrorResponse(
                    errors=[
                        Error(
                            error_key="incorrect_bearer_key",
                            error_message=(
                                "Authorization failure due to incorrect token"
                                ),
                            error_loc=None,
                        )
                    ]
                )
            }
        }

    def get_response(self) -> Dict[Any, Any]:
        response = super().get_response()
        return response


class ForbiddenResponse(BasicErrorResponse):

    def __init__(self):
        super().__init__()
        self.description: str = "Error: Forbidden"
        self.content: Any = {
            "application/json": {
                "example": ErrorResponse(
                    errors=[
                        Error(
                            error_key="http_exception",
                            error_message="Not authenticated",
                            error_loc=None,
                        )
                    ]
                )
            }
        }


class NotFoundError(BasicErrorResponse):

    def __init__(self):
        super().__init__()
        self.description: str = "Error: Not Found"
        self.content: Any = {
            "application/json": {
                "examples": {
                    "example_1": {
                        "summary": "Model error",
                        "value": ErrorResponse(
                            errors=[
                                Error(
                                    error_key="model_not_found",
                                    error_message="Model is unknown",
                                    error_loc=None,
                                )
                            ]
                        ),
                    },
                    "example_2": {
                        "summary": "User error",
                        "value": ErrorResponse(
                            errors=[
                                Error(
                                    error_key="user_not_found",
                                    error_message="User is unknown",
                                    error_loc=None,
                                )
                            ]
                        ),
                    },
                }
            },
        }
