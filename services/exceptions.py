class ServiceError(Exception):
    def __init__(
        self,
        message: str,
        *,
        code: str = "service_error",
        status_code: int = 400,
        details: list[str] | None = None,
    ):
        super().__init__(message)
        self.message = message
        self.code = code
        self.status_code = status_code
        self.details = details or []


class RetrievalError(ServiceError):
    def __init__(
        self,
        message: str,
        *,
        code: str = "retrieval_error",
        status_code: int = 503,
        details: list[str] | None = None,
    ):
        super().__init__(
            message,
            code=code,
            status_code=status_code,
            details=details,
        )
