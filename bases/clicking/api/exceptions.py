from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

class ServiceNotAvailableException(Exception):
    def __init__(self, msg="Service is not available", status_code=500):
        self.msg = msg
        print(f"ServiceNotAvailableException: {msg}")
        self.status_code = status_code

class ModelNotSetException(ServiceNotAvailableException):
    def __init__(self, msg="Model is not set", status_code=500):
        print(f"ModelNotSetException: {msg}")
        super().__init__(msg, status_code)

def add_exception_handlers(app: FastAPI):
    @app.exception_handler(ServiceNotAvailableException)
    async def service_exception_handler(request: Request, exc: ServiceNotAvailableException):
        return JSONResponse(
            status_code=exc.status_code,
            content={"message": exc.msg},
        )

    # ModelNotSetException will be handled by the same handler
    # since it's a subclass of ServiceNotAvailableException