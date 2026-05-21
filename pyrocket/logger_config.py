import contextvars
from datetime import datetime, UTC
import json
import logging
import re
from starlette.requests import Request
from starlette.middleware.base import BaseHTTPMiddleware, DispatchFunction
from starlette.responses import Response

# Helper functions
def current_timestamp() -> str:
    """Returns the current timestamp in a readable format."""
    utc_time = datetime.now(UTC)
    return (
        utc_time.strftime('%Y-%m-%d %H:%M:%S') + f'.{utc_time.microsecond // 1000:03d}Z'
    )

def santize_headers(headers: dict[str, str]) -> dict[str, str]:
    """Removes sensitive information from headers."""
    headers = {k.lower(): v for k, v in headers.items()}
    if 'cookie' in headers:
        headers['cookie'] = re.sub(
            'JSEESIONID_PORTAL=[0-9A-Za-z]+',
            'JSEESIONID_PORTAL=****',
            headers['cookie']
        )
    if 'authorization' in headers:
        headers['authorization'] = headers['authorization'].split(' ')[0] + ' ****'
    return headers

# Unicorn access logs

uvicorn_context = contextvars.ContextVar('request', default=None)

class UvicornAccessContext:
    def __init__(self, request: Request):
        self.request = request
        self.request_id = request.headers.get('X-Request-ID', 'unknown')
        self.start = datetime.now(UTC)
        self.end = None

class UvicornAccessMiddleware(BaseHTTPMiddleware):
    """
    Middleware to add fields that unicorn doesn't provide by default
    """

    async def dispatch(self, request: Request, call_next: Response) -> DispatchFunction:
        context = UvicornAccessContext(request)
        uvicorn_context.set(context)
        response = await call_next(request)
        context.end = datetime.now(UTC)
        return response
    
class UvicornLogFormatter:
    # Create a tempory log formatter to get default attribtues that uvicorn provides
    _empty_record = logging.LogRecord(
        name='', level=0, pathname='', lineno=0, msg='', args=(), exc_info=None
    )
    standard_attrs = set(_empty_record.__dict__.keys())

    """
    Formatter for Uvicorn logs that properly includes extra fields
    """

    @staticmethod
    def format(record: logging.LogRecord) -> str:
        val = {
            'timestamps': current_timestamp(),
            'level': record.levelname.lower(),
            'message': record.getMessage()
        }

        # Add non-standard attributes to the output
        for key, value in record.__dict__.items():
            if key not in UvicornLogFormatter.standard_attrs:
                val[key] = value

        # Add request context if available
        context = uvicorn_context.get()
        if context:
            request = context.request
            val['remoteAddress'] = request.client[0]
            val['requestId'] = context.request_id

        return json.dumps(val, separators=(',', ':'))
    
class UvicornAccessFormatter:
    """
    Format access logs in a structured JSON format

    Uvicorn provides limit information in access logs with args in the form:
    (client_addr, method, path, http_version, status_code)
    https://github.com.encode/uvicorn/blob/master/uvicorn/protocols/http/h11_impl.py#L473-L480

    This formatter enhances logs with additional context from the
    UvicornAccessMiddleware to provide more comprehensive request information.
    """

    @staticmethod
    def format(record: logging.LogRecord) -> str:
        val = {
            'timestamps': current_timestamp(),
            'method': record.args[1],
            'uri': record.args[2],
            'protocol': f'HTTP/{record.args[3]}',
            'status': str(record.args[4]),
            'remoteAddress': record.args[0].split(':')[0],
        }

        val['remoteHost'] = val['remoteAddress']
        context = uvicorn_context.get()
        if context:
            request = context.request
            val['headers'] = santize_headers(request.headers)
            val['params'] = {k: v for k, v in request.query_params.multi_items()}
            val['serverName'] = request.headers.get('host')
            val['requestId'] = context.request_id
            val['startTimestap'] = context.start.strftime('%Y-%m-%d %H:%M:%S.%f') + 'Z'
        return json.dumps(val, separators=(',', ':'))

def configure_app_logging() -> logging.Logger:
    """
    Configures the logger for the Uvicorn server.
    """
    # Confgiure uvicorn logger
    logger = logging.getLogger('uvicorn')
    logger.handlers = []
    logger.setLevel(logging.DEBUG)
    log_handler = logging.StreamHandler()
    log_handler.setFormatter(UvicornLogFormatter())
    logger.addHandler(log_handler)

    # Configure root logger to catch unhandled exceptions
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.ERROR)
    root_handler = logging.StreamHandler()
    root_handler.setFormatter(UvicornLogFormatter())
    root_logger.addHandler(root_handler)

    return logger

def configure_access_logging() -> logging.Logger:
    """
    Configures the logger for Uvicorn access logs.
    """
    logger = logging.getLogger('uvicorn.access')
    logger.handlers = []
    log_handler = logging.StreamHandler()
    log_handler.setFormatter(UvicornAccessFormatter())
    logger.addHandler(log_handler)
    return logger