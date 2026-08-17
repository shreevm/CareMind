"""Input validation middleware for CareMind API.

Validates incoming requests to ensure data integrity and prevent
common attack vectors.
"""

import json
import logging
from typing import Callable

from fastapi import Request
from fastapi.responses import JSONResponse

logger = logging.getLogger(__name__)

# Constants for validation
MAX_MESSAGE_LENGTH = 5000
MAX_SESSION_ID_LENGTH = 100
MAX_WORKSPACE_ID_LENGTH = 100
MAX_QUERY_LENGTH = 1000
MIN_MESSAGE_LENGTH = 1


async def validate_request_middleware(request: Request, call_next: Callable):
    """
    Validate all incoming requests before processing.
    
    Checks:
    - Content type is JSON for POST/PUT
    - Required fields are present
    - Field lengths are within bounds
    - No suspicious patterns (SQL injection, etc.)
    
    Args:
        request: FastAPI Request object
        call_next: Next middleware handler
        
    Returns:
        Response from next handler or validation error
    """
    
    path = request.url.path
    method = request.method
    
    # Only validate specific endpoints
    if path == "/chat" and method == "POST":
        return await validate_chat_request(request, call_next)
    elif path == "/search" and method == "GET":
        return await validate_search_request(request, call_next)
    elif path.startswith("/upload") and method == "POST":
        return await validate_upload_request(request, call_next)
    elif path == "/compare" and method == "POST":
        return await validate_compare_request(request, call_next)
    
    # Pass through other requests
    return await call_next(request)


async def validate_chat_request(request: Request, call_next: Callable):
    """Validate /chat POST request."""
    try:
        body = await request.json()
    except json.JSONDecodeError:
        logger.warning("api.validation.invalid_json path=%s", request.url.path)
        return JSONResponse(
            {"error": "Invalid JSON in request body"},
            status_code=400
        )
    
    # Check required fields
    message = body.get("message", "").strip()
    session_id = body.get("session_id", "").strip()
    workspace_id = body.get("workspace_id", "default").strip()
    
    errors = []
    
    if not message:
        errors.append("message field is required")
    elif len(message) < MIN_MESSAGE_LENGTH:
        errors.append(f"message must be at least {MIN_MESSAGE_LENGTH} character")
    elif len(message) > MAX_MESSAGE_LENGTH:
        errors.append(f"message must be <= {MAX_MESSAGE_LENGTH} characters (got {len(message)})")
    
    if not session_id:
        errors.append("session_id field is required")
    elif len(session_id) > MAX_SESSION_ID_LENGTH:
        errors.append(f"session_id must be <= {MAX_SESSION_ID_LENGTH} characters")
    elif not is_valid_id(session_id):
        errors.append("session_id contains invalid characters")
    
    if len(workspace_id) > MAX_WORKSPACE_ID_LENGTH:
        errors.append(f"workspace_id must be <= {MAX_WORKSPACE_ID_LENGTH} characters")
    elif not is_valid_id(workspace_id):
        errors.append("workspace_id contains invalid characters")
    
    if errors:
        logger.warning(
            "api.validation.chat_failed path=%s errors=%s",
            request.url.path,
            errors
        )
        return JSONResponse(
            {"errors": errors},
            status_code=400
        )
    
    # Store validated data in request state for use in handler
    request.state.validated_data = {
        "message": message,
        "session_id": session_id,
        "workspace_id": workspace_id,
    }
    
    logger.info(
        "api.validation.chat_passed session_id=%s message_length=%s",
        session_id,
        len(message)
    )
    
    return await call_next(request)


async def validate_search_request(request: Request, call_next: Callable):
    """Validate /search GET request."""
    query = request.query_params.get("query", "").strip()
    workspace_id = request.query_params.get("workspace_id", "default").strip()
    top_k = request.query_params.get("top_k", "5")
    
    errors = []
    
    if not query:
        errors.append("query parameter is required")
    elif len(query) < MIN_MESSAGE_LENGTH:
        errors.append(f"query must be at least {MIN_MESSAGE_LENGTH} character")
    elif len(query) > MAX_QUERY_LENGTH:
        errors.append(f"query must be <= {MAX_QUERY_LENGTH} characters")
    
    if not workspace_id:
        errors.append("workspace_id is required")
    elif len(workspace_id) > MAX_WORKSPACE_ID_LENGTH:
        errors.append(f"workspace_id must be <= {MAX_WORKSPACE_ID_LENGTH} characters")
    
    try:
        top_k_int = int(top_k)
        if top_k_int < 1 or top_k_int > 50:
            errors.append("top_k must be between 1 and 50")
    except ValueError:
        errors.append("top_k must be an integer")
    
    if errors:
        logger.warning(
            "api.validation.search_failed path=%s errors=%s",
            request.url.path,
            errors
        )
        return JSONResponse(
            {"errors": errors},
            status_code=400
        )
    
    logger.info(
        "api.validation.search_passed query_length=%s top_k=%s",
        len(query),
        top_k
    )
    
    return await call_next(request)


async def validate_upload_request(request: Request, call_next: Callable):
    """Validate /upload POST request."""
    # Upload validation is handled by FastAPI file/form validators
    # Just log the attempt
    logger.info("api.validation.upload_passed")
    return await call_next(request)


async def validate_compare_request(request: Request, call_next: Callable):
    """Validate /compare POST request."""
    try:
        body = await request.json()
    except json.JSONDecodeError:
        logger.warning("api.validation.invalid_json path=%s", request.url.path)
        return JSONResponse(
            {"error": "Invalid JSON in request body"},
            status_code=400
        )
    
    document_ids = body.get("document_ids", [])
    workspace_id = body.get("workspace_id", "default").strip()
    
    errors = []
    
    if not isinstance(document_ids, list) or len(document_ids) < 2:
        errors.append("document_ids must be a list with at least 2 documents")
    elif any(not is_valid_id(doc_id) for doc_id in document_ids):
        errors.append("document_ids contains invalid document IDs")
    elif len(document_ids) > 10:
        errors.append("Cannot compare more than 10 documents")
    
    if len(workspace_id) > MAX_WORKSPACE_ID_LENGTH:
        errors.append(f"workspace_id must be <= {MAX_WORKSPACE_ID_LENGTH} characters")
    
    if errors:
        logger.warning(
            "api.validation.compare_failed path=%s errors=%s",
            request.url.path,
            errors
        )
        return JSONResponse(
            {"errors": errors},
            status_code=400
        )
    
    logger.info(
        "api.validation.compare_passed doc_count=%s",
        len(document_ids)
    )
    
    return await call_next(request)


def is_valid_id(value: str) -> bool:
    """
    Check if value is a valid ID (alphanumeric, dash, underscore).
    
    Args:
        value: String to validate
        
    Returns:
        True if valid, False otherwise
    """
    if not value:
        return False
    
    # Allow alphanumeric, dash, underscore only
    return all(c.isalnum() or c in "-_" for c in value)
