from fastapi import APIRouter, HTTPException, UploadFile, File
from pydantic import BaseModel
from typing import Optional, Literal

from app.services.model_service import (
    InvalidImageError,
    ModelInitializationError,
    NoHandDetectedError,
    predict_from_base64,
    predict_from_image,
    get_model_status,
)
from app.services.dynamic_model_service import (
    DynamicInvalidImageError,
    DynamicModelInitializationError,
    get_dynamic_model_status,
    predict_dynamic_from_image,
    reset_dynamic_session,
)
from app.schemas.user_schema import (
    DynamicPredictionResponse,
    ModelStatusResponse,
    PredictionResponse,
)

router = APIRouter()


class SentenceActionRequest(BaseModel):
    mode: Literal["static", "dynamic"]
    session_id: Optional[str] = None
    token: Optional[str] = None


class SentenceStateResponse(BaseModel):
    mode: Literal["static", "dynamic"]
    session_id: str
    sentence: str


_sentence_tokens: dict[str, dict[str, list[str]]] = {
    "static": {},
    "dynamic": {},
}

_DYNAMIC_TOKEN_ALIASES = {
    "ily": "i love you",
}


def _resolve_session_id(raw: Optional[str]) -> str:
    cleaned = (raw or "").strip()
    return cleaned or "default"


def _normalize_dynamic_token(token: str) -> str:
    normalized = token.strip().lower().replace("_", " ")
    if not normalized:
        return ""
    return _DYNAMIC_TOKEN_ALIASES.get(normalized, normalized)


def _render_sentence(mode: Literal["static", "dynamic"], tokens: list[str]) -> str:
    if mode == "static":
        return "".join(tokens)
    return " ".join(piece for piece in tokens if piece).strip()


def _session_tokens(mode: Literal["static", "dynamic"], session_id: str) -> list[str]:
    return _sentence_tokens[mode].setdefault(session_id, [])


@router.get("/model/status", response_model=ModelStatusResponse)
async def model_status(mode: str = "static"):
    """
    Returns the current state of the ASL inference model.
    - model_loaded: True if the model and hand detector are initialised.
    - is_trained_weights: True when real trained .pth/.pth.zip weights are active (not a random placeholder).
    - checkpoint_source: filename of the checkpoint that was loaded.
    """
    if mode == "dynamic":
        return get_dynamic_model_status()
    return get_model_status()


class Base64PredictRequest(BaseModel):
    image: str          # base64 encoded image (with or without data-URI prefix)
    user_id: Optional[int] = None


class DynamicPredictRequest(BaseModel):
    session_id: Optional[str] = None


@router.post("/predict", response_model=PredictionResponse)
async def predict(file: UploadFile = File(...)):
    # Validate the uploaded file before sending it into the ML pipeline.
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Invalid file. Please upload an image.")

    image_bytes = await file.read()
    try:
        label, confidence = predict_from_image(image_bytes)
    except InvalidImageError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except NoHandDetectedError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except ModelInitializationError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    # Return the predicted ASL class label for the uploaded frame.
    return PredictionResponse(prediction=label, confidence=confidence)


@router.post("/predict/base64", response_model=PredictionResponse)
async def predict_base64(body: Base64PredictRequest):
    if not body.image:
        raise HTTPException(status_code=400, detail="image field is required")

    try:
        label, confidence = predict_from_base64(body.image)
    except InvalidImageError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except NoHandDetectedError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except ModelInitializationError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    return PredictionResponse(prediction=label, confidence=confidence)


@router.post("/predict/dynamic", response_model=DynamicPredictionResponse)
async def predict_dynamic(
    file: UploadFile = File(...),
    session_id: str = "default",
):
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Invalid file. Please upload an image.")

    image_bytes = await file.read()
    try:
        result = predict_dynamic_from_image(image_bytes, session_id=session_id)
    except DynamicInvalidImageError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except DynamicModelInitializationError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    return DynamicPredictionResponse(**result)


@router.post("/predict/dynamic/reset")
async def reset_dynamic_predict_session(body: DynamicPredictRequest):
    session_id = body.session_id or "default"
    reset_dynamic_session(session_id)
    return {"ok": True, "session_id": session_id}


@router.post("/predict/sentence/enter", response_model=SentenceStateResponse)
async def sentence_enter(body: SentenceActionRequest):
    session_id = _resolve_session_id(body.session_id)
    token = (body.token or "").strip()
    if not token:
        raise HTTPException(status_code=400, detail="token is required for enter action")

    tokens = _session_tokens(body.mode, session_id)
    if body.mode == "static":
        if token.lower() == "space":
            tokens.append(" ")
        else:
            tokens.append(token)
    else:
        normalized = _normalize_dynamic_token(token)
        if not normalized:
            raise HTTPException(status_code=400, detail="token is empty after normalization")
        tokens.append(normalized)

    return SentenceStateResponse(
        mode=body.mode,
        session_id=session_id,
        sentence=_render_sentence(body.mode, tokens),
    )


@router.post("/predict/sentence/delete", response_model=SentenceStateResponse)
async def sentence_delete(body: SentenceActionRequest):
    session_id = _resolve_session_id(body.session_id)
    tokens = _session_tokens(body.mode, session_id)
    if tokens:
        tokens.pop()

    return SentenceStateResponse(
        mode=body.mode,
        session_id=session_id,
        sentence=_render_sentence(body.mode, tokens),
    )


@router.post("/predict/sentence/clear", response_model=SentenceStateResponse)
async def sentence_clear(body: SentenceActionRequest):
    session_id = _resolve_session_id(body.session_id)
    _sentence_tokens[body.mode][session_id] = []

    return SentenceStateResponse(
        mode=body.mode,
        session_id=session_id,
        sentence="",
    )
