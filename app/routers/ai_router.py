
from fastapi import APIRouter, HTTPException
from app.schema.MessageTransform import MessageTransformRequest, MessageTransformResponse
from app.service import transfer_service


router = APIRouter(prefix="/ai", tags=["AI"])


@router.post("/transform", response_model=MessageTransformResponse)
async def transform_message(request: MessageTransformRequest):
    # try:
    result = await transfer_service.transform_message(request)
    return result
    # except Exception as e:
    #     raise HTTPException(status_code=500, detail=str(e))