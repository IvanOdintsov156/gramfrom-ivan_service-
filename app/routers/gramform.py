from fastapi import APIRouter, HTTPException
from app.models.gramform_model import CheckRequest, CheckResponse
from app.services.gramform_service import check_name

router = APIRouter()

@router.post("/check_name")
async def check_name_endpoint(request: CheckRequest):
    return await check_name(request)

