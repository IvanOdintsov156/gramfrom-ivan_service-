from fastapi import APIRouter, HTTPException
from app.models.similarity_models import RecordRequest, AddDataRequest
from app.services.similarity_service import check_similarity, add_data

router = APIRouter()

@router.post("/add-data/")
async def add_data_endpoint(request: AddDataRequest):
    return await add_data(request)

@router.post("/check-similarity/")
async def check_similarity_endpoint(request: RecordRequest):
    return await check_similarity(request)