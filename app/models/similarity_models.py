from pydantic import BaseModel

class RecordRequest(BaseModel):
    id: str
    name: str

class AddDataRequest(BaseModel):
    items: list