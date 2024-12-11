from pydantic import BaseModel

class CheckRequest(BaseModel):
    name: str

class CheckResponse(BaseModel):
    corrected_name: str
    gram_morph_rule: bool
    first_noun_rule: bool
    description: dict
    