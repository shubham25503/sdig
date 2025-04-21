from pydantic import BaseModel
from typing import List

class ImageGenRequest(BaseModel):
    injection_number: int
    selected_areas: List[str]
