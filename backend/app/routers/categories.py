import json
from fastapi import APIRouter

router = APIRouter()

@router.get("/categories")
def categories():
    with open("phase4_hierarchy_map.json") as f:
        return json.load(f)
