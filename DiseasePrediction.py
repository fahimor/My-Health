
from pydantic import BaseModel
# 2. Class which describes Bank Notes measurements
class DiseasePredict(BaseModel):
    sym1: int 
    sym2: int 
    sym3: int 
    sym4: int
    sym5: int