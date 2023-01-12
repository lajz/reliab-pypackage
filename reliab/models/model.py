
from typing import Any, List, Optional, Tuple
from abc import ABC, abstractmethod
from pydantic import BaseModel
from pydantic.dataclasses import dataclass



class Model(BaseModel, ABC):
    
    @dataclass
    class ModelResult():
        model: Optional['GenModel']
        responses: Optional[List[dict]]
    
    @abstractmethod
    def _run(
        self,
        **kwargs
    ) -> ModelResult:
        """Run the model with the given arguments."""

    
# Text model
class GenModel(Model):
    
    @dataclass
    class ModelResult(Model.ModelResult):
        model: Optional['GenModel']
        
        generations: List[List[str]]

        
    @abstractmethod
    def run(self, prompts: List[str], **kwargs) -> ModelResult:
        pass
