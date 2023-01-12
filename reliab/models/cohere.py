
from typing import Any, Dict, List
from reliab.utils import get_from_dict_or_env
from reliab.models.model import GenModel
from pydantic import BaseModel, root_validator

class Cohere(BaseModel):
    
    client : Any
 
    @root_validator()
    def validate_environment(cls, values: Dict) -> Dict:
        """Validate that api key and python package exists in environment."""
        cohere_api_key = get_from_dict_or_env(
            values, "cohere_api_key", "COHERE_API_KEY"
        )
        try:
            import cohere

            values["client"] = cohere.Client(cohere_api_key)
        except ImportError:
            raise ValueError(
                "Could not import openai python package. "
                "Please it install it with `pip install openai`."
            )
        return values


class TextCompletion(GenModel, Cohere):    
        
    model: str
        
    def _run(self, prompts: List[str], **kwargs) -> GenModel.ModelResult:
        responses = []
        generations = []
        for prompt in prompts:
            response = self.client.generate(prompt=prompt, **self.dict(), **kwargs)
            generation = [generation.text for generation in response.generations]
            responses.append(response)
            generations.append(generation)
        return GenModel.ModelResult(
            model=self,
            response=responses,
            generations=generations,
        )