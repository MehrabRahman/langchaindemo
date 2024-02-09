import requests
from typing import Any, List, Mapping, Optional

from langchain_core.callbacks.manager import CallbackManagerForLLMRun
from langchain_core.language_models.llms import LLM

class CustomLLM(LLM):

    #n: int

    @property
    def _llm_type(self) -> str:
        return "custom"

    def _call(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> str:
        #import requests
        #make a API call to model on AWS
        #Retrieve JSON
        #Return string of output
        payload = {
            "inputs": prompt,
            "parameters": { #Try and experiment with the parameters
                "max_new_tokens": 1024,
                "temperature": 0.6,
                "top_p": 0.9,
                "do_sample": False,
                "return_full_text": False
            }
        }
        API_TOKEN = ""
        API_URL = ""
        headers = {"Authorization": f"Bearer {API_TOKEN}"}
        response = requests.post(API_URL, headers=headers, json=payload)
        #print(response.json())
        return response.json()[0]["generated_text"]

    #@property
    #def _identifying_params(self) -> Mapping[str, Any]:
        #"""Get the identifying parameters."""
        #return {"n": self.n}