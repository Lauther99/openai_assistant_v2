import sys
sys.path.append("C:\\Users\\lauth\\OneDrive\\Desktop\\open_ai_assistant_v2")
from src.components.llms import MODEL_TYPE
from langchain_openai import OpenAI as OpenAI_From_Langchain
from src.settings.settings import Settings
from langchain_openai.llms.base import BaseOpenAI
from langchain_openai import OpenAIEmbeddings
from langchain_core.embeddings import Embeddings
from openai import OpenAI as OpenAI_From_OpenAILibrary
from typing import Union

class LLMs:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings

        self.openai_eco_gpt = OpenAI_From_Langchain(
            model=settings.openai.eco_model,
            temperature=0,
            api_key=settings.openai.api_key,
            verbose=True,
            max_tokens=-1
        )
        
        self.openai_supper_gpt = OpenAI_From_Langchain(
            model=settings.openai.super_model,
            temperature=0,
            api_key=settings.openai.api_key,
            verbose=True,
        )
        
        self.openai_embeddings = OpenAIEmbeddings(openai_api_key=settings.openai.api_key,)
        
        self.native_openai = OpenAI_From_OpenAILibrary(
            api_key=settings.openai.api_key
        )
                
    def get_llm(self, llm_type: MODEL_TYPE) -> BaseOpenAI:
        '''Para obtener llms nativos de la libreria de Langchain'''
         
        if llm_type == "gpt-3.5-turbo-instruct":
            return self.openai_eco_gpt
        elif llm_type == "gpt-4":
            return self.openai_supper_gpt
        elif llm_type == "text-embedding-ada-002":
            return self.openai_embeddings
        else:
            raise ValueError(f"Not support model type: {llm_type}")
        
    def get_native_llm(self, llm_type: MODEL_TYPE) -> OpenAI_From_OpenAILibrary:
        '''Para obtener llms nativos de la libreria de OpenAI'''
        
        if llm_type == "native-openai":
            return self.native_openai
        else:
            raise ValueError(f"Not support model type: {llm_type}")
    
    def get_embeddings_llm(self, llm_type: MODEL_TYPE) -> Embeddings:
        if llm_type == "text-embedding-ada-002":
            return self.openai_embeddings
        else:
            raise ValueError(f"Not support model type: {llm_type}")