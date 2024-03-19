# Basic configuration for any chain in jupyter
import sys
sys.path.append("C:\\Users\\lauth\\OneDrive\\Desktop\\open_ai_assistant_v2")
from src.settings.settings import Settings
from langchain_openai import OpenAI
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate

settings = Settings()
openai_api_key = settings.openai.api_key

llm = OpenAI(
    temperature=0,
    api_key=openai_api_key,
    verbose=True,
)

prompt = settings.chain_templates.classifier_chain_template

classifier_chain = LLMChain(
    llm=llm,
    verbose=False,
    prompt=PromptTemplate.from_template(prompt),
)