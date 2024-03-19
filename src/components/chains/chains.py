import sys

sys.path.append("C:\\Users\\lauth\\OneDrive\\Desktop\\open_ai_assistant_v2")
from src.settings.settings import Settings
from src.components.llms.llms import LLMs
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain_core.embeddings import Embeddings
from langchain_openai.llms.base import BaseOpenAI
from src.db.helpers.helpers import find_db_examples
from src.tools.vanna.vanna_tool import VannaTool
from langchain.memory import ConversationSummaryBufferMemory
import uuid
import json
import pandas as pd


class Assistant:
    def __init__(self, **kwargs: dict[str, str]) -> None:
        self.settings = Settings()
        self.llms = LLMs(self.settings)

        self.conversation_id = kwargs.get("conversation_id", str(uuid.uuid4()))

        self.chain_llm = self.llms.get_llm(llm_type=self.settings.openai.eco_model)
        self.embeddings_llm = self.llms.get_llm(
            llm_type=self.settings.openai.embeddings_model
        )

        self.memory = self._init_memory()

    def _init_memory(self) -> ConversationSummaryBufferMemory:
        conversation_id = self.conversation_id
        memory_prompt = PromptTemplate(
            input_variables=["summary", "new_lines"],
            template=self.settings.chain_templates.memory_template,
        )
        memory = ConversationSummaryBufferMemory(
            llm=self.chain_llm, prompt=memory_prompt
        )
        return memory

    # def _get_conversation_id(self):
    #     return self.conversation_id

    # def _get_memory_prompt(self):
    #     return self.memory.prompt

    def _get_current_messages(self):
        return self.memory.chat_memory.messages

    def add_user_message(self, message: str):
        self.memory.chat_memory.add_user_message(message)

    def add_ai_message(self, message: str):
        self.memory.chat_memory.add_ai_message(message)

    def get_memory_summary(self, existing_summary: str = ""):
        summary = self.memory.predict_new_summary(
            messages=self.memory.chat_memory.messages, existing_summary=existing_summary
        )
        return summary

    def get_request(self, summary: str) -> dict[str, str]:
        requirement_chain = RequirementChain(
            self.chain_llm, self.embeddings_llm, self.settings
        )
        res = requirement_chain.get_request(summary)
        return res

    def get_keywords_from_requirement(self, user_requirement: str) -> list[str]:
        keywords_chain = KeywordsChain(
            self.chain_llm, self.embeddings_llm, self.settings
        )
        res = keywords_chain.get_keywords(user_requirement)
        return res

    def classify_request(self, user_request) -> str:
        classify_chain = ClassifyChain(
            self.chain_llm, self.embeddings_llm, self.settings
        )
        res = classify_chain.classify_request(user_request)
        return res

    def greeting_response(self, user_request: str, last_user_message: str) -> str:
        greeting_chain = GreetingChain(
            self.chain_llm, self.embeddings_llm, self.settings
        )
        res = greeting_chain.greeting_response(user_request, last_user_message)
        return res

    def get_table_names(
        self,
        keywords_arr: list[str],
        score_threshold: float = 0.2,
    ) -> list[str]:
        collection = self.settings.chroma.get_context_col()
        results = collection.query(
            query_texts=keywords_arr,
            n_results=5,
            include=["distances", "metadatas"],
        )

        tables = set()
        for distances, metadatas in zip(results["distances"], results["metadatas"]):
            for distance, metadata in zip(distances, metadatas):
                if distance <= score_threshold:
                    tables.add(metadata["table_name"])

        return list(tables)

    def get_answer_from_db(self, question: str) -> pd.DataFrame:
        tool = VannaTool(self.settings)
        sql = tool.vn.generate_sql(question=question)
        return tool.vn.run_sql(sql=sql)

    def get_complete_response(
        self, user_request: str, context_dataframe: pd.DataFrame
    ) -> str:
        chain = CompleteRequestChain(self.chain_llm, self.settings)
        return chain.get_complete_response(user_request, context_dataframe)
    
    def process_final_response(self, user_request: str, answer: str)-> dict[str, any]:
        chain = ProcessResponseChain(self.chain_llm, self.settings)
        return chain.process_final_response(user_request, answer)
        


class RequirementChain:
    def __init__(
        self, chain_llm: BaseOpenAI, embedding_llm: Embeddings, settings: Settings
    ) -> None:
        self.chain_llm = chain_llm
        self.embedding_llm = embedding_llm
        self.settings = settings

    def get_prompt(self, summary: str) -> str:
        prompt = self.settings.chain_templates.requirement_chain_template
        collection = self.settings.chroma.get_summary_col(self.embedding_llm)
        results = find_db_examples(query=summary, collection=collection)

        for index, result in enumerate(results):
            prompt += f"summary: {result[0].page_content}\n"
            request = result[0].metadata["request"]
            prompt += f"request: {request}\n\n"

            if index == len(results) - 1:
                prompt += """summary: '''{summary}''' \nrequest:"""

        return prompt

    def get_request(self, summary: str) -> dict[str, str]:
        prompt = self.get_prompt(summary)
        llm_chain = LLMChain(
            llm=self.chain_llm,
            verbose=False,
            prompt=PromptTemplate.from_template(prompt),
        )

        res = llm_chain.invoke(input={"summary": summary})

        cadena = res["text"]
        cadena_json = "{" + cadena + "}"
        user_request = json.loads(cadena_json)

        return user_request


class KeywordsChain:
    def __init__(
        self, chain_llm: BaseOpenAI, embedding_llm: Embeddings, settings: Settings
    ) -> None:
        self.chain_llm = chain_llm
        self.embedding_llm = embedding_llm
        self.settings = settings

    def get_prompt(self, user_requirement: str) -> str:
        prompt = ""
        collection = self.settings.chroma.get_keywords_col(self.embedding_llm)
        results = find_db_examples(query=user_requirement, collection=collection)

        for result in results:
            prompt += f"user_requirement: {result[0].page_content}\n"
            keywords = result[0].metadata["keywords"]
            prompt += f"response: {keywords}\n\n"

        prompt += """user_requirement: '''{user_requirement}''' \nresponse:"""

        return prompt

    def get_keywords(self, user_requirement: str) -> list[str]:
        prompt = self.get_prompt(user_requirement)

        llm_chain = LLMChain(
            llm=self.chain_llm,
            verbose=False,
            prompt=PromptTemplate.from_template(prompt),
        )

        res = llm_chain.invoke(input={"user_requirement": user_requirement})
        return eval(res["text"])


class ClassifyChain:
    def __init__(
        self, chain_llm: BaseOpenAI, embedding_llm: Embeddings, settings: Settings
    ) -> None:
        self.chain_llm = chain_llm
        self.embedding_llm = embedding_llm
        self.settings = settings

    def get_prompt(self, user_request: str) -> str:
        prompt = self.settings.chain_templates.classifier_chain_template
        collection_examples = ""
        collection = self.settings.chroma.get_classify_col(self.embedding_llm)
        results = find_db_examples(query=user_request, collection=collection, k=5)

        for result in results:
            collection_examples += f"input: {result[0].page_content}\n"
            analysis = result[0].metadata["analysis"]
            collection_examples += f"analysis: {analysis}\n"
            response = result[0].metadata["response"]
            collection_examples += f"response: {response}\n\n"

        prompt = prompt.format(
            examples=collection_examples, user_request="{user_request}"
        )

        return prompt

    def classify_request(self, user_request: str) -> str:
        prompt = self.get_prompt(user_request)

        llm_chain = LLMChain(
            llm=self.chain_llm,
            verbose=False,
            prompt=PromptTemplate.from_template(prompt),
        )

        response = llm_chain.invoke(input={"user_request": user_request})
        response = str(response["text"])
        lineas = response.strip().split("\n")
        pares = [linea.strip().split(": ", 1) for linea in lineas]
        datos = {clave.strip(): valor.strip() for clave, valor in pares}
        user_request = json.dumps(datos, indent=4)
        user_request = json.loads(user_request)
        return user_request


class GreetingChain:
    def __init__(
        self, chain_llm: BaseOpenAI, embedding_llm: Embeddings, settings: Settings
    ) -> None:
        self.chain_llm = chain_llm
        self.embedding_llm = embedding_llm
        self.settings = settings

    def get_prompt(self) -> str:
        return self.settings.chain_templates.greeting_chain_template

    def greeting_response(
        self, coversation_summary: str, last_user_message: str
    ) -> str:
        prompt = self.get_prompt()

        llm_chain = LLMChain(
            llm=self.chain_llm,
            verbose=False,
            prompt=PromptTemplate.from_template(prompt),
        )

        res = llm_chain.invoke(
            input={
                "coversation_summary": coversation_summary,
                "user_message": last_user_message,
            }
        )
        return res["text"]


class CompleteRequestChain:
    def __init__(self, chain_llm: BaseOpenAI, settings: Settings) -> None:
        self.chain_llm = chain_llm
        self.settings = settings

    def get_prompt(self, context_dataframe: pd.DataFrame) -> str:
        if len(context_dataframe) < 10:
            return self.settings.chain_templates.short_complete_request_chain_template
        else:
            return self.settings.chain_templates.long_complete_request_chain_template

    def get_complete_response(
        self, user_request: str, context_dataframe: pd.DataFrame
    ) -> str:
        prompt = self.get_prompt(context_dataframe)

        llm_chain = LLMChain(
            llm=self.chain_llm,
            verbose=False,
            prompt=PromptTemplate.from_template(prompt),
        )
        
        res = llm_chain.invoke(
            input={
                "user_request": user_request,
                "context_dataframe": context_dataframe.head(10),
            }
        )

        return res["text"]


class ProcessResponseChain:
    def __init__(self, chain_llm: BaseOpenAI, settings: Settings) -> None:
        self.chain_llm = chain_llm
        self.settings = settings

    def get_prompt(self) -> str:
        return self.settings.chain_templates.process_final_response_template

    def process_final_response(self, user_request: str, answer: str) -> dict[str, any]:
        prompt = self.get_prompt()

        llm_chain = LLMChain(
            llm=self.chain_llm,
            verbose=False,
            prompt=PromptTemplate.from_template(prompt),
        )

        res = llm_chain.invoke(
            input={
                "user_input": user_request,
                "actual_answer": answer,
            }
        )
        
        res = str(res["text"]).replace('"', '')
        lineas = res.strip().split("\n")
        pares = [linea.strip().split(": ", 1) for linea in lineas]
        datos = {clave.strip(): valor.strip() for clave, valor in pares}
        res = json.dumps(datos, indent=4)
        res = json.loads(res)
        return res



# chain = Assistant()
# summary = "The human is asking for information about a measurement system and AI asks for an identifier. The human then provides the tag of the system, which is F980-40."
# requirement = chain.get_requirement(summary)
# print(requirement)

# keywords = chain.get_keywords_from_requirement(requirement["requirement"])
# print(keywords)
