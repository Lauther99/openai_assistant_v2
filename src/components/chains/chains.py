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
from src.components.chains import REQUEST_TYPES
from langchain.memory import ConversationSummaryBufferMemory
import uuid
import json
import pandas as pd
from src.components.memory.memory import Memory
from src.utils.parsers import txt_2_Json
from src.components.memory import MEMORY_TYPES


class Assistant:
    def __init__(self, **kwargs: dict[str, str]) -> None:
        self.settings = Settings()
        self.llms = LLMs(self.settings)

        self.conversation_id = kwargs.get("conversation_id", str(uuid.uuid4()))

        self.chain_llm = self.llms.get_llm(llm_type=self.settings.openai.eco_model)
        self.embeddings_llm = self.llms.get_llm(
            llm_type=self.settings.openai.embeddings_model
        )
        self.general_chain = LLMChain(
            llm=self.chain_llm,
            verbose=False,
            prompt=PromptTemplate.from_template(
                self.settings.chain_templates.general_chain_template
            ),
        )
        self.memory = Memory()

    # Chat management
    def add_user_message(self, message: str):
        self.memory.add_user_message(message)

    def add_ai_message(self, message: str):
        self.memory.add_ai_message(message)

    def predict_summary(self, existing_summary: str = ""):
        self.summary = self.memory.predict_summary(existing_summary)
        return self.summary

    def get_current_messages(self):
        return self.memory.get_current_messages()

    # Assistant Flow
    # def get_request(self, summary: str) -> dict[str, str]:
    #     requirement_chain = RequirementChain(
    #         self.general_chain, self.embeddings_llm, self.memory, self.settings
    #     )
    #     res = requirement_chain.get_request(summary)
    #     return res
    def get_request(self) -> dict[str, str]:
        requirement_chain = RequirementChain(
            self.general_chain, self.embeddings_llm, self.memory, self.settings
        )
        res = requirement_chain.get_request()
        return res

    def get_keywords_from_requirement(self, user_requirement: str) -> list[str]:
        keywords_chain = KeywordsChain(
            self.chain_llm, self.embeddings_llm, self.settings
        )
        res = keywords_chain.get_keywords(user_requirement)
        return res

    def improved_resquest_filter(
        self,
        user_request: str,
        coversation_summary: str,
        user_message: str,
    ) -> dict[str, any]:
        response = {}

        classify_chain = ClassifyChain(
            self.general_chain, self.embeddings_llm, self.settings, self.memory
        )

        request_type = classify_chain.simple_filter(user_request)
        response["response_type"] = request_type["type"]

        if request_type["type"] == REQUEST_TYPES["simple"]:
            response["response"] = classify_chain.greeting_response(
                coversation_summary, user_message
            )
            response["analysis"] = request_type["analysis"]

        elif request_type["type"] == REQUEST_TYPES["complex"]:

            # keywords = self.get_keywords_from_requirement(user_request)
            # related_data_tables = self.get_related_data_tables(keywords_arr=keywords)

            # complex_info = classify_chain.complex_filter(
            #     user_request, related_data_tables
            # )
            # complex_type = str(complex_info["type"]).strip()

            # response["response_type"] = complex_type
            # response["analysis"] = str(complex_info["analysis"]).strip()

            df = self.ask_sql(user_request)
            response["response"] = df["response"]
            response["dataframe"] = df["dataframe"]
            response["sql"] = df["sql"]

            # if complex_type == REQUEST_TYPES["complete"]:
            #     df = self.ask_sql(user_request)
            #     response["response"] = df["response"]
            #     response["dataframe"].append(df["dataframe"])
            #     response["sql"] = df["sql"]
            # elif complex_type == REQUEST_TYPES["incomplete"]:
            #     response["response"] = "pregunta incompleta"

            # else:
            #     raise ValueError(f"Not support request complex type: {complex_type}")
        else:
            raise ValueError(f"Not support request type: {request_type}")

        return response

    def get_related_data_tables(
        self,
        keywords_arr: list[str],
        score_threshold: float = 0.2,
    ) -> list[tuple[str, str, str]]:

        collection = self.settings.chroma.get_context_col()
        results = collection.query(
            query_texts=keywords_arr,
            n_results=5,
            include=["distances", "metadatas"],
        )

        data = set()

        for distances, metadatas in zip(results["distances"], results["metadatas"]):
            for distance, metadata in zip(distances, metadatas):
                if distance <= score_threshold:
                    data.add(
                        (
                            metadata["table_name"],
                            metadata["ddl"],
                            metadata["description"],
                        )
                    )

        return list(data)

    def ask_sql(self, question: str, max_attempts: int = 3) -> dict[str, any]:
        response = {"sql": "", "dataframe": "", "response": ""}
        tool = VannaTool(self.settings)
        sql = tool.generate_sql(question)
        response["sql"] = sql

        attempts = 0

        while attempts < max_attempts:
            try:
                if sql != "No SELECT statement could be found in the SQL code":
                    df = tool.run_sql(sql)
                    summary = tool.generate_summary(sql, df)
                    response["dataframe"] = df
                    response["response"] = summary
                    break  # Salir del bucle si no hay excepciones
            except Exception as e:
                attempts += 1
                print(f"Error al ejecutar SQL, intento {attempts}: {e}")

        return response

    # Post process
    def process_response(self, user_request: str, answer: str) -> dict[str, any]:
        chain = ProcessResponseChain(self.general_chain, self.settings)
        return chain.process_final_response(user_request, answer)


class RequirementChain:
    def __init__(
        self,
        general_chain: LLMChain,
        embedding_llm: Embeddings,
        memory: Memory,
        settings: Settings,
    ) -> None:
        self.general_chain = general_chain
        self.embedding_llm = embedding_llm
        self.settings = settings
        self.memory = memory

    # def get_prompt(self, summary: str) -> str:
    #     prompt = self.settings.chain_templates.requirement_chain_template
    #     collection = self.settings.chroma.get_summary_col(self.embedding_llm)
    #     collection_examples = ""
    #     results = find_db_examples(query=summary, collection=collection)

    #     for _, result in enumerate(results):
    #         collection_examples += f"summary: {result[0].page_content}\n"
    #         request = result[0].metadata["request"]
    #         collection_examples += f"request: {request}\n\n"

    #     prompt = prompt.format(examples=collection_examples, summary=summary)

    #     return prompt

    def get_prompt(self) -> str:
        prompt = self.settings.chain_templates.requirement_chain_template
        current_messages = self.memory.get_current_messages()
        conversation = ""

        for message in current_messages:
            m = message["content"]
            if message["type"] == MEMORY_TYPES["AI"]:
                conversation += f"AI Message: {m}\n"
            else:
                conversation += f"Human Message: {m}\n"

        prompt = prompt.format(conversation=conversation)
        return prompt

    def get_request(self) -> dict[str, str]:
        p = self.get_prompt()
        res = self.general_chain.invoke(input={"task": p})
        response = txt_2_Json(str(res["text"]))

        return response


class KeywordsChain:
    def __init__(
        self, chain_llm: BaseOpenAI, embedding_llm: Embeddings, settings: Settings
    ) -> None:
        self.chain_llm = chain_llm
        self.embedding_llm = embedding_llm
        self.settings = settings

    def get_prompt(self, user_requirement: str) -> str:
        prompt = self.settings.chain_templates.keywords_chain_template
        collection = self.settings.chroma.get_keywords_col(self.embedding_llm)
        results = find_db_examples(
            query=user_requirement, collection=collection, score_threshold=0.6
        )

        for result in results:
            prompt += f"user_requirement: {result[0].page_content}\n"
            keywords = result[0].metadata["keywords"]
            prompt += f"response: {keywords}\n"

        prompt += """END OF EXAMPLES\n\nuser_requirement: '''{user_requirement}''' \nresponse:"""
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
        self,
        general_chain: LLMChain,
        embedding_llm: Embeddings,
        settings: Settings,
        memory: Memory,
    ) -> None:
        self.general_chain = general_chain
        self.embedding_llm = embedding_llm
        self.settings = settings
        self.memory = memory

    # Prompts
    def simple_filter_prompt(self, user_request: str) -> str:
        prompt = self.settings.chain_templates.simple_classifier_chain_template
        collection_examples = ""
        collection = self.settings.chroma.get_classify_col(self.embedding_llm)
        results = find_db_examples(query=user_request, collection=collection, k=5)

        for result in results:
            collection_examples += f"input: {result[0].page_content}\n"
            analysis = result[0].metadata["analysis"]
            collection_examples += f"analysis: {analysis}\n"
            response = result[0].metadata["response"]
            collection_examples += f"type: {response}\n\n"

        prompt = prompt.format(examples=collection_examples, user_request=user_request)

        return prompt

    def complex_filter_prompt(
        self, data: list[tuple[str, str, str]], user_request: str
    ) -> str:
        prompt = self.settings.chain_templates.complex_classifier_chain_template
        tables_info = ""

        for _, item in enumerate(data):
            table_name = item[0]
            description = item[1]
            ddl = item[2]

            tables_info += (
                f"\nTable: {table_name}\nDescription: {description}\nDDL: {ddl}\n\n"
            )

        prompt = prompt.format(tables_info=tables_info, user_request=user_request)
        return prompt

    def greeting_response_prompt(self, message: str) -> str:
        template = self.settings.chain_templates.greeting_chain_template
        current_messages = self.memory.get_current_messages()
        conversation = ""

        for message in current_messages:
            m = message["content"]
            if message["type"] == MEMORY_TYPES["AI"]:
                conversation += f"AI Message: {m}\n"
            else:
                conversation += f"Human Message: {m}\n"

        prompt = template.format(conversation=conversation, user_message=message)
        return prompt

    # Filters
    def simple_filter(self, user_request: str) -> dict[str, any]:
        p = self.simple_filter_prompt(user_request)

        res = self.general_chain.invoke(input={"task": p})
        response = txt_2_Json(str(res["text"]))
        return response

    def complex_filter(
        self, user_request: str, data: list[tuple[str, str, str]]
    ) -> dict[str, any]:
        p = self.complex_filter_prompt(data, user_request)

        res = self.general_chain.invoke(input={"task": p})
        response = txt_2_Json(str(res["text"]))

        return response

    # Responses
    def greeting_response(self, last_user_message: str) -> str:
        p = self.greeting_response_prompt(last_user_message)

        res = self.general_chain.invoke(input={"task": p})
        return res["text"]


class ProcessResponseChain:
    def __init__(self, general_chain: LLMChain, settings: Settings) -> None:
        self.general_chain = general_chain
        self.settings = settings

    def get_translator_prompt(self, request: str, answer: str) -> str:
        t = self.settings.chain_templates.translator_template
        prompt = t.format(user_input=request, actual_answer=answer)
        return prompt

    def translate_answer(self, user_request: str, answer: str) -> str:
        p = self.get_translator_prompt(user_request, answer)

        res = self.general_chain.invoke(input={"task": p})
        response = txt_2_Json(str(res["text"]))

        return response

    def process_final_response(self, user_request: str, answer: str) -> dict[str, any]:
        res = self.translate_answer(user_request, answer)
        return res
