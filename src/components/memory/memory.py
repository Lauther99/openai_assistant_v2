import sys

sys.path.append("C:\\Users\\lauth\\OneDrive\\Desktop\\open_ai_assistant_v2")
from src.settings.settings import Settings
from src.components.llms.llms import LLMs
from src.components.memory import MEMORY_TYPES
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
import uuid
import json


class Memory:
    def __init__(self, **kwargs: dict[str, str]) -> None:
        self.settings = Settings()
        self.chat_memory = []

        self.llms = LLMs(self.settings)
        self.chain_llm = self.llms.get_llm(llm_type=self.settings.openai.eco_model)

    def get_current_messages(self):
        return self.chat_memory

    # Chat management
    def add_user_message(self, message: str):
        if (
            len(self.chat_memory) == 0
            or self.chat_memory[-1]["type"] == MEMORY_TYPES["AI"]
        ):
            self.chat_memory.append(
                {
                    "type": MEMORY_TYPES["HUMAN"],
                    "content": message,
                }
            )
            return
        raise ValueError("Can not Human message")

    def add_ai_message(self, message: str):
        if (
            len(self.chat_memory) > 0
            and self.chat_memory[-1]["type"] == MEMORY_TYPES["HUMAN"]
        ):
            self.chat_memory.append(
                {
                    "type": MEMORY_TYPES["AI"],
                    "content": message,
                }
            )
            return
        raise ValueError("Can not add AI message")

    def get_memory_summary(self, existing_summary: str = ""):
        prompt = self.settings.chain_templates.memory_template

        new_lines = ""
        if len(self.chat_memory) > 0:
            for message in self.chat_memory:
                if message["type"] == MEMORY_TYPES["HUMAN"]:
                    message_content = message["content"]
                    new_lines += f"Human: {message_content}"
                elif message["type"] == MEMORY_TYPES["AI"]:
                    message_content = message["content"]
                    new_lines += f"AI: {message_content}"
                else:
                    message_type = message["type"]
                    raise ValueError(f"Not support message type: {message_type}")
        else:
            raise ValueError("Not messages found in the current conversation")
        
        llm_chain = LLMChain(
            llm=self.chain_llm,
            verbose=False,
            prompt=PromptTemplate.from_template(prompt),
        )

        res = llm_chain.invoke(
            input={"current_summary": existing_summary, "new_lines": new_lines}
        )

        return res["text"]
