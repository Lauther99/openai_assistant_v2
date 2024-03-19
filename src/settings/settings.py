import sys

sys.path.append("C:\\Users\\lauth\\OneDrive\\Desktop\\open_ai_assistant_v2")
from src.settings.env_config import Config
from langchain_community.vectorstores.chroma import Chroma
from langchain_core.embeddings import Embeddings

import chromadb
import chromadb.utils.embedding_functions as embedding_functions
from chromadb.api.models.Collection import Collection


class Template:

    memory_template: str = """
        Progressively summarize the lines of conversation provided, adding onto the previous summary returning a new summary. 
        Be detailed with important information. 
        Do not use information from the example this is just an example of what you have to do.
        Do not hallucinate or try to predict the conversation, work exclusively with the new lines
        
        EXAMPLE
        Current summary:
        The human is greeting AI, then human is asking for the average temperature of a measurement and AI says that need the ID of the measurement system.
        
        New lines of conversation:
        Human: I have a Tag, is MS-125JJG
        AI: Thank you, the average temperature for the measurement system with Tag MS-125JJG is 25°C.

        New summary:
        The human is greeting AI, then human is asking for the average temperature of a measurement and AI says that need the ID of the measurement system, and Human gives the Tag of the measurement system that is MS-125JJG instead of the ID and AI says that the average temperature of the measurement system is 25°C.
        END OF EXAMPLE

        Begin!
        Current summary:
        {summary}

        New lines of conversation:
        {new_lines}

        New summary:"""

    requirement_chain_template: str = (
        """
        You will have conversation summary between a human an AI. You have to find the last intention from the conversation summary.
        Do not hallucinate or try to predict the intention in the conversation. Work exclusively with the summary information in triple simple quotes.
        You have to aswer in JSON format, watch this this examples to get an idea:\n"""
    )

    # classifier_chain_template: str = """
    # You are a classifier assistant. Follow this types to classify inputs:
    #     '''
    #     Type 1, greeting-like: Input is a greeting-like or is like a friend conversation.
    #     Type 2, incomplete: When input is too general/incomplete/ambiguous.
    #     Type 3, complete: When input is completely detailed request.
    #     '''
    #     Use the following format to answer:
    #     analysis: Your analysis for the input.
    #     response: greeting-like/complete/incomplete
        
    #     Follow this examples:\n
    #     {examples}
        
    #     End of examples
        
    #     Begin!
    #     input: {user_request}
    # """
    
    classifier_chain_template: str = """
    You are a classifier assistant. Follow this types to classify inputs:
        '''
        Type 1, simple: When the user request is simple to answer with the context below.
        Type 2, complex-incomplete: When you detect the input is too general/incomplete/ambiguous and you can not answer with the context below.
        Type 3, complex-complete: When input is completely detailed request and you can not answer with the context below.
        '''
        <Context>
        Your name is M-Assistant.
        If the user asks you what you can do, then offer your help giving this list of suggestions:
        "List of measurement systems"
        "List of meters for a specific measurement system"
        "Average temperature for measuring system id/tag/measuring system name"
        <End of the context>
        
        You might use the following format to answer:
        analysis: Your analysis for the input.
        response: complex-incomplete/complete/incomplete
        
        Follow this examples:\n
        {examples}
        
        End of examples
        
        Begin!
        input: {user_request}
    """

    greeting_chain_template: str = (
        """
        Your name is M-Assistant, you are an assistant who will receive and greet people.
        If you are asked for a task regarding measurement systems, respond that you can help obtain the following information of temperature, pressure, viscosity, among other parameters of the existing measurement systems in the database.

        If the user does not know what to ask you, then offer your help in what they can get:
        "List of measurement systems"
        "List of meters for a specific measurement system"
        "Average temperature for measuring system id/tag/measuring system name"
        
        Here is the conversation summary:
        {coversation_summary}
        
        Answer the last user message:
        {user_message}
        """
    )
    
    short_complete_request_chain_template: str = (
        """
        Answer the following user request: '''{user_request}'''
        Your context is the context dataframe below:
        
        DATAFRAME
        {context_dataframe}
        END OF DATAFRAME
        
        
        Do not try to answer by yourself, you must analyze the DataFrame and explain information in this to the user.
        """
    )
    
    long_complete_request_chain_template: str = (
        """
        Answer the following user request: '''{user_request}'''
        You will have a part of the dataframe (10 examples) this is going to be your context:
        
        DATAFRAME
        {context_dataframe}
        END OF DATAFRAME
        
        
        Just analyze the DataFrame and explain the columns, not the data inside just the columns.
        Do not try to answer by yourself.
        Do not give the dataframe in the response or any othr data inside the table like names, ids, etc. Because this context is a part of a large dataframe.
        Do not add in your answer information about the 10 examples from the complete dataframe.
        Do not add in your answer information about the dataframe example given above.
        The total words in your answer must be in a range of 40 to 100 words.
        """
    )
    
    process_final_response_template: str = (
        """
        You will be given a user input and the actual answer, you have to translate the answer to the language from the user input.
        
        Give the final response in this format with "final_answer" as the key of a dictionary:
        '''
        "final_answer": Your final translated answer
        '''
        
        Begin!
        user_input:'''{user_input}'''
        actual_answer:'''{actual_answer}''''
        final_answer:
        """
    )


class OpenAISettings:
    api_key: str = Config.get_openai_config()["OPENAI_API_KEY"]
    eco_model: str = Config.get_openai_config()["OPENAI_ECO_MODEL"]
    super_model: str = Config.get_openai_config()["OPENAI_SUPER_MODEL"]
    embeddings_model: str = Config.get_openai_config()["OPENAI_EMBEDDINGS_MODEL"]


class ChromaDBSetup:
    @staticmethod
    def get_keywords_col(embeddings: Embeddings) -> Chroma:
        '''Keywords collection was created with langchain library'''
        
        directory: str = Config.get_chromadb_config()["CHROMADB_DIRECTORY"]
        kw_collection: str = Config.get_chromadb_config()["KEYWORDS_COLLECTION"]
        # normalized_path = os.path.abspath(os.path.normpath(directory))

        return Chroma(
            embedding_function=embeddings,
            persist_directory=directory,
            collection_name=kw_collection,
        )

    @staticmethod
    def get_summary_col(embeddings: Embeddings) -> Chroma:
        '''Summary collection was created with langchain library'''
        
        directory: str = Config.get_chromadb_config()["CHROMADB_DIRECTORY"]
        sm_collection: str = Config.get_chromadb_config()["SUMMARY_COLLECTION"]
        # normalized_path = os.path.abspath(os.path.normpath(directory))

        collection = Chroma(
            embedding_function=embeddings,
            persist_directory=directory,
            collection_name=sm_collection,
        )

        return collection

    @staticmethod
    def get_classify_col(embeddings: Embeddings) -> Chroma:
        '''Classify collection was created with langchain library'''
        
        directory: str = Config.get_chromadb_config()["CHROMADB_DIRECTORY"]
        sm_collection: str = Config.get_chromadb_config()["CLASSIFIER_COLLECTION"]
        # normalized_path = os.path.abspath(os.path.normpath(directory))

        collection = Chroma(
            embedding_function=embeddings,
            persist_directory=directory,
            collection_name=sm_collection,
        )

        return collection

    @staticmethod
    def get_context_col() -> Collection:
        '''Context collection was created with chromadb library'''
        
        openai_api_key = Config.get_openai_config()["OPENAI_API_KEY"]
        model_name = Config.get_openai_config()["OPENAI_EMBEDDINGS_MODEL"]
        chromadb_directory = Config.get_chromadb_config()["CHROMADB_DIRECTORY"]
        context_col_name = Config.get_chromadb_config()["CONTEXT_COLLECTION"]
        
        openai_ef = embedding_functions.OpenAIEmbeddingFunction(
            api_key=openai_api_key,
            model_name=model_name,
        )
        chroma_client = chromadb.PersistentClient(path=chromadb_directory)
        collection = chroma_client.get_collection(
            name=context_col_name, embedding_function=openai_ef
        )
        return collection
    
class SQLSettings:
    db_user: str = Config.get_sqldatabase_config()["DB_USER"]
    db_pwd: str = Config.get_sqldatabase_config()["DB_PWD"]
    db_host: str = Config.get_sqldatabase_config()["DB_HOST"]
    db_name: str = Config.get_sqldatabase_config()["DB_NAME"]
    db_driver: str = Config.get_sqldatabase_config()["DB_DRIVER"]

class VannaSettings:
    api_key: str = Config.get_vanna_config()["VANNA_API_KEY"]
    model: str = Config.get_vanna_config()["VANNA_MODEL"]
    
# Example 3
# requirement: 'The human wants to know the average temperature for the months of January and February for the measurement system with tag FF-103.'
# thought: 'Human is asking for the average temperature. He is specifying the tag of the measurement system. And he is also specifying a period of time. I think question is kind complete.'
# type: 'complete'
# response: 'Sure, I'm on it, just wait a time before asking database.'


class Settings:
    chain_templates: Template = Template()
    openai: OpenAISettings = OpenAISettings()
    chroma: ChromaDBSetup = ChromaDBSetup()
    sql: SQLSettings = SQLSettings()
    vanna: VannaSettings = VannaSettings()
