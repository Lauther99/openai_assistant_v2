import environ

env = environ.Env()
environ.Env.read_env()


class Config:
    @staticmethod
    def get_openai_config():
        return {
            "OPENAI_API_KEY": env("OPENAI_API_KEY"),
            "OPENAI_ECO_MODEL": env("OPENAI_ECO_MODEL"),
            "OPENAI_SUPER_MODEL": env("OPENAI_SUPER_MODEL"),
            "OPENAI_EMBEDDINGS_MODEL": env("OPENAI_EMBEDDINGS_MODEL"),
        }

    @staticmethod
    def get_chromadb_config():
        return {
            "CHROMADB_DIRECTORY": env("CHROMADB_DIRECTORY"),
            "KEYWORDS_COLLECTION": env("KEYWORDS_COLLECTION"),
            "SUMMARY_COLLECTION": env("SUMMARY_COLLECTION"),
            "CLASSIFIER_COLLECTION": env("CLASSIFIER_COLLECTION"),
            "CONTEXT_COLLECTION": env("CONTEXT_COLLECTION"),
        }

    # @staticmethod
    # def get_mongodb_config():
    #     MONGODB_URL = os.getenv("MONGODB_URL")
    #     MONGODB_DATABASE_NAME = os.getenv("MONGODB_DATABASE_NAME")
    #     return MONGODB_URL, MONGODB_DATABASE_NAME

    @staticmethod
    def get_sqldatabase_config():
        return {
            "DB_USER": env("USER"),
            "DB_PWD": env("PWD"),
            "DB_HOST": env("SERVER"),
            "DB_NAME": env("DBNAME"),
            "DB_DRIVER": env("ODBCDRIVER"),
        }

    @staticmethod
    def get_vanna_config():
        return {
            "VANNA_API_KEY": env("VANNA_API_KEY"),
            "VANNA_MODEL": env("VANNA_MODEL"),
        }
