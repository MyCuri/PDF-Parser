from pymongo.mongo_client import MongoClient
from pymongo.server_api import ServerApi

import config


class MongoDBHandler:
    def __init__(self):
        print(config.MONGODB_URI)
        self.client = MongoClient(config.MONGODB_URI, server_api=ServerApi("1"))
        self.pdf_collection = self.client[config.MONGODB_DATABASE][
            config.MONGODB_PDF_COLLECTION
        ]
        self.prompt_collection = self.client[config.MONGODB_DATABASE][
            config.MONGODB_PROMPTS_COLLECTION
        ]

    def insert_pdf(self, pdf_name: str):
        data = {"filename": pdf_name}
        result = self.pdf_collection.insert_one(data)
        return result.inserted_id

    def insert_prompt(self, name: str, prompt: str):
        data = {"name": name, "prompt": prompt}
        result = self.prompt_collection.insert_one(data)
        return result.inserted_id

    def delete_pdf_filename(self, filename):
        self.pdf_collection.delete_many({"filename": filename})

    def delete_prompt(self, name):
        self.prompt_collection.delete_one({"name": name})

    def get_pdf_filenames(self):
        return list(
            set(
                doc["filename"]
                for doc in self.pdf_collection.find({}, {"filename": 1, "_id": 0})
            )
        )

    def get_prompts(self):
        return {
            doc["name"]: doc["prompt"]
            for doc in self.prompt_collection.find(
                {}, {"name": 1, "prompt": 1, "_id": 0}
            )
        }

    def close(self):
        """Close the MongoDB connection."""
        self.client.close()
