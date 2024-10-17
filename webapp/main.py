import os

import pandas as pd
from fastapi import FastAPI
from fastapi.responses import RedirectResponse
from langchain_community.llms.llamafile import Llamafile
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables.base import RunnableLambda
from pydantic import BaseModel
from qdrant_client import QdrantClient, models
from sentence_transformers import SentenceTransformer

df = pd.read_csv(f"{os.getcwd()}/wine-ratings.csv")
df = df[df["variety"].notna()]
data = df.sample(700).to_dict("records")

encoder = SentenceTransformer("all-MiniLM-L6-v2")  # Model to create embeddings


# create the vector database client
qdrant = QdrantClient(":memory:")  # Create in-memory Qdrant instance


# Create collection to store wines
qdrant.recreate_collection(
    collection_name="top_wines",
    vectors_config=models.VectorParams(
        # Vector size is defined by used model
        size=encoder.get_sentence_embedding_dimension(),
        distance=models.Distance.COSINE,
    ),
)

# vectorize!
qdrant.upload_points(
    collection_name="top_wines",
    points=[
        models.PointStruct(
            id=idx,
            vector=encoder.encode(doc["notes"]).tolist(),
            payload=doc,
        )
        # data is the variable holding all the wines
        for idx, doc in enumerate(data)
    ],
)

app = FastAPI()


class Body(BaseModel):
    query: str


@app.get("/")
def root():
    return RedirectResponse(url="/docs", status_code=301)


@app.post("/ask")
def ask(body: Body):
    """
    Use the query parameter to interact with the llava-v1.5-7b model
    using the Qdrant vector store for Retrieval Augmented Generation.
    """
    search_result = search(body.query)
    chat_bot_response = assistant(body.query, search_result)
    return {"response": chat_bot_response}


def search(query):
    """
    Send the query to Qdrant vectorstore and return the top result
    """

    hits = qdrant.search(
        collection_name="top_wines", query_vector=encoder.encode(query).tolist(), limit=1
    )

    result = hits[0].payload
    print(result)
    return result


def assistant(query, context):

    messages = [
        # Set the system characteristics for this chat bot
        (
            "system",
            "Asisstant is a chatbot that helps you find the best wine for your taste.",
        ),
        ("user", "{query}"),
        ("assistant", "{context}"),
    ]

    client = Llamafile()

    prompt_template = ChatPromptTemplate.from_messages(messages)

    chain = prompt_template | RunnableLambda(lambda x: str(x)) | client

    response = chain.invoke({"query": query, "context": context})

    print(response)
    return str(response)
