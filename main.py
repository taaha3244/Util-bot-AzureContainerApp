
import os
from langchain import hub
from dotenv import load_dotenv
from langchain_community.vectorstores import Qdrant
from qdrant_client import QdrantClient
from qdrant_client.http import models
from qdrant_client import QdrantClient
from langchain_openai import OpenAIEmbeddings
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from typing import Optional

# Load environment variables from .env file
load_dotenv()


def util_bot(question: str) -> Optional[str]:
    """
    Processes a given question using a combination of Qdrant vector search and an LLM (Large Language Model) response generation.
    This function embeds the input question using OpenAI's embedding model, searches for relevant context using Qdrant,
    and generates a response based on the context found and the input question using an LLM.

    Args:
        question (str): The user's question to be answered.

    Returns:
        Optional[str]: The generated answer, or None if no answer could be generated.

    Raises:
        Exception: If an error occurs during processing.
    """

    try:
        # Environment variable setup
        openai_api_key = os.getenv('OPENAI_API_KEY')
        qdrant_api_key = os.getenv('QDRANT_API_KEY')
        qdrant_endpoint = os.getenv('QDRANT_CLOUD_URL')

        # Initialization
        embeddings_model = OpenAIEmbeddings(model='text-embedding-3-small', openai_api_key=openai_api_key)
        qdrant_client = QdrantClient(url=qdrant_endpoint, api_key=qdrant_api_key)
        qdrant = Qdrant(client=qdrant_client, collection_name="multiple-util-bot-quanntized",
                        embeddings=embeddings_model)
        retriever = qdrant.as_retriever(search_kwargs={"k": 10})

        # Retrieve documents
        docs = retriever.get_relevant_documents(question)

        # Construct the prompt template for the LLM
        prompt = hub.pull("pwoc517/more-crafted-rag-prompt")
        # RAG Chain
        llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0, openai_api_key=openai_api_key)

        def format_docs(docs):
            return "\n\n".join(doc.page_content for doc in docs)

        rag_chain = (
            {"context": retriever | format_docs, "question": RunnablePassthrough()}
            | prompt
            | llm
            | StrOutputParser()
        )

        result = rag_chain.invoke(question)
        return result

    except Exception as e:
        print(f"Error processing the query: {e}")
        return None

util_bot('What is the peak time during summers according to E6 and what are the charges??')

