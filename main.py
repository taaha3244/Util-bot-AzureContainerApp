
import os
from dotenv import load_dotenv
from langchain_community.vectorstores import Qdrant
from qdrant_client import QdrantClient
from qdrant_client.http import models
from qdrant_client import QdrantClient
from langchain_openai import OpenAIEmbeddings
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain_openai import ChatOpenAI
from typing import Optional

# Load environment variables from .env file
load_dotenv()

# Now you can use os.getenv to access the variables
openai_api_key = os.getenv('openai-api')
qdrant_api_key = os.getenv('qdrant-api')
qdrant_endpoint = os.getenv('qdrant-endpoint')



def util_bot(question: str) :
    """
    Processes a given question using a combination of Qdrant vector search and an LLM (Large Language Model) response generation.

    This function embeds the input question using OpenAI's embedding model, searches for relevant context using Qdrant,
    and generates a response based on the context found and the input question using an LLM.

    Args:
        question (str): The user's question to be answered.
        openai_api_key (str): API key for accessing OpenAI's services.

    Returns:
        Optional[str]: The generated answer, or None if no answer could be generated.

    Raises:
        Exception: If an error occurs during processing.
    """


    # Initialize Qdrant client
    qdrant_client = QdrantClient(url=qdrant_endpoint, api_key=qdrant_api_key)

    # Initialize embeddings using OpenAI
    embeddings = OpenAIEmbeddings(model='text-embedding-3-small', openai_api_key=openai_api_key)

    try:
        # Embed the input question for vector search
        query_result = embeddings.embed_query(question)

        # Perform vector search in the "util-bot" collection
        response = qdrant_client.search(
            collection_name="util-bot-quantized",
            query_vector=query_result,
            limit=3  # Retrieve top 3 closest vectors
        )

        # Construct the prompt template for the LLM
        prompt=PromptTemplate(
        template=""""
        Given the context from the utility bills document, answer the following question. The document includes general text, definitions, formulas, and charts related to utility bills. Use the information provided in the document to construct your answer. If the answer is not explicitly found in the document, it's okay to say "The context does not include any reference for the question asked" instead of guessing.

        Example:
        Question: What is the formula for calculating the monthly electricity bill?
        Answer:
        ```
         The monthly electricity bill is calculated using the formula: Bill = Rate per kWh * Number of kWh consumed. This formula takes into account the rate charged per kilowatt-hour (kWh) and the total energy consumption in kWh.
        ```

        <utility_bills_document>
        {context}
        </utility_bills_document>

        Question: {question}

        Helpful Answer:
        """,
        input_variables=["context", "question"]
)


        # Initialize LLM and the chain for generating the response
        llm = ChatOpenAI(model='gpt-3.5-turbo-0125',openai_api_key=openai_api_key)
        chain = LLMChain(llm=llm, prompt=prompt)

        # Generate the response
        result = chain({
            "question": question,
            "context": "\n".join([doc.payload['page_content'] for doc in response])  # Concatenate context from search results
        })

        return result['text']
    except Exception as e:
        # Log the exception or handle it as per the application's error handling policy
        print(f"Error processing question: {e}")
        return None
