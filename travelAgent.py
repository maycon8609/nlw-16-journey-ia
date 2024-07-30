import os
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.agent_toolkits.load_tools import load_tools
from langchain.agents import create_react_agent, AgentExecutor
from langchain import hub

import json

from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores import Chroma
import bs4

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import PromptTemplate

from langchain_core.runnables import RunnableSequence

OPENAI_API_KEY= os.environ['OPENAI_API_KEY']

llm = ChatOpenAI(model="gpt-3.5-turbo")

def research_agent(query, llm):
    """
    Execute a research query using the specified language model and tools.
    :param query: The query to be researched.
    :param llm: The language model to be used.
    :return: The output of the research query.
    """
    tools = load_tools(["ddg-search", "wikipedia"], llm=llm)
    prompt = hub.pull("hwchase17/react")
    agent = create_react_agent(llm, tools, prompt)
    agent_executor = AgentExecutor(agent=agent, tools=tools, prompt=prompt)
    webContext = agent_executor.invoke({ "input": query })
    return webContext['output']

def load_data():
    """
    Load data from a specific web page, split the documents into chunks,
    create a vector store using Chroma with OpenAI embeddings, and return
    a retriever for the data.
    """
    loader = WebBaseLoader(
    web_paths= ("https://www.dicasdeviagem.com/inglaterra/",),
    bs_kwargs=dict(
        parse_only=bs4.SoupStrainer(
            class_=("postcontentwrap", "pagetitleloading background-imaged loading-dark")
        )
      ),
    )
    docs = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splits = text_splitter.split_documents(docs)
    vectorstore = Chroma.from_documents(documents=splits, embedding=OpenAIEmbeddings())
    retriever = vectorstore.as_retriever()
    return retriever

def get_relevant_docs(query):
    """
    Retrieve relevant documents based on a given query using the loaded data retriever.
    """
    retriever = load_data()
    relevant_documents = retriever.invoke(query)
    return relevant_documents

def supervisor_agent(query, llm, webContext, relevant_documents):
    """
    Define a supervisor agent for a travel agency scenario.
    Combines user input, web context, and relevant documents to generate a detailed
    travel itinerary.
    :param query: User input for the travel itinerary.
    :param llm: Language model for generating responses.
    :param webContext: Context of events and flight prices.
    :param relevant_documents: Documents relevant to the travel itinerary.
    :return: Detailed and complete travel itinerary response.
    """
    prompt_template = """
    Você é um gerente de uma agência de viagens. Sua resposta final deverá ser um roteiro de viagem completo e detalhado. 
    Utilize o contexto de eventos e preços de passagens, o input do usuário e também os documentos relevantes para elaborar o roteiro.
    Contexto: {webContext}
    Documento relevante: {relevant_documents}
    Usuário: {query}
    Assistente:
    """

    prompt = PromptTemplate(
      input_variables=['webContext', 'relevant_documents', 'query'],
      template = prompt_template
    )

    sequence = RunnableSequence(prompt | llm)
    response = sequence.invoke({
        "webContext": webContext,
        "relevant_documents": relevant_documents,
        "query": query
    })
    return response

def get_response(query, llm):
    """
    Get a detailed and complete travel itinerary response based on the user query
    and language model.
    :param query: The user input for the travel itinerary.
    :param llm: The language model for generating responses.
    :return: Detailed and complete travel itinerary response.
    """
    webContext = research_agent(query, llm)
    relevant_documents = get_relevant_docs(query)
    response = supervisor_agent(query, llm, webContext, relevant_documents)
    return response

def lambda_handler(event, context):
    """
    Handle incoming event data, extract a query from the body, get a response,
    and return a JSON object with the response details.
    
    :param event: Dictionary containing the input data.
    :param context: Object providing runtime information.
    :return: Dictionary with a success message and the response details in JSON format.
    """
    body = json.loads(event.get('body', {}))
    query = body.get('question', 'Parametro question não fornecido')
    response = get_response(query, llm).content
    return {
      "statusCode": 200,
      "headers": {
        "Content-Type": "application/json"
      },
      "body": json.dumps({
        "message": "Tarefa concluída com sucesso",
        "details": response
      }),
    }
