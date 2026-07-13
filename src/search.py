import os
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.prompts import PromptTemplate
from langchain_postgres import PGVector

load_dotenv()

for k in ("GOOGLE_API_KEY", "OPENAI_API_KEY", "DATABASE_URL", "PG_VECTOR_COLLECTION_NAME", "PDF_PATH"):
    if not os.getenv(k):
        raise RuntimeError(f"Environment variable {k} is not set")

PROMPT_TEMPLATE = """
CONTEXTO:
{contexto}

REGRAS:
- Responda somente com base no CONTEXTO.
- Se a informação não estiver explicitamente no CONTEXTO, responda:
  "Não tenho informações necessárias para responder sua pergunta."
- Nunca invente ou use conhecimento externo.
- Nunca produza opiniões ou interpretações além do que está escrito.

EXEMPLOS DE PERGUNTAS FORA DO CONTEXTO:
Pergunta: "Qual é a capital da França?"
Resposta: "Não tenho informações necessárias para responder sua pergunta."

Pergunta: "Quantos clientes temos em 2024?"
Resposta: "Não tenho informações necessárias para responder sua pergunta."

Pergunta: "Você acha isso bom ou ruim?"
Resposta: "Não tenho informações necessárias para responder sua pergunta."

PERGUNTA DO USUÁRIO:
{pergunta}

RESPONDA A "PERGUNTA DO USUÁRIO"
"""

def search_prompt(question = None):
    embeddings = OpenAIEmbeddings(model = os.getenv("OPENAI_EMBEDDING_MODEL","text-embedding-3-small"))

    store = PGVector(
        embeddings = embeddings,
        collection_name = os.getenv("PG_VECTOR_COLLECTION_NAME"),
        connection = os.getenv("DATABASE_URL"),
        use_jsonb = True,
    )

    results = store.similarity_search(question, k = 3)

    context = "\n".join([f"Documento {i+1}: {doc.page_content}" for i, doc in enumerate(results)])

    prompt = PromptTemplate.from_template(PROMPT_TEMPLATE)
    llm = ChatOpenAI(model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"), temperature=0)
    chain = prompt | llm

    response = chain.invoke({"contexto": context, "pergunta": question})
    return response.content