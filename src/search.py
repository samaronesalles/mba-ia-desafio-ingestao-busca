from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate

from config import OPENAI_MODEL
from db import get_vector_store

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


def search_prompt(question=None):
    store = get_vector_store()
    results = store.similarity_search(question, k=3)

    context = "\n".join(
        [f"Documento {i + 1}: {doc.page_content}" for i, doc in enumerate(results)]
    )

    prompt = PromptTemplate.from_template(PROMPT_TEMPLATE)
    llm = ChatOpenAI(model=OPENAI_MODEL, temperature=0)
    chain = prompt | llm

    response = chain.invoke({"contexto": context, "pergunta": question})
    return response.content
