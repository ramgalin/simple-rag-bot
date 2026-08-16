from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

# rewrite a follow-up into a standalone question using the history
CONDENSE_SYSTEM = (
    "Given the chat history and the latest user question, rewrite the question "
    "as a standalone question understandable without the chat history. "
    "Resolve pronouns and references using the history. "
    "Do NOT answer it — only reformulate. "
    "If it is already standalone, return it unchanged."
)

CONDENSE_PROMPT = ChatPromptTemplate.from_messages([
    ("system", CONDENSE_SYSTEM),
    MessagesPlaceholder("chat_history"),
    ("human", "{question}"),
])

# answer using retrieved context + history
RAG_SYSTEM = (
    "You are a helpful assistant that answers questions using the provided context.\n"
    "Use ONLY the information in the context to answer.\n"
    "If the context does not contain the answer, say you don't know — "
    "do not rely on outside knowledge and do not make things up.\n"
    "You MUST write your entire answer in {answer_language}. "
    "This is fixed and does not depend on the language of the context or the topic.\n"
    "Cite the sources you rely on using their bracket numbers, e.g. [1], [2].\n\n"
    "Context:\n{context}"
)

RAG_PROMPT = ChatPromptTemplate.from_messages([
    ("system", RAG_SYSTEM),
    MessagesPlaceholder("chat_history"),
    ("human", "{question}"),
])