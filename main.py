from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.chat_history import (
    BaseChatMessageHistory,
    InMemoryChatMessageHistory,
)
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

load_dotenv()
model = ChatOpenAI(model="gpt-3.5-turbo")

store = {}  # fake database


def get_session_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in store:
        store[session_id] = InMemoryChatMessageHistory() #session id yoksa olustur, memory chat olayini InMemoryChatMessageHistory fonksiyonu otomatik yapiyor
    return store[session_id]


prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "you are a helpful assistant. answer all questions to the best of your ability.",
        ),
        MessagesPlaceholder(variable_name="messages"),
    ]
)

chain = prompt | model

config = {"configurable": {"session_id": "abcde123"}} # unique session id

with_message_history = RunnableWithMessageHistory(chain, get_session_history) # chain yapisi ve session history getiren fonksiyon

if __name__ == "__main__":
    while True:
        user_input = input(">")
        for r in with_message_history.stream( # kelimeler uretildikce gonderilir, kullanici deneyimi acisinda anlik yazar
            [
                HumanMessage(content=user_input),
            ],
            config=config,
        ):
            print(r.content, end=" ")
