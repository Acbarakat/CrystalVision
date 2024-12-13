from datetime import datetime

from langchain_community.chat_message_histories import RedisChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.messages import ChatMessage


def get_message_history(
    session_id: str, url: str = "redis://localhost:6379/0"
) -> RedisChatMessageHistory:
    return RedisChatMessageHistory(session_id, url=url)


class MyRunnableWithMessageHistory(RunnableWithMessageHistory):
    def _get_input_messages(self, i):
        username = i["username"]
        result = []
        for v in super()._get_input_messages(i):
            r = ChatMessage(content=v.content, role=f"User ({username})")
            r.timestamp = (
                datetime.now().isoformat()
                if not hasattr(v, "timestamp")
                else v.timestamp
            )
            result.append(r)
        return result

    def _get_output_messages(self, o):
        val = super()._get_output_messages(o)
        for v in val:
            if not hasattr(v, "timestamp"):
                v.timestamp = datetime.now().isoformat()
        return val
