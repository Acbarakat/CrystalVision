from typing import (
    Any,
    Dict,
    List,
    Literal,
    Optional,
    Sequence,
    cast,
)

from langchain_ollama import ChatOllama
from langchain_ollama.chat_models import _lc_tool_call_to_openai_tool_call
from ollama import Message
from langchain_core.messages import (
    HumanMessage,
    SystemMessage,
    AIMessage,
    ChatMessage,
    ToolMessage,
    BaseMessage,
)


class MyChatOllama(ChatOllama):
    def _convert_messages_to_ollama_messages(
        self, messages: List[BaseMessage]
    ) -> Sequence[Message]:
        ollama_messages: List = []
        for message in messages:
            role: Literal["user", "assistant", "system", "tool"]
            tool_call_id: Optional[str] = None
            tool_calls: Optional[List[Dict[str, Any]]] = None
            if isinstance(message, HumanMessage):
                role = "user"
            elif isinstance(message, AIMessage):
                role = "assistant"
                tool_calls = (
                    [
                        _lc_tool_call_to_openai_tool_call(tool_call)
                        for tool_call in message.tool_calls
                    ]
                    if message.tool_calls
                    else None
                )
            elif isinstance(message, SystemMessage):
                role = "system"
            elif isinstance(message, ToolMessage):
                role = "tool"
                tool_call_id = message.tool_call_id
            elif isinstance(message, ChatMessage):
                # role = message.role
                role = "user"
            else:
                raise ValueError("Received unsupported message type for Ollama.")

            content = ""
            images = []
            if isinstance(message.content, str):
                content = message.content
            else:
                for content_part in cast(List[Dict], message.content):
                    if content_part.get("type") == "text":
                        content += f"\n{content_part['text']}"
                    elif content_part.get("type") == "tool_use":
                        continue
                    elif content_part.get("type") == "image_url":
                        image_url = None
                        temp_image_url = content_part.get("image_url")
                        if isinstance(temp_image_url, str):
                            image_url = temp_image_url
                        elif (
                            isinstance(temp_image_url, dict)
                            and "url" in temp_image_url
                            and isinstance(temp_image_url["url"], str)
                        ):
                            image_url = temp_image_url["url"]
                        else:
                            raise ValueError(
                                "Only string image_url or dict with string 'url' "
                                "inside content parts are supported."
                            )

                        image_url_components = image_url.split(",")
                        # Support data:image/jpeg;base64,<image> format
                        # and base64 strings
                        if len(image_url_components) > 1:
                            images.append(image_url_components[1])
                        else:
                            images.append(image_url_components[0])

                    else:
                        raise ValueError(
                            "Unsupported message content type. "
                            "Must either have type 'text' or type 'image_url' "
                            "with a string 'image_url' field."
                        )
            # Should convert to ollama.Message once role includes tool, and tool_call_id is in Message # noqa: E501
            msg: dict = {
                "role": role,
                "content": content,
                "images": images,
            }
            if tool_calls:
                msg["tool_calls"] = tool_calls  # type: ignore
            if tool_call_id:
                msg["tool_call_id"] = tool_call_id
            ollama_messages.append(msg)

        return ollama_messages
