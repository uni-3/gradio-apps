from typing import List, Union

import openai
import gradio as gr
from pydantic import BaseModel

SYSTEM_PROMPT = """
あなたはプロのギタリストです。音楽や作曲に対する質問に答えてください
"""

MODEL = "gpt-3.5-turbo"
# MODEL = "gpt-4"

class Message(BaseModel):
    role: str
    content: str

def openai_create(
        messages: List[Message],
        tempareture: float,
        openai_api_key: str,
        ) -> str:
    openai.api_key = openai_api_key
    response = openai.ChatCompletion.create(
        model=MODEL,
        messages=[m.dict() for m in messages],
        temperature=tempareture,
        max_tokens=400,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0.6,
    )

    return response.choices[0]["message"]["content"]

def chatgpt_clone(
        input: str,
        messages: list[Message],
        tempareture: float,
        openai_api_key: str,
        ) -> Union[List[tuple[str, str]], List[Message]]:
    init = Message(role="system", content=SYSTEM_PROMPT)
    messages = messages or [init]

    m = Message(role="user", content=input)
    messages.append(m)

    res = openai_create(messages, tempareture, openai_api_key)
    messages.append(Message(role="assistant", content=res))
    # tuple で履歴をテキストボックスに送る必要がある
    history = [(messages[i].content, messages[i + 1].content) for i in range(1, len(messages) - 1, 2)]
    return history, messages

def main():
    with gr.Blocks() as block:
        with gr.Row():
            openai_api_key_textbox = gr.Textbox(
                label="api key",
                placeholder="OpenAI API key を入力(sk-...) ↵️",
                lines=1,
                type="password"
            )
            temperature = gr.Number(label="temperature", value=0.9)

        gr.Markdown("""
        let's talk
        """)
        chatbot = gr.Chatbot(label="chat")
        message = gr.Textbox(placeholder="input message", label="message")
        state = gr.State()

        with gr.Row():
            submit = gr.Button("submit")
            clear = gr.Button("clear")
        submit.click(fn=chatgpt_clone,
                     inputs=[message, state, temperature, openai_api_key_textbox],
                     outputs=[chatbot, state]
                     )
        clear.click(lambda: [], None, chatbot)
        clear.click(lambda: [], None, state)

        block.launch(server_name="0.0.0.0", debug=True)

if __name__ == '__main__':
    main()