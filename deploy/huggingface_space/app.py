from __future__ import annotations

import sys
from pathlib import Path

import gradio as gr


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from virtual_child_rl_system import VirtualChildRLSystem  # noqa: E402


SYSTEMS: dict[str, VirtualChildRLSystem] = {}


def get_system(algorithm: str) -> VirtualChildRLSystem:
    system = SYSTEMS.get(algorithm)
    if system is None:
        system = VirtualChildRLSystem(algorithm=algorithm)
        SYSTEMS[algorithm] = system
    return system


def reset_chat(algorithm: str):
    SYSTEMS[algorithm] = VirtualChildRLSystem(algorithm=algorithm)
    system = SYSTEMS[algorithm]
    opening = system.start_session()
    return [{"role": "assistant", "content": opening}], system.render_caregiver_summary()


def chat(message: str, history: list[dict], algorithm: str):
    system = get_system(algorithm)
    if not system.session.started:
        history = [{"role": "assistant", "content": system.start_session()}]

    history = list(history)
    history.append({"role": "user", "content": message})
    result = system.respond(message)
    history.append({"role": "assistant", "content": result["assistant_message"]})
    return history, system.render_caregiver_summary()


with gr.Blocks(title="Virtual Child Companion RL Demo", theme=gr.themes.Soft()) as demo:
    gr.Markdown(
        """
        # Virtual Child Companion AI
        這個 Hugging Face Space 會示範 `DQN` 或 `Q-learning` 如何驅動虛擬兒女陪伴對話，
        並在互動過程中整理出家屬可讀的健康摘要。
        """
    )

    with gr.Row():
        algorithm = gr.Radio(
            choices=["dqn", "q_learning"],
            value="dqn",
            label="RL Algorithm",
        )
        reset = gr.Button("Reset Session")

    chatbot = gr.Chatbot(type="messages", height=520, label="對話視窗")
    caregiver_summary = gr.Markdown(label="Caregiver Summary")
    textbox = gr.Textbox(label="長者輸入", placeholder="例如：我今天五點就起床去公園練氣功了")
    submit = gr.Button("Send")

    demo.load(fn=lambda: reset_chat("dqn"), outputs=[chatbot, caregiver_summary])
    algorithm.change(fn=reset_chat, inputs=algorithm, outputs=[chatbot, caregiver_summary])
    reset.click(fn=reset_chat, inputs=algorithm, outputs=[chatbot, caregiver_summary])
    submit.click(fn=chat, inputs=[textbox, chatbot, algorithm], outputs=[chatbot, caregiver_summary])
    submit.click(fn=lambda: "", outputs=textbox)
    textbox.submit(fn=chat, inputs=[textbox, chatbot, algorithm], outputs=[chatbot, caregiver_summary])
    textbox.submit(fn=lambda: "", outputs=textbox)


if __name__ == "__main__":
    demo.launch()
