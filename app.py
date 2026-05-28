import gradio as gr
from langchain_core.messages import HumanMessage
from agent_system.orchestratory import create_super_graph

from dotenv import load_dotenv
from pathlib import Path

load_dotenv(dotenv_path=Path(".env"))

# ── cache graphs ────────────────────────────────────────────────────────────
_graphs = {}

def get_graph(provider: str):
    if provider not in _graphs:
        _graphs[provider] = create_super_graph(provider)
    return _graphs[provider]


# ── core inference ──────────────────────────────────────────────────────────
def chat(message: str, history: list, provider: str) -> str:
    """history: list of (user_str, bot_str) tuples — Gradio tuple format"""
    graph = get_graph(provider)

    lc_messages = []
    for user_msg, _ in history:
        lc_messages.append(HumanMessage(content=user_msg))

    lc_messages.append(HumanMessage(content=message))

    result = graph.invoke({"messages": lc_messages})
    return result["messages"][-1].content


# ── detect gradio version ───────────────────────────────────────────────────
import gradio as gr
_gr_version = tuple(int(x) for x in gr.__version__.split(".")[:2])
_use_messages = _gr_version >= (4, 0)


# ── CSS ─────────────────────────────────────────────────────────────────────
CSS = """
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=DM+Sans:wght@300;400;500&display=swap');

:root {
    --bg:      #0d0f14;
    --surface: #13161e;
    --border:  #1e2330;
    --accent:  #00e5ff;
    --accent2: #7c3aed;
    --text:    #e2e8f0;
    --muted:   #64748b;
    --radius:  12px;
}
body, .gradio-container {
    background: var(--bg) !important;
    font-family: 'DM Sans', sans-serif !important;
    color: var(--text) !important;
}
#header { text-align:center; padding:32px 0 8px; border-bottom:1px solid var(--border); margin-bottom:16px; }
#header h1 { font-family:'Space Mono',monospace; font-size:1.8rem; color:var(--accent); letter-spacing:-0.02em; margin:0; }
#header p  { color:var(--muted); font-size:0.85rem; margin:6px 0 0; }
.message { border-radius:var(--radius) !important; color:var(--text) !important; }
textarea, input[type=text] {
    background:var(--surface) !important; border:1px solid var(--border) !important;
    color:var(--text) !important; border-radius:var(--radius) !important;
}
textarea:focus, input[type=text]:focus {
    border-color:var(--accent) !important;
    box-shadow:0 0 0 2px rgba(0,229,255,0.15) !important;
}
button.primary {
    background:linear-gradient(135deg,var(--accent2),var(--accent)) !important;
    border:none !important; border-radius:var(--radius) !important;
    color:#fff !important; font-family:'Space Mono',monospace !important;
}
button.secondary {
    background:var(--surface) !important; border:1px solid var(--border) !important;
    color:var(--muted) !important; border-radius:var(--radius) !important;
}
::-webkit-scrollbar { width:6px; }
::-webkit-scrollbar-track { background:var(--bg); }
::-webkit-scrollbar-thumb { background:var(--border); border-radius:3px; }
"""

PROVIDERS = ["OpenAI", "qwen", "llama"]

# ── build UI ─────────────────────────────────────────────────────────────────
with gr.Blocks(css=CSS, title="GAIA Agent") as demo:

    gr.HTML("""
    <div id="header">
        <h1>⬡Q&A Agent</h1>
        <p>Powered by LangGraph · RAG · Multi-tool reasoning</p>
    </div>
    """)

    with gr.Row():

        # sidebar
        with gr.Column(scale=1, min_width=200):
            provider_dd = gr.Dropdown(
                choices=PROVIDERS, value="OpenAI",
                label="🤖 LLM Provider", interactive=True,
            )
            clear_btn = gr.Button("🗑 Clear chat", variant="secondary", size="sm")
            gr.Markdown("""
---
**Tips**
- Ask factual / multi-step questions
- Agent can search the web & read files
- Switch provider to compare models
            """)

        # chat panel
        with gr.Column(scale=4):

            # ── version-safe Chatbot ─────────────────────────────────────────
            if _use_messages:
                chatbot = gr.Chatbot(label="", height=520, show_label=False, type="messages")
            else:
                chatbot = gr.Chatbot(label="", height=520, show_label=False)

            with gr.Row():
                msg_box = gr.Textbox(
                    placeholder="Ask anything…", show_label=False,
                    scale=5, lines=1, max_lines=4,
                )
                send_btn = gr.Button("Send →", variant="primary", scale=1)

    # ── callbacks ────────────────────────────────────────────────────────────
    def respond(message, history, provider):
        if not message.strip():
            return history, ""
        history = history or []

        if _use_messages:
            # history is list of {"role":..., "content":...}
            tuples = [(m["content"], "") for m in history if m["role"] == "user"]
        else:
            # history is list of (user, bot) tuples
            tuples = history

        try:
            reply = chat(message, tuples, provider)
        except Exception as e:
            reply = f"⚠️ Error: {e}"

        if _use_messages:
            history.append({"role": "user",      "content": message})
            history.append({"role": "assistant", "content": reply})
        else:
            history.append((message, reply))

        return history, ""

    send_btn.click(respond, [msg_box, chatbot, provider_dd], [chatbot, msg_box])
    msg_box.submit(respond, [msg_box, chatbot, provider_dd], [chatbot, msg_box])
    clear_btn.click(lambda: ([], ""), outputs=[chatbot, msg_box])


if __name__ == "__main__":
    print(f"Gradio {gr.__version__} — messages_format={_use_messages}")
    demo.launch(share=False, server_port=7860)