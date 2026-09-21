"""Gradio UI for the YouTube RAG Chatbot.

Paste a YouTube video URL, capture its transcript at runtime, and ask questions that are
answered strictly from that transcript (RAG over the loaded video(s)).

Run with:
    python app.py
"""
import os

import gradio as gr

from src.rag_pipeline import YouTubeRagChatbot

# Start with no preloaded videos - the user adds them from the UI.
bot = YouTubeRagChatbot(video_urls=[])
bot.initialize_models()

ASSETS_DIR = os.path.join(os.path.dirname(__file__), "assets")


def _write_avatar(filename: str, emoji: str, bg: str) -> str:
    """Write a small circular emoji-avatar SVG to disk and return its path.

    Local file paths are the well-supported way to set gr.Chatbot avatar_images - inline
    data: URIs are not reliably loaded by the Gradio frontend.
    """
    os.makedirs(ASSETS_DIR, exist_ok=True)
    path = os.path.join(ASSETS_DIR, filename)
    svg = (
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 64 64">'
        f'<circle cx="32" cy="32" r="32" fill="{bg}"/>'
        f'<text x="32" y="44" font-size="34" text-anchor="middle">{emoji}</text></svg>'
    )
    with open(path, "w", encoding="utf-8") as f:
        f.write(svg)
    return path


USER_AVATAR = _write_avatar("user_avatar.svg", "\U0001F9D1", "#6366f1")  # 🧑 on indigo
BOT_AVATAR = _write_avatar("bot_avatar.svg", "\U0001F3AC", "#ec4899")  # 🎬 on pink

CUSTOM_CSS = """
#header-banner {
    background: linear-gradient(135deg, #6366f1 0%, #a855f7 60%, #ec4899 100%);
    border-radius: 16px;
    padding: 22px 28px;
    margin-bottom: 6px;
    color: white !important;
}
#header-banner h1 { margin: 0 0 6px 0; font-size: 1.6rem; }
#header-banner p { margin: 0; opacity: 0.92; }
#video-panel, #chat-panel {
    border: 1px solid var(--border-color-primary);
    border-radius: 14px;
    padding: 14px;
}
#status_box { min-height: 1.4em; }
footer { display: none !important; }
#chatbot, #chatbot * { font-size: 13px !important; }
"""

THEME = gr.themes.Soft(primary_hue="indigo", secondary_hue="pink", neutral_hue="slate")


def load_video(url: str, loaded_state: list, history: list):
    if not url or not url.strip():
        return "\u26A0\uFE0F Please paste a YouTube video URL or ID.", loaded_state, gr.update(), gr.update(), gr.update(), history
    previously_loaded = set(loaded_state)
    status = bot.add_video(url.strip())
    loaded_state = list(bot.loaded_video_ids)
    prefix = "\u2705 " if loaded_state else "\u26A0\uFE0F "
    videos_md = _videos_markdown(loaded_state)
    msg_update, send_update = _chat_controls(loaded_state)

    newly_added = [vid for vid in loaded_state if vid not in previously_loaded]
    if newly_added:
        summary = bot.summarize_video(newly_added[-1])
        welcome = (
            f"\U0001F44B Welcome! I've loaded **`{newly_added[-1]}`**."
            + (f" {summary}" if summary else "")
            + " Feel free to ask me anything about it!"
        )
        history = (history or []) + [{"role": "assistant", "content": welcome}]

    return prefix + status, loaded_state, videos_md, msg_update, send_update, history


def _videos_markdown(video_ids: list) -> str:
    if not video_ids:
        return "_No videos loaded yet._"
    chips = "  \n".join(f"\U0001F3A5 `{vid}`" for vid in video_ids)
    return f"**Loaded videos ({len(video_ids)}):**\n\n{chips}"


def _chat_controls(loaded_state: list):
    """Chat input/send button are disabled until at least one video is loaded."""
    if loaded_state:
        return (
            gr.update(interactive=True, placeholder="Ask something about the loaded video(s)..."),
            gr.update(interactive=True),
        )
    return (
        gr.update(interactive=False, placeholder="Load a YouTube video above to start chatting..."),
        gr.update(interactive=False),
    )


def reset_videos(_loaded_state):
    bot.reset()
    msg_update, send_update = _chat_controls([])
    return "\U0001F5D1\uFE0F All loaded videos cleared.", [], _videos_markdown([]), [], msg_update, send_update, []


def user_submit(message, history):
    history = history or []
    history.append({"role": "user", "content": message})
    return "", history


def _as_text(content) -> str:
    """Gradio's Chatbot may normalize string content into a list of content blocks."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(b.get("text", "") for b in content if isinstance(b, dict))
    return str(content)


def bot_respond(history):
    question = _as_text(history[-1]["content"])
    if question.strip().lower() == "exit":
        history.append({
            "role": "assistant",
            "content": "\U0001F44B Goodbye! This chat session has ended. Click **New Chat** below to start again.",
        })
        return history, gr.update(interactive=False, placeholder="Chat ended - click 'New Chat' to continue"), gr.update(interactive=False)

    try:
        answer, sources = bot.ask(question)
    except Exception as exc:
        answer, sources = f"\u26A0\uFE0F Something went wrong answering that question: {exc}", []
    history.append({"role": "assistant", "content": answer})
    return history, gr.update(), gr.update()


def new_chat(loaded_state: list):
    msg_update, send_update = _chat_controls(loaded_state)
    return [], msg_update, send_update


with gr.Blocks(title="YouTube RAG Chatbot") as demo:
    gr.Markdown(
        "# \U0001F3AC YouTube RAG Chatbot\n"
        "Paste a YouTube link, capture its transcript at runtime, and chat with it - answers are "
        "grounded strictly in what was actually said. If it's not in the transcript, you get an "
        "honest `I don't know.`",
        elem_id="header-banner",
    )

    with gr.Row():
        with gr.Column(scale=1, elem_id="video-panel"):
            gr.Markdown("### \U0001F4FA Video library")
            video_url = gr.Textbox(
                label="YouTube video URL or ID",
                placeholder="https://www.youtube.com/watch?v=...",
            )
            with gr.Row():
                load_btn = gr.Button("\u2795 Load video", variant="primary")
                reset_btn = gr.Button("\U0001F5D1\uFE0F Clear")
            status_box = gr.Markdown("Load a video to get started.", elem_id="status_box")
            videos_box = gr.Markdown(_videos_markdown([]))
            loaded_state = gr.State([])

        with gr.Column(scale=2, elem_id="chat-panel"):
            gr.Markdown("### \U0001F4AC Chat")
            chatbot = gr.Chatbot(
                label=None,
                show_label=False,
                height=460,
                elem_id="chatbot",
                avatar_images=(USER_AVATAR, BOT_AVATAR),
                placeholder="_Ask a question about the loaded video(s) to get started..._",
            )
            msg = gr.Textbox(
                label=None, show_label=False,
                placeholder="Load a YouTube video above to start chatting...",
                interactive=False,
            )
            with gr.Row():
                send_btn = gr.Button("\u27A4 Send", variant="primary", interactive=False)
                new_chat_btn = gr.Button("\U0001F195 New Chat")
            gr.Examples(
                examples=["What is a large language model?", "What is the capital of France?"],
                inputs=msg,
                label="Try an example",
            )

    load_btn.click(load_video, [video_url, loaded_state, chatbot], [status_box, loaded_state, videos_box, msg, send_btn, chatbot])
    video_url.submit(load_video, [video_url, loaded_state, chatbot], [status_box, loaded_state, videos_box, msg, send_btn, chatbot])
    reset_btn.click(reset_videos, loaded_state, [status_box, loaded_state, videos_box, loaded_state, msg, send_btn, chatbot])

    msg.submit(user_submit, [msg, chatbot], [msg, chatbot]).then(bot_respond, chatbot, [chatbot, msg, send_btn])
    send_btn.click(user_submit, [msg, chatbot], [msg, chatbot]).then(bot_respond, chatbot, [chatbot, msg, send_btn])
    new_chat_btn.click(new_chat, loaded_state, [chatbot, msg, send_btn])

if __name__ == "__main__":
    demo.launch(theme=THEME, css=CUSTOM_CSS)


