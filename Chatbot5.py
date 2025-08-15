from langchain_core.prompts import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    MessagesPlaceholder,
    SystemMessagePromptTemplate,
)
from langchain_community.chat_message_histories import StreamlitChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import HumanMessage

import streamlit as st
import streamlit.components.v1 as components
import base64

# Page config
st.set_page_config(page_title="AI Text & Image Assistant", page_icon="🤖", layout="wide")

# --- Global custom CSS ---
st.markdown(
    """
    <style>
    .stApp {
      background: linear-gradient(180deg, #f8fbff 0%, #ffffff 50%);
      color: #0b1224;
    }
    @keyframes fadeSlideIn {
      0% { opacity: 0; transform: translateY(10px); }
      100% { opacity: 1; transform: translateY(0); }
    }
    .user-bubble {
      background: #7b61ff;
      color: white;
      padding: 12px 16px;
      border-radius: 16px 16px 0px 16px;
      max-width: 80%;
      margin: 8px 0;
      animation: fadeSlideIn 0.3s ease-out;
      box-shadow: 0 4px 12px rgba(123, 97, 255, 0.2);
    }
    .ai-bubble {
      background: linear-gradient(135deg, #00b8ff, #7b61ff);
      color: white;
      padding: 12px 16px;
      border-radius: 16px 16px 16px 0px;
      max-width: 80%;
      margin: 8px 0;
      animation: fadeSlideIn 0.3s ease-out;
      box-shadow: 0 4px 12px rgba(0, 184, 255, 0.2);
    }
    /* Typing dots animation */
    .typing { display: inline-block; }
    .typing span {
      animation: blink 1.4s infinite both;
      font-size: 24px;
      line-height: 0;
    }
    .typing span:nth-child(2) { animation-delay: 0.2s; }
    .typing span:nth-child(3) { animation-delay: 0.4s; }
    @keyframes blink {
      0% { opacity: 0.2; }
      20% { opacity: 1; }
      100% { opacity: 0.2; }
    }
    /* Mobile responsive */
    @media screen and (max-width: 768px) {
      .hero {
        flex-direction: column !important;
        text-align: center;
      }
      .hero .right {
        width: 100% !important;
        margin-top: 15px;
      }
      .hero-img {
        max-width: 90% !important;
      }
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# --- Sidebar ---
st.sidebar.header("Configuration")
api_key = st.sidebar.text_input("Enter your Google API Key", type="password")
model_choice = st.sidebar.selectbox("Model", ["gemini-1.5-flash"], index=0)

# Initialize chat history persistence
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# --- Main app ---
col1, col2 = st.columns([3, 1])

with col1:
    st.header("AI Chat & Image Analysis")

    # Single uploader for both header display & AI analysis
    uploaded_image = st.file_uploader("Attach an image (optional)", type=["png", "jpg", "jpeg", "webp"])
    user_input = st.text_input("Enter your question:", key="user_input", placeholder="e.g., Explain gradient descent")

    # Hero image logic
    if uploaded_image:
        img_bytes = uploaded_image.read()
        mime = uploaded_image.type if hasattr(uploaded_image, "type") and uploaded_image.type else "image/png"
        b64 = base64.b64encode(img_bytes).decode()
        header_src = f"data:{mime};base64,{b64}"
    else:
        header_src = "https://images.unsplash.com/photo-1518770660439-4636190af475?auto=format&fit=crop&w=1200&q=80"

    # Hero section
    hero_html = f"""
    <div class="hero" style="display:flex;gap:28px;align-items:center;padding:20px;border-radius:14px;
        background:linear-gradient(90deg,rgba(123,97,255,0.06),rgba(0,184,255,0.03));
        box-shadow:0 10px 30px rgba(11,13,24,0.06);">
      <div class="left" style="flex:1;">
        <div class="logo" style="font-size:28px;font-weight:800;
            background:linear-gradient(90deg,#7b61ff,#00b8ff);
            -webkit-background-clip:text;color:transparent;margin-bottom:6px;">
            AI Chat & Image Assistant
        </div>
        <div class="lead" style="color:#273043;font-size:15px;margin-bottom:12px;">
            Ask questions, upload images, and get instant insights.
        </div>
      </div>
      <div class="right" style="width:320px;text-align:center;">
        <img class="hero-img" src="{header_src}" alt="Hero image" style="width:100%;max-width:320px;
            border-radius:12px;animation:float 6s ease-in-out infinite;
            box-shadow:0 12px 30px rgba(11,13,24,0.08);" />
      </div>
    </div>
    """
    components.html(hero_html, height=260)

    # Display previous chat history
    for msg in st.session_state.chat_history:
        bubble_class = "user-bubble" if msg["role"] == "user" else "ai-bubble"
        st.markdown(f"<div class='{bubble_class}'>{msg['content']}</div>", unsafe_allow_html=True)

    if not api_key:
        st.warning("Please enter your Google API key in the sidebar.")
    else:
        prompt = ChatPromptTemplate(
            messages=[
                SystemMessagePromptTemplate.from_template(
                    "You are a helpful AI assistant. Please respond to user queries in English."
                ),
                MessagesPlaceholder(variable_name="chat_history"),
                HumanMessagePromptTemplate.from_template("{question}"),
            ]
        )

        msgs = StreamlitChatMessageHistory(key="langchain_messages")
        model = ChatGoogleGenerativeAI(model=model_choice, google_api_key=api_key)
        chain = prompt | model | StrOutputParser()

        chain_with_history = RunnableWithMessageHistory(
            chain,
            lambda session_id: msgs,
            input_messages_key="question",
            history_messages_key="chat_history",
        )

        if user_input or uploaded_image:
            st.session_state.chat_history.append({"role": "user", "content": user_input if user_input else "(Image attached)"})
            st.markdown(f"<div class='user-bubble'>{user_input if user_input else '(Image attached)'}</div>", unsafe_allow_html=True)

            with st.chat_message("assistant"):
                message_placeholder = st.empty()
                full_response = ""
                # Typing animation placeholder
                message_placeholder.markdown(
                    "<div class='ai-bubble'><div class='typing'><span>●</span><span>●</span><span>●</span></div></div>",
                    unsafe_allow_html=True
                )

                try:
                    if uploaded_image:
                        human_msg = HumanMessage(
                            content=[
                                {"type": "text", "text": user_input if user_input else "Describe this image"},
                                {"type": "image_url", "image_url": f"data:image/jpeg;base64,{b64}"}
                            ]
                        )
                        response = model.invoke([human_msg])
                        full_response = response.content
                        st.session_state.chat_history.append({"role": "assistant", "content": full_response})
                        message_placeholder.markdown(f"<div class='ai-bubble'>{full_response}</div>", unsafe_allow_html=True)
                    else:
                        config = {"configurable": {"session_id": "any"}}
                        response = chain_with_history.stream({"question": user_input}, config)
                        for res in response:
                            full_response += res or ""
                            message_placeholder.markdown(f"<div class='ai-bubble'>{full_response}▌</div>", unsafe_allow_html=True)
                        st.session_state.chat_history.append({"role": "assistant", "content": full_response})
                        message_placeholder.markdown(f"<div class='ai-bubble'>{full_response}</div>", unsafe_allow_html=True)

                except Exception as e:
                    message_placeholder.markdown(f"<div class='ai-bubble'>Error: {e}</div>", unsafe_allow_html=True)

    # Auto-scroll
    st.markdown("""
    <script>
    var chatBox = window.parent.document.querySelector('.stApp');
    chatBox.scrollTop = chatBox.scrollHeight;
    </script>
    """, unsafe_allow_html=True)

with col2:
    st.header("About")
    st.markdown("- Stylish, animated chat bubbles")
    st.markdown("- One file uploader for both header display & AI analysis")
    st.markdown("- Supports text & image-based queries")
    st.markdown("- Persistent chat history")
    st.info("Requires valid Google Gemini API key with Vision access.")

# Footer
st.markdown("---")
st.markdown("Made with ❤️ — AI Chat & Image Assistant")
