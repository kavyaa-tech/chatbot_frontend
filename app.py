import streamlit as st
import requests


st.set_page_config(page_title="GrantU Chat Assistant", layout="wide")



def query_local_llm(prompt, system_prompt, model):
    url = "http://127.0.0.1:1234/v1/chat/completions"
    headers = {"Content-Type": "application/json"}

    model_map = {
        "LLaMA 2": "llama-2-7b-chat",
        "LLaMA 3": "llama-3-8b-chat",
        "Mistral": "mistral-7b-instruct",
        "Custom": "tinyllama-1.1b-chat-v1.0"
    }

    selected_model = model_map.get(model, "tinyllama-1.1b-chat-v1.0")

    data = {
        "model": selected_model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.7,
        "max_tokens": 300
    }

    try:
        response = requests.post(url, headers=headers, json=data)
        response.raise_for_status()
        return response.json()["choices"][0]["message"]["content"]
    except Exception as e:
        return f"Error: {e}"


system_prompt = """
You are GrantBot, an education assistant for GrantU.

Help users with:
- Scholarships (finding, applying, eligibility, tips)
- Mentorship (how to connect, benefits)
- Application help (essays, documents)

If asked anything else, kindly guide them back to these topics.
"""

with st.sidebar:
    st.title("⚙️ Settings")
    model = st.selectbox("LLM Model", ["LLaMA 2", "LLaMA 3", "Mistral", "Custom"])
    st.caption("GrantBot v1.0")


st.markdown("<h1 style='font-size: 36px;'>🎓 GrantU Chat Assistant</h1>", unsafe_allow_html=True)
st.markdown("<p style='font-size: 18px; color: gray;'>Ask about scholarships, mentorship, or application guidance.</p>", unsafe_allow_html=True)


st.markdown("### Featured Mentors")

mentor_profiles = [
    {
        "name": "Aditi Sharma",
        "role": "Data Scientist, Google",
        "skills": "Machine Learning, Deep Learning",
        "photo": "https://randomuser.me/api/portraits/women/44.jpg"
    },
    {
        "name": "Rohan Mehta",
        "role": "Chef, ITC",
        "skills": "Knife Skills, Flavor Balancing",
        "photo": "https://randomuser.me/api/portraits/men/46.jpg"
    },
    {
        "name": "Sneha Patel",
        "role": "cosmeticsitst, MARS Cosmetics",
        "skills": "Skin Analysis, Precision Application",
        "photo": "https://randomuser.me/api/portraits/women/65.jpg"
    }
]

cols = st.columns(3)
for i, mentor in enumerate(mentor_profiles):
    with cols[i]:
        st.image(mentor["photo"], width=100)
        st.markdown(f"**{mentor['name']}**")
        st.caption(mentor["role"])
        st.markdown(f"*{mentor['skills']}*")


preset_prompts = [
    "List mentors who specialize in Molecular Biology with over 10 years of mentoring experience.",
    "Share profiles of mentors who’ve worked with Meta or Microsoft.",
    "Tell me 4 mentors with 12+ years experience in Investment Banking.",
    "Show me mentors with experience in data science and machine learning for undergraduate students."
]

cols = st.columns(len(preset_prompts))
for i, prompt in enumerate(preset_prompts):
    if cols[i].button(prompt):
        st.session_state.suggested_prompt = prompt


st.markdown("")

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

for msg in st.session_state.chat_history:
    with st.chat_message(msg["role"]):
        st.markdown(msg["text"])

suggestion = st.session_state.pop("suggested_prompt", None)
if suggestion:
    st.chat_message("user").markdown(suggestion)
    st.session_state.chat_history.append({"role": "user", "text": suggestion})

    bot_reply = query_local_llm(suggestion, system_prompt, model)
    st.chat_message("assistant").markdown(bot_reply)
    st.session_state.chat_history.append({"role": "assistant", "text": bot_reply})

user_input = st.chat_input("Ask your question here...")
if user_input:
    st.chat_message("user").markdown(user_input)
    st.session_state.chat_history.append({"role": "user", "text": user_input})

    response = query_local_llm(user_input, system_prompt, model)
    st.chat_message("assistant").markdown(response)
    st.session_state.chat_history.append({"role": "assistant", "text": response})
