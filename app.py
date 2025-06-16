import os
import streamlit as st
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.chat_models import ChatOpenAI
from pinecone import Pinecone
from sentence_transformers import SentenceTransformer
from langchain.chat_models import ChatOpenAI

# Load environment variables (PINECONE_API_KEY, etc.)
from dotenv import load_dotenv
load_dotenv()

# Initialize Pinecone
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
index = pc.Index(os.getenv("PINECONE_INDEX_NAME"))

# Embedding model (must match the dimension of your Pinecone index)
embed_model = SentenceTransformer("all-MiniLM-L6-v2")  # 384 dims

# LLM (TinyLLaMA running locally)
llm = ChatOpenAI(
    model_name="tinyllama",
    openai_api_base="http://localhost:1234/v1",
    openai_api_key="not-needed",
    temperature=0.5
)


# Prompts
prompt_direct = ChatPromptTemplate.from_template(
    "Answer the query directly based on the following retrieved context.\nQuery: {question}\nContext: {context}\nAnswer:"
)

prompt_hyde = ChatPromptTemplate.from_template(
    "Generate a concise description of a professional's skills and role based on the query.\nQuery: {question}\nDescription: A professional with the following qualifications:"
)

# HyDE pipeline to generate query context
generate_docs_for_retrieval = prompt_hyde | llm | StrOutputParser()

# Query Pinecone

def retrieve_docs(hypothetical_doc, top_k=5):
    vector = embed_model.encode(hypothetical_doc).tolist()
    results = index.query(vector=vector, top_k=top_k, include_metadata=True)
    docs = []
    for match in results['matches']:
        docs.append({
            "id": match['id'],
            "content": match['metadata'].get('text', ''),
            "metadata": match['metadata'],
            "score": match['score']
        })
    return docs

# Full pipeline
def retrieve_and_answer(query):
    hypothetical_doc = generate_docs_for_retrieval.invoke({"question": query})
    retrieved_docs = retrieve_docs(hypothetical_doc)
    context = "\n".join([doc['content'] for doc in retrieved_docs])
    answer = (prompt_direct | llm | StrOutputParser()).invoke({"question": query, "context": context})
    return answer, hypothetical_doc, retrieved_docs

# Streamlit UI
st.set_page_config(page_title="GrantU Chat Assistant", layout="wide")

with st.sidebar:
    st.title("⚙️ Settings")
    model = st.selectbox("LLM Model", ["TinyLLaMA", "Custom"])
    st.caption("GrantBot v1.0")

st.markdown("<h1 style='font-size: 36px;'>🎓 GrantU Chat Assistant</h1>", unsafe_allow_html=True)

# Suggested Prompts
preset_prompts = [
    "List mentors in Molecular Biology with 10+ years experience.",
    "Share mentors from Meta or Microsoft.",
    "Tell me 4 mentors with 12+ years in Investment Banking.",
    "Show mentors in Data Science and ML."
]

# Mentor Profiles Display
mentor_profiles = [
    {"name": "Aditi Sharma", "role": "Data Scientist, Google", "skills": "ML, DL", "photo": "https://randomuser.me/api/portraits/women/44.jpg"},
    {"name": "Rohan Mehta", "role": "Chef, ITC", "skills": "Knife Skills", "photo": "https://randomuser.me/api/portraits/men/46.jpg"},
    {"name": "Sneha Patel", "role": "Cosmetologist, MARS", "skills": "Skin Analysis", "photo": "https://randomuser.me/api/portraits/women/65.jpg"}
]

cols = st.columns(3)
for i, mentor in enumerate(mentor_profiles):
    with cols[i]:
        st.image(mentor["photo"], width=100)
        st.markdown(f"**{mentor['name']}**")
        st.caption(mentor["role"])
        st.markdown(f"*{mentor['skills']}*")


cols = st.columns(len(preset_prompts))
for i, prompt in enumerate(preset_prompts):
    if cols[i].button(prompt):
        st.session_state.suggested_prompt = prompt

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

for msg in st.session_state.chat_history:
    with st.chat_message(msg["role"]):
        st.markdown(msg["text"])

suggestion = st.session_state.pop("suggested_prompt", None)
if suggestion:
    st.chat_message("user").markdown(suggestion)
    st.session_state.chat_history.append({"role": "user", "text": suggestion})
    answer, hypo_doc, retrieved = retrieve_and_answer(suggestion)
    st.chat_message("assistant").markdown(f"**LLM Answer:** {answer}\n\n---\n**Hypothetical Doc:** {hypo_doc}")
    for doc in retrieved:
        st.info(f"**{doc['content']}**\n_Matched Metadata_: {doc['metadata']}\n_Score_: {round(doc['score'], 3)}")
    st.session_state.chat_history.append({"role": "assistant", "text": answer})

user_input = st.chat_input("Ask your question here...")
if user_input:
    st.chat_message("user").markdown(user_input)
    st.session_state.chat_history.append({"role": "user", "text": user_input})
    answer, hypo_doc, retrieved = retrieve_and_answer(user_input)
    st.chat_message("assistant").markdown(f"**LLM Answer:** {answer}\n\n---\n**Hypothetical Doc:** {hypo_doc}")
    for doc in retrieved:
        st.info(f"**{doc['content']}**\n_Matched Metadata_: {doc['metadata']}\n_Score_: {round(doc['score'], 3)}")
    st.session_state.chat_history.append({"role": "assistant", "text": answer})
