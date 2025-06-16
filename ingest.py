import os
import uuid
import psycopg2
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from pinecone import Pinecone

# Load environment variables
load_dotenv()

# PostgreSQL config
db_params = {
    'dbname': os.getenv("DB_NAME"),
    'user': os.getenv("DB_USER"),
    'password': os.getenv("DB_PASSWORD"),
    'host': os.getenv("DB_HOST"),
    'port': os.getenv("DB_PORT", "5432")
}

# Pinecone config
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME")

# Load embedding model
print("📦 Loading embedding model...")
model = SentenceTransformer("all-MiniLM-L6-v2")

# Initialize Pinecone
print("🔌 Connecting to Pinecone...")
pc = Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index(PINECONE_INDEX_NAME)
print("✅ Connected to Pinecone index.")

def fetch_profiles():
    print("📄 Fetching profiles from PostgreSQL...")
    try:
        conn = psycopg2.connect(**db_params)
        cursor = conn.cursor()
        cursor.execute("""
            SELECT full_name, years_exp, current_org, past_org, skill_set, linkedin_profile FROM profiles
        """)
        rows = cursor.fetchall()
        cursor.close()
        conn.close()
        print(f"✅ Retrieved {len(rows)} profiles.")
        return rows
    except Exception as e:
        print(f"❌ Database error: {e}")
        return []

def embed_texts(profiles):
    print("🧠 Generating embeddings...")
    texts = [
        f"Name: {p[0]}, Years of Experience: {p[1]}, Current Org: {p[2]}, Past Orgs: {p[3]}, Skills: {p[4]}, LinkedIn: {p[5]}"
        for p in profiles
    ]
    embeddings = model.encode(texts, show_progress_bar=True)
    return embeddings, texts

def batch_upsert_to_pinecone(profiles, embeddings):
    print("📤 Uploading to Pinecone...")
    batch_size = 100
    vectors = []
    for i, (profile, embed) in enumerate(zip(profiles, embeddings)):
        metadata = {
            "name": profile[0],
            "years_exp": profile[1],
            "current_org": profile[2],
            "past_org": profile[3],
            "skills": profile[4],
            "linkedin": profile[5]
        }
        vector = {
            "id": str(uuid.uuid4()),
            "values": embed.tolist(),
            "metadata": metadata
        }
        vectors.append(vector)

        if len(vectors) == batch_size or i == len(profiles) - 1:
            index.upsert(vectors=vectors)
            print(f"📦 Upserted batch of {len(vectors)}")
            vectors = []

if __name__ == "__main__":
    profiles = fetch_profiles()
    if profiles:
        embeddings, _ = embed_texts(profiles)
        batch_upsert_to_pinecone(profiles, embeddings)
        print("✅ Ingestion completed.")
    else:
        print("⚠️ No profiles to ingest.")
