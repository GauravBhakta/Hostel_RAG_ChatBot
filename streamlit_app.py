# Import necessary libraries
import streamlit as st
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import re
from langchain_groq import ChatGroq
import os
from dotenv import load_dotenv

# Set API key for GROQ
load_dotenv(dotenv_path="D:/UPES/Semester 2/Data Mining/GROQ_API_KEY.env")

# Add your data + chunking logic here
hostel_data = {
    "rules": """
1. No smoking or alcohol allowed in the hostel premises.
2. Visitors are not allowed after 9 PM.
3. Keep the hostel clean and tidy.
4. Respect the privacy of other residents.
5. Report any maintenance issues to the hostel management.
6. Silence hours after 10:30 PM.
7. No pets allowed in the hostel.
8. Follow the check-in and check-out timings strictly.
9. Use of electrical appliances is prohibited in the rooms.
10. Any damage to hostel property will be charged to the resident.
    """,
    "facilities": """
1. Common kitchen available for all residents.
2. Laundry facilities provided.
3. Wi-Fi connectivity throughout the hostel.
4. Study area with seating and desks.
5. Gym or fitness center available.
    """,
    "mess": """
1. Breakfast served from 7 AM to 9 AM.
2. Lunch served from 12 PM to 2 PM.
3. Dinner served from 7 PM to 9 PM.
4. Special meals available on request (vegetarian, vegan, etc.).
5. Non-vegetarian meals include chicken, and mutton served on Wednesdays and Sundays.
""",
    "complaints": """
1. Complaints can be registered by scanning the QR code available on each floor.
2. Complaints will be addressed within 24 hours of submission.
3. Residents are encouraged to resolve issues through direct communication with the management if problem is not solved within 24 hours.
    """,
    "wifi":"""
1. Wi-Fi is available 24/7 for all residents.
2. Wi-Fi password is provided at the time of check-in.
3. For any connectivity issues, please contact the IT Department in School of Law at 4th floor.
4. Usage of Wi-Fi for illegal activities is strictly prohibited and may lead to disciplinary action.
"""
}

def create_chunks(data):
    documents = []
    for category, text in data.items():
        lines = text.strip().split("\n")
        for line in lines:
            line = line.strip()
            if line == "":
                continue
            line = re.sub(r'^\d+\.\s*', '', line)
            documents.append(f"{category}: {line}")
    return documents

# Inititialize model and create FAISS index
documents = create_chunks(hostel_data)
@st.cache_resource
def setup_rag(documents):
    model = SentenceTransformer('all-MiniLM-L6-v2')
    embeddings = model.encode(documents)

    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(np.array(embeddings))

    return model, index
model, index = setup_rag(documents)
# Retrival Function
def get_relevant_context(query):
    query_embedding = model.encode([query])
    distances, indices = index.search(query_embedding, k=3)
    results = [documents[idx] for idx in indices[0]]
    return "\n".join(results)

# LLM Setup
llm = ChatGroq(model = "llama-3.1-8b-instant")

# Response Function
def chatbot_response(query):
    context = get_relevant_context(query)

    prompt = f"""
You are a hostel assistant.

STRICT RULES:
- Answer ONLY using the provided context.
- DO NOT add new information.
- If answer is not present, say:
"Please contact the hostel office for more information."

Context:
{context}

Question:
{query}
"""

    response = llm.invoke(prompt)
    return response.content

# Build Streamlit Interface
st.set_page_config(page_title="Hostel RAG Chatbot")

st.title("Hostel RAG Chatbot")
st.markdown("""
### How it works
This chatbot uses:
- **Embeddings** to convert text into vectors  
- **FAISS** for similarity search  
- **LLM (Groq)** to generate responses  
""")
st.markdown("### Try asking:")
st.write("- When do we eat at night?")
st.write("- Is smoking allowed?")
st.write("- How do I complain?")
query = st.text_input("Ask your question:")

if query:
    st.write("Searching...")
    answer = chatbot_response(query)
    st.success(answer)



