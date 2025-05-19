import os
import openai
import streamlit as st
import re
from dotenv import load_dotenv, find_dotenv
import numpy as np
import faiss
import pycountry
import csv
import logging
from openai import OpenAIError, RateLimitError
import json
import datetime
from google.oauth2.credentials import Credentials
from google.auth.transport.requests import Request
from gspread.auth import authorize
import uuid
from flask import Flask, request, jsonify
from intake_module import run_physio_intake

# =============================
# Load environment variables
# ============================
_ = load_dotenv(find_dotenv())

st.markdown("""
<style>
/* === USER MESSAGES === */
div[data-testid="stChatMessage"] div:has(div:has(img[alt="👤"])) {
    justify-content: flex-end;
    text-align: right;
}
div[data-testid="stChatMessage"] div:has(div:has(img[alt="👤"])) > div:nth-child(2) {
    background-color: #dbe9f4;
    border-radius: 12px;
    padding: 10px 15px;
    margin-bottom: 10px;
    max-width: 80%;
}

/* === ASSISTANT MESSAGES === */
div[data-testid="stChatMessage"] div:has(div:has(img[alt="🌍"])) {
    justify-content: flex-start;
    text-align: left;
}
div[data-testid="stChatMessage"] div:has(div:has(img[alt="🌍"])) > div:nth-child(2) {
    background-color: #f1f0f0;
    border-radius: 12px;
    padding: 10px 15px;
    margin-bottom: 10px;
    max-width: 80%;
}
</style>
""", unsafe_allow_html=True)

# ===================
# OpenAI API Key
# ===================
openai.api_key = os.getenv("OPENAI_API_KEY")

# ===========================
# For debugging purposes
# ===========================
print("OPENAI_API_KEY:", os.getenv("OPENAI_API_KEY"))

def authenticate_google_sheets():
    creds = Credentials(
        None,
        refresh_token=os.getenv("GOOGLE_REFRESH_TOKEN"),
        token_uri=os.getenv("GOOGLE_TOKEN_URI"),
        client_id=os.getenv("GOOGLE_CLIENT_ID"),
        client_secret=os.getenv("GOOGLE_CLIENT_SECRET"),
        scopes=[
            "https://www.googleapis.com/auth/spreadsheets",
            "https://www.googleapis.com/auth/drive"
        ]
    )
    creds.refresh(Request())
    client = authorize(creds)
    return client

# ================================
# Logging Function to Google Sheet
# ================================
def log_to_google_sheets(data):
    try:
        client = authenticate_google_sheets()
        sheet = client.open("Chatlogs Terrapeak").sheet1

        row = [
            datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            data.get("name", ""),
            data.get("email", ""),
            data.get("company", ""),
            data.get("phone", ""),
            data.get("country", ""),
            data.get("question", ""),
            data.get("response", ""),
            data.get("intent", ""),
            data.get("cta_triggered", ""),
            data.get("message_number", ""),
            data.get("session_id", "")
        ]

        sheet.append_row(row)
        return True

    except Exception as e:
        print(f"[Google Sheets Logging Error] {e}")
        return False


# ====================================================
# Hide Streamlit's default menu, header, and footer
# ====================================================
hide_st_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            header {visibility: hidden;}
            </style>
            """
st.markdown(hide_st_style, unsafe_allow_html=True)

# ====================================================
# STEP 1: Define and Store Your Articles (RAG Source)
# ====================================================
# Your optimized TerraPeak launch article is stored here.
articles = [
    {
        "title": "TerraPeak Official Launch",
        "content": """March 5, 2025 – Singapore        
    TerraPeak Consulting officially launches, offering expert-led market expansion, sales growth strategies, and practical AI integration to global businesses. Specializing in APAC market entry and growth support for Asian SMEs and family businesses, TerraPeak aims to redefine strategic growth.
    Founded by experienced market and sales strategists, TerraPeak combines exploration with sustainable, strategic growth. With proven expertise, TerraPeak guides companies in harnessing AI to improve sales and operational efficiency.
    Core Offerings:
    - Expert Market Expansion into APAC
    - Revenue-Driven Sales Growth
    - Seamless AI Integration
    - Family Business Growth & Transformation
    Committed to responsible, ethical, and sustainable growth, TerraPeak offers tailored solutions ensuring long-term success and resilience. Businesses seeking expansion, transformation, and innovation are encouraged to reach out via connect@terrapeakgroup.com."""
    },
    {
        "title": "Unlocking Opportunities: A Guide to Doing Business in Asia",
        "content": """Asia’s markets are diverse, each with distinct cultures, regulations, and consumer preferences. Successful market entry requires careful planning and cultural understanding.
    1. Recognize Diversity: Each Asian market differs significantly. Independent research on consumer preferences, economic conditions, and regulatory landscapes is crucial.
    2. Understand Cultural Nuances: Personal relationships and trust-building are essential. Face-to-face interactions and awareness of local business etiquette enhance partnership opportunities.
    3. Navigate Regulations: Legal frameworks vary widely. Consulting local legal experts helps ensure compliance and protection, particularly for intellectual property rights.
    4. Adapt Products and Services: Localization involves more than translation; products, pricing strategies, and marketing channels should align with local tastes and usage patterns.
    5. Leverage Local Partnerships: Strategic partnerships offer invaluable market insights, reduce entry costs, and minimize risks associated with unfamiliar markets.
    6. Invest in Talent and Training: Hiring skilled local talent and providing basic cross-cultural training ensures smooth operations and effective market penetration.
    7. Stay Agile and Innovative: Regularly reassessing market trends and technological advancements allows businesses to remain competitive and responsive in dynamic Asian markets."""
    },
    {
        "title": "AI & SMEs: 10 Key Stats Revealing Growth, Challenges, and Opportunities",
        "content": """Artificial Intelligence (AI) is rapidly changing how SMEs and family businesses operate, offering significant productivity gains, enhanced customer engagement, and cost efficiencies. Adoption among SMEs is growing quickly, with many businesses already using AI-powered solutions like chatbots, social media automation, and generative AI.
    SMEs widely recognize AI’s benefits, including improved efficiency, automated marketing, sales forecasting, and better customer service. However, common concerns include knowledge gaps, high initial costs, uncertainty about return on investment (ROI), cybersecurity, and data privacy.
    Practical, user-friendly AI solutions designed specifically for SMEs are making adoption easier. Cloud-based AI services (AI-as-a-Service) and generative AI tools have increased accessibility, allowing SMEs to automate processes, create engaging content, and enhance productivity without large upfront investments.
    To fully leverage AI’s potential, SMEs should:
    - Develop clear AI adoption strategies and roadmaps.
    - Establish measurable KPIs to track AI effectiveness.
    - Use cost-effective AI tools tailored to their specific business needs.
    SMEs strategically adopting AI gain a competitive edge, achieve sustainable growth, and drive long-term efficiency."""
    }
]

# ============================================================
# STEP 2: Create an Embedding Function Using a Client Instance
# ============================================================
def get_embedding(text, model="text-embedding-3-small"):
    """
    Generate a numeric embedding for a given text using OpenAI's new SDK (v1.x).
    """
    if not text or not isinstance(text, str) or not text.strip():
        raise ValueError("Text for embedding must be a non-empty string.")

    client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    response = client.embeddings.create(
        input=text.strip(),
        model=model
    )
    
    embedding = response.data[0].embedding
    return np.array(embedding)
# ===================================================================
# STEP 3: Generate Embeddings for the Articles and Build FAISS Index
# ===================================================================
# Generate embeddings for each article
article_embeddings = [
    get_embedding(article["content"])
    for article in articles
    if article.get("content") and isinstance(article["content"], str) and article["content"].strip()
]

# Determine the dimensionality of the embeddings
embedding_dim = len(article_embeddings[0])

# Create a FAISS index (using L2 distance)
index = faiss.IndexFlatL2(embedding_dim)

# Convert embeddings to a NumPy array of type float32
embeddings_np = np.array(article_embeddings).astype('float32')
index.add(embeddings_np)
print("FAISS index created with", index.ntotal, "articles.")

# ====================================================================
# STEP 4: Create a Function to Retrieve Relevant Articles for a Query
# ====================================================================
def retrieve_relevant_articles(query, k=2):
    """
    Retrieve the indices and distances of the k most relevant articles for the given query.
    Includes error handling to avoid crashes on embedding or index issues.
    """
    try:
        # Generate an embedding for the query text
        query_embedding = get_embedding(query).astype('float32')
        query_embedding = np.expand_dims(query_embedding, axis=0)  # FAISS requires a 2D array

        # Search the FAISS index for the top-k similar articles
        distances, indices = index.search(query_embedding, k)

        return indices[0], distances[0]

    except Exception as e:
        print(f"[Error] Failed to retrieve relevant articles: {e}")
        return [], []

# ============================================================
# STEP 5: Build a Prompt that Integrates the Retrieved Context
# ============================================================
def build_prompt_with_context(user_query, k=2):
    """
    Build a prompt that includes trimmed article context for faster GPT responses.
    """
    indices, _ = retrieve_relevant_articles(user_query, k)

    labeled_contexts = []
    for i in indices:
        article = articles[i]
        trimmed_content = article["content"][:1000]  # Limit content to avoid long prompts
        labeled_context = f"Source: {article['title']}\n{trimmed_content}"
        labeled_contexts.append(labeled_context)

    full_context = "\n\n".join(labeled_contexts)

    prompt = (
        f"You are an AI assistant responding to the user's question using the most relevant context below.\n"
        f"Use the sources to support your answer clearly.\n\n"
        f"{full_context}\n\n"
        f"User Question: {user_query}\n\n"
        f"Answer:"
    )

    return prompt

# ================================================================
# CUSTOM UI: Inject custom CSS for styling using Terrapeak colors 
# ================================================================
st.markdown(
    """
    <style>
    /* Global Page Background */
    .reportview-container, .main {
        background-color: #f4f4f2;
    }
    /* Header styling */
    .header {
        background-color: #E0E0DB;
        padding: 10px;
        border-radius: 10px;
        text-align: center;
        margin-bottom: 20px;
    }
    .header img {
        width: 50px;
        height: 50px;
        vertical-align: middle;
    }
    .header h1 {
        display: inline;
        margin-left: 10px;
        vertical-align: middle;
        color: #131313;
        font-family: sans-serif;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ============================
# Session State Initialization
# ============================
# Ensure session_id is initialized for tracking
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())[:8]

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
    
if "chat_enabled" not in st.session_state:
    st.session_state.chat_enabled = False  # Set to True to allow input field to appear

# Initialize chat context
if "chat_context" not in st.session_state:
    st.session_state.chat_context = [
        {'role': 'system', 'content': """
You are Fysio, the professional virtual assistant of **MoveWell Physiotherapy & Rehab Centre** —an expert-led physical therap clinic specializing in all common physiotherapy injuries and sports related injuries.
Your personality reflects **MoveWell Physiotherapy & Rehab Centre** values: clear, confident, helpful, and grounded in real-world expertise. You speak in a friendly and professional tone—always aiming to guide visitors with clarity, empathy, and practical questions. You are knowledgeable, supportive, and service-oriented.
**Important:** Always respond in the same language as the user’s question. If the user asks in Dutch (or any other language), reply in that language. If the user switches language mid-conversation, adjust your language accordingly.

- Greet new users professionally and warmly
- Offer to guide them through a brief **physio intake questionnaire**
- Log their answers to assist physiotherapists in pre-assessment
- Explain basic background on common conditions when asked
- Provide polite suggestions and next steps, including booking

🏥 About the Clinic:
MoveWell helps patients restore healthy movement, recover from injury, and prevent future issues. Services include:
- Orthopedic physiotherapy (e.g., knee, back, shoulder)
- Sports rehab and performance programs
- Post-surgical recovery plans
- Pain management and chronic condition care
- Ergonomics and lifestyle advice

🧠 Conditions You Commonly See:
1. **Knee Osteoarthritis**  
   → Pain with stairs, walking, and standing. Often in adults 45+ with joint stiffness and inflammation. Treated with strength training, manual therapy, and mobility work.

2. **Frozen Shoulder (Adhesive Capsulitis)**  
   → Progressive shoulder stiffness and pain, especially when reaching overhead or behind. Most common in adults 40–60, sometimes following trauma or inactivity.

3. **Post-Surgical ACL Rehab**  
   → Typically in younger patients post-knee surgery. Key issues are weakness, balance, and return-to-sport concerns. Focused on progressive loading and movement control.

🩺 Intake Workflow:
If a user mentions pain, injury, referral or stiffness:
→ Offer to “start a quick intake”
→ Ask 10–12 structured questions about symptoms, history, and goals
→ Store answers under `st.session_state.intake` or log to Google Sheets

💬 Tone of Voice:
Professional, calm, supportive. Use plain language to explain conditions.
Never offer a diagnosis — always suggest follow-up with a licensed physiotherapist.

🤖 Interaction Rules:
If someone says “Hi”, “Hello”, “How are you?”, or anything casual—respond warmly and professionally, and offer to help. Example replies:
“Hi there! 👋 I’m Fysio, your virtual assistant here at **MoveWell Physiotherapy & Rehab Centre**. How can I assist you today?”
“Doing great—thanks for asking! What can I help you with today?”
“Nice to meet you too! I can walk you through our services and connect you with a therapist if needed.”

If someone asks "What does Physio clinic do?":
“**MoveWell Physiotherapy & Rehab Centre** helps all patients with common or sports related injuries with professionalism and practical solutions.”

If a user asks for a live chat:
- First ask: “I’d be happy to help—could you share your question here first?”
- If they ask a second time: “No problem, a clinician will get back to you within 1 working day.”
- If it’s urgent: Provide phone number +651234 5678 and email movewell@physio.com.

🌍 Core Services (4 Pillars)
#1 Sports related injuries
#2 common conditions
#3 rehabilitation
#4 After surgery care

🧭 Company Values
- Empathy
- professionalism
- integrity



"""}
    ]

LIVE_CHAT_KEYWORDS = [
    "speak", "talk", "call", "consultant", "real person", "human", "live chat", "contact someone"
]

def detect_intent(user_input: str) -> str:
    system_msg = (
        "You are an assistant that classifies the intent of a user's message. "
        "Return only one of the following: 'handoff', 'general', or 'other'."
    )

    prompt = f"""
Message: "{user_input}"

What is the user's intent? 
Return just one word: handoff, general, or other.
"""

    try:
        response = get_completion_from_messages([
            {"role": "system", "content": system_msg},
            {"role": "user", "content": prompt}
        ])
        return response.strip().lower()
    
    except Exception:
        lowered = user_input.lower()
        if any(keyword in lowered for keyword in LIVE_CHAT_KEYWORDS):
            return "handoff"
        return "general"

# ==============================================
# OpenAI Communication Function (uses Chat API)
# ==============================================
def get_completion_from_messages(user_messages, model="gpt-3.5-turbo-0125", temperature=0, max_history=6):
    try:
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            return "API key is missing. Please check your environment settings."

        client = openai.OpenAI(api_key=api_key)

        # Retain the system prompt and only the last few interactions to reduce token bloat
        preserved_context = [m for m in st.session_state.chat_context if m["role"] == "system"]
        recent_history = st.session_state.chat_context[-max_history:]
        messages = preserved_context + recent_history + user_messages

        response = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature,
            timeout=15  # Set a timeout (in seconds) to avoid long hangs
        )

        return response.choices[0].message.content

    except RateLimitError:
        logging.warning("Rate limit reached. Try again shortly.")
        return "We're handling a high volume of requests right now. Please try again in a moment."

    except OpenAIError as e:
        logging.error(f"OpenAI API error: {e}")
        return "Hmm, something went wrong while reaching our assistant. Please try again shortly."

    except Exception as e:
        logging.exception("Unexpected error occurred.")
        return "Oops, an unexpected error occurred. Please try again or contact support."

# ===========================
# UI PURPOSE for User Details Input
# ===========================
st.markdown(
    """
    <style>
    /* This moves the header text upward */
    .contact-header {
        margin-top: -80px;
        padding-top: 0;
    }
    /* This moves the input fields upward */
    .contact-form {
        margin-top: 0px;  /* Adjust this value as needed */
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Header text moved upward by the .contact-header class
st.markdown('<div class="contact-header">📢 <strong>Enter your contact details before chatting with our AI assistant:</strong></div>', unsafe_allow_html=True)

# Wrap the input fields in a container with the .contact-form class
st.markdown('<div class="contact-form">', unsafe_allow_html=True)

name = st.text_input("Enter your name:", key="name_input")
email = st.text_input("Enter your email:", key="email_input")
company = st.text_input("Enter your company name:", key="company_input")
phone = st.text_input("Enter your phone number:", key="phone_input")
country_list = sorted([country.name for country in pycountry.countries])
country = st.selectbox("Select Country", country_list, key="country_dropdown")

st.markdown('</div>', unsafe_allow_html=True)

def is_valid_email(email):
    return re.match(r"[^@]+@[^@]+\.[^@]+", email)

def is_valid_phone(phone):
    return re.match(r"^\+?\d{10,15}$", phone)

def validate_and_start():
    if not is_valid_email(email):
        return "❌ Invalid email."
    if not is_valid_phone(phone):
        return "❌ Invalid phone number."
    
    st.session_state.chat_enabled = True

    # Log user data to Google Sheets
    log_to_google_sheets({
        "name": name,
        "email": email,
        "company": company,
        "phone": phone,
        "country": country
    })

    return "✅ **Details saved!**"

if st.button("Submit Details", key="submit_button"):
    validation_message = validate_and_start()
    st.markdown(validation_message, unsafe_allow_html=True)

# ========== PHYSIO INTAKE TRIGGER ==========
if "physio_mode" not in st.session_state:
    st.session_state.physio_mode = False

if st.button("🩺 Start Physio Intake"):
    st.session_state.physio_mode = True
    st.session_state.intake = {}

if st.session_state.get("physio_mode", False):
    run_physio_intake(name, email, company, phone, country, log_to_google_sheets)    

    # ✅ Personalized welcome message
    st.session_state.chat_history.append({
        "role": "assistant",
        "content": f"Hi {name}! 👋 I’m Fysio, your virtual assistant here at TerraPeak. How can I help you today?"
    })
   
# ========================================================
# CUSTOM UI: Display Chat History with Styled Chat Bubbles
# =========================================================
st.markdown("---")
st.markdown("**💬 Chat with the Terrapeak Automated Consultant:**")

if st.session_state.chat_enabled:
    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"], avatar="👤" if msg["role"] == "user" else "🌍"):
            st.markdown(msg["content"])
            
# ============================================
# CUSTOM UI: Chat Input Field with Send Button
# ============================================
if st.session_state.chat_enabled:
    user_input = st.chat_input("Type your message here...")

    if user_input:
        with st.chat_message("user", avatar="👤"):
            st.markdown(user_input)

        st.session_state.chat_history.append({
            "role": "user",
            "content": user_input
        })

        # ✅ Track how many messages the user has sent
        message_number = len([
            m for m in st.session_state.chat_history if m["role"] == "user"
        ])

        # 🔍 INTENT DETECTION with GPT + fallback
        intent = detect_intent(user_input)
        print("Detected intent:", intent)  # Optional debug

        if intent == "handoff":
            user_name = name.strip().split(" ")[0].capitalize() if name else "there"

            styled_cta = f"""<div style='
                background-color: #2f5d50;
                color: #ffffff;
                padding: 14px;
                border-radius: 12px;
                text-align: center;
                width: fit-content;
                font-weight: bold;
                font-family: sans-serif;
                margin-top: 10px;
            '>
            📅 <a href="https://calendly.com/terrapeakgroup/terrapeak_group_call" target="_blank" style='color: white; text-decoration: none;'>
                Book a 30-Minute Call with TerraPeak
            </a>
            </div>"""

            with st.chat_message("assistant", avatar="🌍"):
                st.markdown(f"Absolutely, {user_name} 👋 I can connect you with one of our consultants:", unsafe_allow_html=True)
                st.markdown(styled_cta, unsafe_allow_html=True)

                # ✅ LOG that CTA was triggered
            log_to_google_sheets({
                "name": name,
                "email": email,
                "company": company,
                "phone": phone,
                "country": country,
                "question": user_input,
                "response": "[CTA Triggered – No GPT reply]",
                "intent": intent,
                "cta_triggered": "yes",
                "message_number": message_number,
                "session_id": st.session_state.session_id
            })                

            st.stop()  # ✅ Skip GPT if it's a handoff

        # === GPT ASSISTANT RESPONSE ===
        rag_prompt = build_prompt_with_context(user_input.strip(), k=2)
        assistant_response = get_completion_from_messages([{
            "role": "user",
            "content": rag_prompt
        }])

        with st.chat_message("assistant", avatar="🌍"):
            st.markdown(assistant_response)

        st.session_state.chat_history.append({
            "role": "assistant",
            "content": assistant_response
        })

        # === OPTIONAL CTA after 6 messages ===
        user_name = name.strip().split(" ")[0].capitalize() if name else "there"
        recent_user_messages = [m["content"].lower() for m in st.session_state.chat_history if m["role"] == "user"]

        styled_cta = f"""<div style='
            background-color: #2f5d50;
            color: #ffffff;
            padding: 14px;
            border-radius: 12px;
            text-align: center;
            width: fit-content;
            font-weight: bold;
            font-family: sans-serif;
            margin-top: 10px;
        '>
        📅 <a href="https://calendly.com/terrapeakgroup/terrapeak_group_call" target="_blank" style='color: white; text-decoration: none;'>
            Book a 30-Minute Call with TerraPeak
        </a>
        </div>"""

        if len(recent_user_messages) >= 6 and "consultant_offer_shown" not in st.session_state:
            with st.chat_message("assistant", avatar="🌍"):
                st.markdown(f"{user_name}, if you'd prefer to speak directly with a TerraPeak consultant, feel free to book a time below:", unsafe_allow_html=True)
                st.markdown(styled_cta, unsafe_allow_html=True)
            st.session_state.consultant_offer_shown = True

        # ✅ Log to Google Sheets
        log_to_google_sheets({
            "name": name,
            "email": email,
            "company": company,
            "phone": phone,
            "country": country,
            "question": user_input,
            "response": assistant_response,
            "intent": intent,
            "cta_triggered": "no",
            "message_number": message_number,
            "session_id": st.session_state.session_id
        })

# ==============================================
# Flask API endpoint for FB → Chatbot forwarding
# ==============================================
api = Flask(__name__)

@api.route("/endpoint", methods=["POST"])
def chatbot_endpoint():
    payload = request.get_json(silent=True) or {}
    user_message = payload.get("message", "")
    if not user_message:
        return jsonify({"error": "No message provided"}), 400

    # Build RAG prompt + get GPT response
    rag = build_prompt_with_context(user_message, k=2)
    reply = get_completion_from_messages([{"role": "user", "content": rag}])

    return jsonify({"reply": reply})

if __name__ == "__main__":
    # When you run `python main.py`, Streamlit will take over.
    # To run the Flask API, use gunicorn: `gunicorn main:api`
    pass
