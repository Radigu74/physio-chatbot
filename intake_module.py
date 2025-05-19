
import streamlit as st
import json

def run_physio_intake(name, email, company, phone, country, log_to_google_sheets_fn):
    st.subheader("🧾 Physiotherapy Intake Form")

    intake = {}

    intake["region"] = st.selectbox("Where is the issue located?", 
        ["Neck", "Back", "Shoulder", "Elbow", "Hand/Wrist", "Hip", "Knee", "Ankle/Foot", "Other"])

    intake["duration"] = st.radio("When did the issue start?", 
        ["< 1 week", "1–4 weeks", "1–3 months", "> 3 months"])

    intake["onset"] = st.radio("How did it start?", 
        ["Suddenly (injury)", "Gradually", "After surgery", "Unknown"])

    intake["symptoms"] = st.multiselect("How does it feel?", 
        ["Sharp", "Dull ache", "Tingling", "Burning", "Stiffness", "No pain"])

    intake["pain_level"] = st.slider("Pain level (0 = none, 10 = worst)", 0, 10, 5)

    intake["worsening_factors"] = st.text_input("What makes it worse?")

    intake["relieving_factors"] = st.text_input("What helps relieve it?")

    intake["activities_affected"] = st.multiselect("Which activities are affected?", 
        ["Sleep", "Walking", "Work", "Exercise", "Driving", "Dressing"])

    intake["prior_injury"] = st.radio("Any previous injury or surgery in this area?", ["Yes", "No"])

    intake["goals"] = st.text_input("What is your goal with physiotherapy?")

    intake["red_flags"] = st.multiselect("Any of these symptoms?", 
        ["Night pain", "Groin numbness", "Weight loss", "Bladder/Bowel issues", "Fever", "None of the above"])

    intake["extra_notes"] = st.text_area("Anything else we should know?")

    if st.button("✅ Submit Intake"):
        # Save to session state
        st.session_state.intake = intake

        # Log to Google Sheets
        log_to_google_sheets_fn({
            "name": name,
            "email": email,
            "company": company,
            "phone": phone,
            "country": country,
            "question": "[Physio Intake]",
            "response": json.dumps(intake)
        })

        st.success("✅ Intake submitted. You may now chat with the assistant.")
        st.session_state.chat_enabled = True
        st.session_state.physio_mode = False
