import streamlit as st
import pandas as pd
import joblib
import numpy as np
import ollama

# Load the trained model
model = joblib.load("random_forest_model.pkl")

# Define feature input fields
def user_input_features():
    st.sidebar.header("Enter Student Information")
    
    gender = st.sidebar.selectbox("Gender", ["male", "female"])
    race_ethnicity = st.sidebar.selectbox("Race/Ethnicity", ["group A", "group B", "group C", "group D", "group E"])
    parental_education = st.sidebar.selectbox("Parental Level of Education", 
                                             ["some high school", "high school", "some college", "associate's degree", "bachelor's degree", "master's degree"])
    lunch = st.sidebar.selectbox("Lunch Type", ["standard", "free/reduced"])
    test_prep = st.sidebar.selectbox("Test Preparation Course", ["none", "completed"])
    
    reading_score = st.sidebar.slider("Reading Score", 0, 100, 50)
    writing_score = st.sidebar.slider("Writing Score", 0, 100, 50)
    
    data = {
        "gender": [gender],
        "race_ethnicity": [race_ethnicity],
        "parental_level_of_education": [parental_education],
        "lunch": [lunch],
        "test_preparation_course": [test_prep],
        "reading_score": [reading_score],
        "writing_score": [writing_score]
    }
    return pd.DataFrame(data)

st.title("EduLytics: Student Math Score Prediction")
st.write("Enter student details to predict their math score.")

input_data = user_input_features()

# Prediction Section
if st.button("Predict Score"):
    prediction = model.predict(input_data)[0]
    st.success(f"Predicted Math Score: {round(prediction, 2)}")

# ---- Chatbot Section ----
st.header("EduLytics Chatbot")
st.write("Ask any questions related to student performance!")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display previous messages
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

# User input for chatbot
user_input = st.text_input("Type your message here:")
send_button = st.button("Send")

if send_button and user_input:
    # Append user message to session state
    st.session_state.messages.append({"role": "user", "content": user_input})

    # Generate chatbot response
    response = ollama.chat(model='tinyllama', messages=st.session_state.messages)
    ai_reply = response['message']['content']
    
    # Append bot response
    st.session_state.messages.append({"role": "assistant", "content": ai_reply})

    # Display bot response
    with st.chat_message("assistant"):
        st.write(ai_reply)


