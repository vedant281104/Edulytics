import streamlit as st
import pandas as pd
import joblib
import numpy as np
import ollama
import sqlite3

# Initialize database
def init_db():
    conn = sqlite3.connect("users.db")
    c = conn.cursor()
    c.execute("CREATE TABLE IF NOT EXISTS users (id INTEGER PRIMARY KEY, username TEXT, password TEXT)")
    conn.commit()
    conn.close()

# User authentication
def register_user(username, password):
    conn = sqlite3.connect("users.db")
    c = conn.cursor()
    c.execute("INSERT INTO users (username, password) VALUES (?, ?)", (username, password))
    conn.commit()
    conn.close()

# Validate user login
def validate_user(username, password):
    conn = sqlite3.connect("users.db")
    c = conn.cursor()
    c.execute("SELECT * FROM users WHERE username = ? AND password = ?", (username, password))
    user = c.fetchone()
    conn.close()
    return user

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

st.title("EduLytics: Student Score Prediction")
st.write("Enter student details to predict their score.")

# User Authentication Section
if "authenticated" not in st.session_state:
    st.session_state.authenticated = False
    st.session_state.username = ""

menu = ["Login", "Register"] if not st.session_state.authenticated else ["Predict Score", "Chatbot"]
choice = st.sidebar.selectbox("Menu", menu)

if choice == "Login":
    username = st.text_input("Username")
    password = st.text_input("Password", type="password")
    if st.button("Login"):
        if validate_user(username, password):
            st.session_state.authenticated = True
            st.session_state.username = username
            st.success("Login Successful")
            st.rerun()
        else:
            st.error("Invalid Credentials")

elif choice == "Register":
    username = st.text_input("Username")
    password = st.text_input("Password", type="password")
    if st.button("Register"):
        register_user(username, password)
        st.success("Registration Successful. Please log in.")
        st.rerun()

elif choice == "Predict Score" and st.session_state.authenticated:
    input_data = user_input_features()

    # Prediction Section
    if st.button("Predict Score"):
        prediction = model.predict(input_data)[0]
        st.success(f"Predicted Score: {round(prediction, 2)}")

elif choice == "Chatbot" and st.session_state.authenticated:
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

if __name__ == "__main__":
    init_db()
