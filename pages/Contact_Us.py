import streamlit as st
from send_email import send_email

st.header("Contact Us")

st.text("Your feedback is precious to us so please provide us with your insights as to how we can improve our website.")

with st.form(key="email_form"):
    user_email = st.text_input("Enter your email address: ")
    raw_message = st.text_area("Your Feedback")
    message = f"""\
Subject: New email from {user_email}

From: {user_email}
{raw_message}
"""
    button = st.form_submit_button("Submit")
    if button:
        send_email(message)
        st.info("Your email was sent successfully")