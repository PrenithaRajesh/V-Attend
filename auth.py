# auth.py
import streamlit as st
from streamlit import session_state as state
import os
from dotenv import load_dotenv
import redis
from Registration import register

load_dotenv()

hostname = os.getenv('REDIS_HOST')
portnumber = os.getenv('REDIS_PORT')
password = os.getenv('REDIS_PASSWORD')
r = redis.Redis(host=hostname, port=portnumber, password=password)

def create_user(username, password):
    if r.exists(username):
        return "Username already exists. Please choose a different one."  # Username already exists
    else:
        r.set(username, password)
        return True

def authenticate_user(username, password):
    stored_password = r.get(username)
    if stored_password and stored_password.decode() == password:
        return True
    else:
        return False

def signup():
    st.subheader("Sign Up")
    new_username = st.text_input("New Username")
    new_password = st.text_input("New Password", type='password')
    confirm_password = st.text_input("Confirm Password", type='password')

    if new_password != confirm_password:
        st.error("Passwords do not match")
        return

    if st.button("Sign Up"):
        result = create_user(new_username, new_password)
        if result == True:
            st.success("User created successfully. Please log in.")
        else:
            st.error(result)

def main():
    st.set_page_config(page_title='v-attend | authentication')  # Moved here
    st.title("Authentication App")

    if "logged_in" not in state:
        state.logged_in = False

    if state.logged_in:
        register()
    else:
        st.subheader("Login")
        username = st.text_input("Username")
        password = st.text_input("Password", type='password')
        if st.button("Login"):
            if authenticate_user(username, password):
                st.success("Logged in as {}".format(username))
                state.logged_in = True
                register()
            else:
                st.error("Invalid username or password")

        st.write("---")

        if st.checkbox("Don't have an account? Sign Up"):
            signup()

if __name__ == '__main__':
    main()
