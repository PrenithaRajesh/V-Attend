import streamlit as st
import redis
import os
from dotenv import load_dotenv

st.subheader('Add RegNo of students on leave')

load_dotenv()

hostname = os.getenv('REDIS_HOST')
portnumber = os.getenv('REDIS_PORT')
password = os.getenv('REDIS_PASSWORD')

r = redis.StrictRedis(host=hostname, port=portnumber, password=password)

def add_to_leave_list(reg_numbers):
    r.lpush('vattend:onLeave', *reg_numbers)

def main():
    leave_reg_numbers = st.text_input('Enter registration numbers separated by commas (e.g., ABC123,XYZ456):')
    leave_reg_numbers = [reg.strip() for reg in leave_reg_numbers.split(',') if reg.strip()]  
    if st.button('Add to Leave List'):
        if leave_reg_numbers:
            add_to_leave_list(leave_reg_numbers)
            st.success('Students added to leave list successfully.')
        else:
            st.warning('Please enter at least one registration number.')

if __name__ == '__main__':
    main()
