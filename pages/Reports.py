import streamlit as st 
from Home import app as face_rec

st.subheader('Reports')

def mark_and_retrieve_status(status_filter):
    with st.spinner('Calculating attendance...'):
        face_rec.mark_attendance()
        st.success('Attendance calculated successfully!')

    with st.spinner('Retrieving Data from Redis DB ...'):    
        vattend_status = face_rec.retrieve_status(status_filter=status_filter)
        st.dataframe(vattend_status)

filter_option = st.selectbox('Filter Option', ['All', 'Present', 'Absent', 'On Leave'])

if st.button('Generate Reports'):
    if filter_option == 'All':
        mark_and_retrieve_status(status_filter=None)
    elif filter_option == 'Present':
        mark_and_retrieve_status(status_filter='present')
    elif filter_option == 'Absent':
        mark_and_retrieve_status(status_filter='absent')
    elif filter_option == 'On Leave':
        mark_and_retrieve_status(status_filter='onLeave')


