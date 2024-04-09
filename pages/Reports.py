import streamlit as st 
from Home import app as face_rec

st.subheader('Reports')

def mark_attendance():
    with st.spinner('Calculating attendance...'):
        face_rec.mark_attendance()
if st.button('Generate Reports'):
    mark_attendance()
    st.success('Attendance calculated successfully!')

    with st.spinner('Retrieving Data from Redis DB ...'):    
        vattend_status = face_rec.retrieve_status()
        st.dataframe(vattend_status)



