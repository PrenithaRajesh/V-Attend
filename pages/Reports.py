import streamlit as st 
from Home import app as face_rec
from streamlit_webrtc import webrtc_streamer
import av
import time

st.subheader('Reports')

with st.spinner('Retriving Data from Redis DB ...'):    
    redis_face_db = face_rec.retrive_data(name='vattend:register')
    st.dataframe(redis_face_db)
    
st.success("Data sucessfully retrived from Redis")
