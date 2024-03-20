import streamlit as st
import app as face_rec

import numpy as np
from streamlit_webrtc import webrtc_streamer
import av

st.set_page_config(page_title='v-attend | registration')
st.header('Student Registration Form')

registration_form = face_rec.RegistrationForm()

# form
sname = st.text_input(label='Name',placeholder='Full Name')
regNo = st.text_input(label='RegNo', placeholder="Registration Number")


# Collect facial embedding of that person
def video_callback_func(frame):
    img = frame.to_ndarray(format='bgr24') # 3d array bgr
    reg_img, embedding = registration_form.get_embedding(img)
    if embedding is not None:
        with open('face_embedding.txt',mode='ab') as f:
            np.savetxt(f,embedding)
    
    return av.VideoFrame.from_ndarray(reg_img,format='bgr24')

webrtc_streamer(key='registration',video_frame_callback=video_callback_func)


# save the data in redis database
if st.button('Submit'):
    return_val = registration_form.save_data_in_redis_db(sname,regNo)
    if return_val == True:
        st.success(f"{regNo} registered sucessfully")
    elif return_val == 'name_false':
        st.error('Please enter the name: Name cannot be empty or spaces')
        
    elif return_val == 'file_false':
        st.error('face_embedding.txt is not found. Please refresh the page and execute again.')
        
