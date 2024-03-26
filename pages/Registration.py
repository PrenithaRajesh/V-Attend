# register.py
import streamlit as st
import cv2
import numpy as np
import redis
import os
from dotenv import load_dotenv
from insightface.app import FaceAnalysis
from streamlit_webrtc import webrtc_streamer
import av

load_dotenv()

hostname = os.getenv('REDIS_HOST')
portnumber = os.getenv('REDIS_PORT')
password = os.getenv('REDIS_PASSWORD')

r = redis.StrictRedis(host=hostname, port=portnumber, password=password)

faceapp = FaceAnalysis(name='buffalo_l', root='./models', providers=['CPUExecutionProvider'])
faceapp.prepare(ctx_id=0, det_size=(640, 640))

class RegistrationForm:
    def __init__(self):
        self.sample = 0

    def reset(self):
        self.sample = 0

    def get_embedding(self, frame):
        results = faceapp.get(frame, max_num=1)
        embeddings = None
        for res in results:
            self.sample += 1
            x1, y1, x2, y2 = res['bbox'].astype(int)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 1)
            text = f"samples = {self.sample}"
            cv2.putText(frame, text, (x1, y1), cv2.FONT_HERSHEY_DUPLEX, 0.6, (255, 255, 0), 2)
            embeddings = res['embedding']
        return frame, embeddings

    def save_data_in_redis_db(self, name, regNo):
        if name is not None and name.strip() != '':
            key = regNo
        else:
            return 'name_false'

        if 'face_embedding.txt' not in os.listdir():
            return 'file_false'

        x_array = np.loadtxt('face_embedding.txt', dtype=np.float32)

        received_samples = int(x_array.size / 512)
        x_array = x_array.reshape(received_samples, 512)
        x_mean = x_array.mean(axis=0)
        x_mean = x_mean.astype(np.float32)
        x_mean_bytes = x_mean.tobytes()

        r.hset(name='vit:register', key=key, value=x_mean_bytes)
        os.remove('face_embedding.txt')
        self.reset()
        return True

def register(logged_in):
    if not logged_in:
        st.error("You need to log in to access this page.")
        return

    st.header('Student Registration Form')

    registration_form = RegistrationForm()

    # form
    sname = st.text_input(label='Name',placeholder='Full Name')
    regNo = st.text_input(label='RegNo', placeholder="Registration Number")

    def video_callback_func(frame):
        img = frame.to_ndarray(format='bgr24') # 3d array bgr
        reg_img, embedding = registration_form.get_embedding(img)
        if embedding is not None:
            with open('face_embedding.txt', mode='ab') as f:
                np.savetxt(f, embedding)
        return av.VideoFrame.from_ndarray(reg_img, format='bgr24')

    webrtc_streamer(key='registration', video_frame_callback=video_callback_func)

    if st.button('Submit'):
        return_val = registration_form.save_data_in_redis_db(sname, regNo)
        if return_val == True:
            st.success(f"{regNo} registered successfully")
        elif return_val == 'name_false':
            st.error('Please enter the name: Name cannot be empty or spaces')
        elif return_val == 'file_false':
            st.error('face_embedding.txt is not found. Please refresh the page and execute again.')

if __name__ == "__main__":
    logged_in = st.session_state.get("logged_in", False)
    register(logged_in)
