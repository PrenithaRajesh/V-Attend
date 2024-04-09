import streamlit as st 
from Home import app as face_rec
from streamlit_webrtc import webrtc_streamer
from twilio.rest import Client
import av
import os

account_sid = os.getenv('TWILIO_ACCOUNT_SID')
auth_token = os.getenv('TWILIO_AUTH_TOKEN')
client = Client(account_sid, auth_token)
token = client.tokens.create()

st.subheader('Exit Camera')

with st.spinner('Retrieving Data from Redis DB ...'):    
    redis_face_db = face_rec.retrive_data(name='vattend:register')

st.success("Data sucessfully retrived from Redis")

# Create an instance of RealTimePred class
real_time_pred = face_rec.RealTimePred(camera='outgoing')

def video_frame_callback(frame):
    global setTime
    
    img = frame.to_ndarray(format="bgr24") 
    pred_img = real_time_pred.face_prediction(img, redis_face_db, 'facial_features')
    
    # Call saveLogs_redis to update the logs
    real_time_pred.saveLogs_redis()
    
    return av.VideoFrame.from_ndarray(pred_img, format="bgr24")

webrtc_streamer(key="realtimePrediction", rtc_configuration={"iceServers": token.ice_servers}, video_frame_callback=video_frame_callback)
