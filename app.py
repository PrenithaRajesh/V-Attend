import cv2
import numpy as np
import pandas as pd
import redis
import os
from streamlit_webrtc import webrtc_streamer
from twilio.rest import Client
import av
from dotenv import load_dotenv
import streamlit as st

from sklearn.metrics import pairwise
from datetime import datetime, timedelta

load_dotenv()

hostname = os.getenv('REDIS_HOST')
portnumber = os.getenv('REDIS_PORT')
password = os.getenv('REDIS_PASSWORD')

account_sid = os.getenv('TWILIO_ACCOUNT_SID')
auth_token = os.getenv('TWILIO_AUTH_TOKEN')
client = Client(account_sid, auth_token)

token = client.tokens.create()

r = redis.StrictRedis(host=hostname, port=portnumber, password=password)

from insightface.app import FaceAnalysis
faceapp = FaceAnalysis(name='buffalo_l',root='./models',providers=['CPUExecutionProvider'])
faceapp.prepare(ctx_id=0, det_size=(640, 640))

# Retrive Data from database
def retrive_data(name):
    retrive_dict= r.hgetall(name)
    retrive_series = pd.Series(retrive_dict)
    retrive_series = retrive_series.apply(lambda x: np.frombuffer(x,dtype=np.float32))
    index = retrive_series.index
    index = list(map(lambda x: x.decode(), index))
    retrive_series.index = index
    retrive_df =  retrive_series.to_frame().reset_index()
    retrive_df.columns = ['name_regNo','facial_features']
    retrive_df[['RegNo','Name']] = retrive_df['name_regNo'].apply(lambda x: x.split('@')).apply(pd.Series)
    return retrive_df[['RegNo','Name','facial_features']]

def convert_utc_to_ist(utc_datetime):
    ist_offset = timedelta(hours=5, minutes=30)
    ist_datetime = utc_datetime + ist_offset
    return ist_datetime

embeddings_list = []

# Registration Form
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
            embeddings_list.append(embeddings)  # Add to global list
        return frame, embeddings

    def save_data_in_redis_db(self, name, regNo, phNumber):
        if name is not None:
            if name.strip() != '':
                key = f'{regNo}@{name}'
            else:
                return 'name_false'
        else:
            return 'name_false'

        if embeddings_list:
            x_mean = np.mean(embeddings_list, axis=0).astype(np.float32)
            x_mean_bytes = x_mean.tobytes()
            r.hset(name='vattend:register', key=key, value=x_mean_bytes)
            r.hset(name='vattend:phNumber', key=key, value=phNumber)  # Save phone number to new Redis hash table
            self.reset()
            embeddings_list.clear()  # Clear the global list
            return True
        else:
            return 'file_false'

def register():
    registration_form = RegistrationForm()

    sname = st.text_input(label='Name', placeholder='Full Name')
    regNo = st.text_input(label='RegNo', placeholder="Registration Number")
    phNumber = st.text_input(label='Phone Number', placeholder="Phone Number")
    input_type = st.radio("Select input type:", ("Image", "Video"))

    if input_type == "Image":
        file = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png"])
        if file is not None:
            img = cv2.imdecode(np.fromstring(file.read(), np.uint8), 1)
            reg_img, embedding = registration_form.get_embedding(img)
            st.image(reg_img, channels="BGR")

    elif input_type == "Video":
        def video_callback_func(frame):
            img = frame.to_ndarray(format='bgr24')
            reg_img, embedding = registration_form.get_embedding(img)
            return av.VideoFrame.from_ndarray(reg_img, format='bgr24')

        webrtc_streamer(key='registration', rtc_configuration={"iceServers": token.ice_servers}, video_frame_callback=video_callback_func)

    if st.button('Submit'):
        return_val = registration_form.save_data_in_redis_db(sname, regNo, phNumber)
        if return_val == True:
            st.success(f"{regNo} registered successfully")
        elif return_val == 'name_false':
            st.error('Please enter the name: Name cannot be empty or spaces')
        elif return_val == 'file_false':
            st.error('No face detected. Please try again.')

# Prediction
def ml_search_algorithm(dataframe,feature_column,test_vector,
                        name_regNo=['RegNo','Name'],thresh=0.5):
    dataframe = dataframe.copy()
    X_list = dataframe[feature_column].tolist()
    x = np.asarray(X_list)
    
    similar = pairwise.cosine_similarity(x,test_vector.reshape(1,-1))
    similar_arr = np.array(similar).flatten()
    dataframe['cosine'] = similar_arr

    data_filter = dataframe.query(f'cosine >= {thresh}')
    if len(data_filter) > 0:
        data_filter.reset_index(drop=True,inplace=True)
        argmax = data_filter['cosine'].argmax()
        person_regNo, person_name = data_filter.loc[argmax][name_regNo]
        
    else:
        person_name = 'Unknown'
        person_regNo = 'Unknown'
        
    return person_name, person_regNo

# saving logs every minute
class RealTimePred:
    def __init__(self, camera):
        self.camera = camera
        self.logs = dict(regNo=[], name=[], current_time=[])

    def reset_dict(self):
        self.logs = dict(regNo=[], name=[], current_time=[])

    def saveLogs_redis(self):
        dataframe = pd.DataFrame(self.logs)
        dataframe.drop_duplicates('name', inplace=True)
        name_list = dataframe['name'].tolist()
        regNo_list = dataframe['regNo'].tolist()
        ctime_list = dataframe['current_time'].tolist()
        encoded_data = []
        for name, regNo, ctime in zip(name_list, regNo_list, ctime_list):
            if name != 'Unknown':
                concat_string = f"{regNo}@{name}@{ctime}"
                encoded_data.append(concat_string)

        if len(encoded_data) > 0:
            r.rpush(f'vattend:{self.camera}', *encoded_data)

        self.reset_dict()
                
    def face_prediction(self, test_image, dataframe, feature_column, name_regNo=['RegNo', 'Name'], thresh=0.5):
        current_time = str(datetime.now())

        results = faceapp.get(test_image)
        test_copy = test_image.copy()

        for res in results:
            x1, y1, x2, y2 = res['bbox'].astype(int)
            embeddings = res['embedding']
            person_name, person_regNo = ml_search_algorithm(dataframe, feature_column, test_vector=embeddings,
                                                            name_regNo=name_regNo, thresh=thresh)

            if person_name == 'Unknown':
                color = (0, 0, 255)  # bgr
            else:
                color = (0, 255, 0)

            cv2.rectangle(test_copy, (x1, y1), (x2, y2), color)

            # Display name and regno
            text_name = f"Name: {person_name}"
            text_regno = f"Reg No: {person_regNo}"
            cv2.putText(test_copy, text_name, (x1, y1 - 20), cv2.FONT_HERSHEY_DUPLEX, 0.7, color, 2)
            cv2.putText(test_copy, text_regno, (x1, y1 - 50), cv2.FONT_HERSHEY_DUPLEX, 0.7, color, 2)

            cv2.putText(test_copy, current_time, (x1, y2 + 10), cv2.FONT_HERSHEY_DUPLEX, 0.7, color, 2)

            self.logs['regNo'].append(person_regNo)
            self.logs['name'].append(person_name)
            self.logs['current_time'].append(current_time)

        return test_copy

def mark_attendance():
    incoming_logs = r.lrange('vattend:incoming', 0, -1)
    outgoing_logs = r.lrange('vattend:outgoing', 0, -1)

    registered_keys = r.hkeys('vattend:register')
    registered_persons = [key.decode('utf-8') for key in registered_keys]

    last_times = {}

    for log in incoming_logs:
        log_str = log.decode('utf-8')
        regNo, name, log_time_str = log_str.split('@')
        log_time = datetime.strptime(log_time_str, '%Y-%m-%d %H:%M:%S.%f')
        last_times[(regNo, name)] = {'in_time': log_time}

    for log in outgoing_logs:
        log_str = log.decode('utf-8')
        regNo, name, log_time_str = log_str.split('@')
        log_time = datetime.strptime(log_time_str, '%Y-%m-%d %H:%M:%S.%f')
        if (regNo, name) in last_times:
            last_times[(regNo, name)]['out_time'] = log_time
        else:
            last_times[(regNo, name)] = {'out_time': log_time}  

    for person_key, times in last_times.items():
        regNo, name = person_key
        last_in_time = times.get('in_time')
        last_out_time = times.get('out_time')
        
        if last_out_time and last_in_time and last_out_time < last_in_time:
            r.hset('vattend:status', f'{regNo}@{name}', 'present')
        if last_in_time and not last_out_time:
            r.hset('vattend:status', f'{regNo}@{name}', 'present')
        else:
            r.hset('vattend:status', f'{regNo}@{name}', 'absent')

    for person_key in registered_persons:
        regNo, name = person_key.split('@')
        if (regNo, name) not in last_times:
            r.hset('vattend:status', f'{regNo}@{name}', 'present')


def retrieve_status(status_filter=None):
    registered_keys = r.hkeys('vattend:register')
    status_list = []
    
    last_times = {}
    
    incoming_logs = r.lrange('vattend:incoming', 0, -1)
    for log in incoming_logs:
        log_str = log.decode('utf-8')
        regNo, name, log_time_str = log_str.split('@')
        
        log_time_str = log_time_str.split('.')[0]
        log_time = datetime.strptime(log_time_str, '%Y-%m-%d %H:%M:%S')
        last_times[(regNo, name)] = {'in_time': log_time}

    outgoing_logs = r.lrange('vattend:outgoing', 0, -1)
    for log in outgoing_logs:
        log_str = log.decode('utf-8')
        regNo, name, log_time_str = log_str.split('@')
        log_time_str = log_time_str.split('.')[0]
        log_time = datetime.strptime(log_time_str, '%Y-%m-%d %H:%M:%S')
        if (regNo, name) in last_times:
            last_times[(regNo, name)]['out_time'] = log_time
        else:
            last_times[(regNo, name)] = {'out_time': log_time}

    mark_on_leave()
    
    for key in registered_keys:
        regNo_name = key.decode('utf-8')
        regNo, name = regNo_name.split('@')
        
        times = last_times.get((regNo, name), {})
        last_in_time = str(times.get('in_time', '-'))  # Convert to string
        last_out_time = str(times.get('out_time', '-'))  # Convert to string

        status_dict = r.hgetall('vattend:status')
        status = status_dict.get(regNo_name.encode(), b'').decode()

        if status_filter:
            if status!=status_filter:
                continue
        
        status_list.append({'RegNo': regNo, 'Name':name, 'Last In-Time': last_in_time, 'Last Out-Time': last_out_time, 'Status': status})

    return status_list

def mark_on_leave():
    leave_reg_numbers = r.lrange('vattend:onLeave', 0, -1)
    leave_reg_numbers = [reg.decode('utf-8') for reg in leave_reg_numbers]

    for reg_number in leave_reg_numbers:
        status_key = next((key.decode('utf-8') for key in r.hkeys('vattend:status') if key.decode('utf-8').startswith(reg_number)), None)
        if status_key:
            r.hset('vattend:status', status_key, 'onLeave')