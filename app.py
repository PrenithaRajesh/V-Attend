import cv2
import numpy as np
import pandas as pd
import redis
import os
from streamlit_webrtc import webrtc_streamer
import av
from dotenv import load_dotenv
import streamlit as st

from sklearn.metrics import pairwise
from datetime import datetime, timedelta

load_dotenv()

hostname = os.getenv('REDIS_HOST')
portnumber = os.getenv('REDIS_PORT')
password = os.getenv('REDIS_PASSWORD')

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
        return frame, embeddings

    def save_data_in_redis_db(self, name, regNo):
        if name is not None:
            if name.strip() != '':
                key = f'{regNo}@{name}'
            else:
                return 'name_false'
        else:
            return 'name_false'

        x_array = np.loadtxt('face_embedding.txt', dtype=np.float32)

        received_samples = int(x_array.size / 512)
        x_array = x_array.reshape(received_samples, 512)
        x_mean = x_array.mean(axis=0)
        x_mean = x_mean.astype(np.float32)
        x_mean_bytes = x_mean.tobytes()

        r.hset(name='vattend:register', key=key, value=x_mean_bytes)
        os.remove('face_embedding.txt')
        self.reset()
        return True

def register():
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
            r.lpush(f'vattend:{self.camera}', *encoded_data)

        self.reset_dict()

    def mark_attendance(self):
        # Get the logs from the database
        logs = r.lrange('vattend:logs', 0, -1)
        
        # Get the current time
        current_time = datetime.now()
        # Set the cutoff time
        cutoff_time = datetime(current_time.year, current_time.month, current_time.day, 22, 0, 0)

        # Get the list of registered persons
        registered_keys = r.hkeys('vattend:register')
        registered_persons = [key.decode('utf-8') for key in registered_keys]

        # Iterate through logs and mark attendance
        for log in logs:
            log_str = log.decode('utf-8')
            regNo, name, log_time_str = log_str.split('@')
            log_time = datetime.strptime(log_time_str, '%Y-%m-%d %H:%M:%S.%f')
            if log_time <= cutoff_time:
                r.hset('vattend:status', f'{regNo}@{name}', 'present')
            else:
                r.hset('vattend:status', f'{regNo}@{name}', 'absent')

        # Mark registered persons who are not in logs or last timestamp > 10pm as absent
        for person_key in registered_persons:
            regNo, name = person_key.split('@')
            last_log_time = None
            for log in logs:
                log_str = log.decode('utf-8')
                log_regNo, log_name, log_time_str = log_str.split('@')
                if log_regNo == regNo and log_name == name:
                    last_log_time = datetime.strptime(log_time_str, '%Y-%m-%d %H:%M:%S.%f')
            if last_log_time is None or last_log_time > cutoff_time:
                r.hset('vattend:status', f'{regNo}@{name}', 'absent')
                
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

        # Call method to mark attendance
        self.mark_attendance()

        return test_copy
