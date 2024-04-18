import redis
import os
from dotenv import load_dotenv
from insightface.app import FaceAnalysis
from app import RealTimePredClass
import streamlit as st

st.subheader('Image to Reports')

def main():
    uploaded_images = st.file_uploader("Upload multiple images", accept_multiple_files=True, type=['jpg', 'jpeg', 'png'])

    if uploaded_images:
        images = [cv2.imdecode(np.fromstring(img.read(), np.uint8), 1) for img in uploaded_images]

        if st.button('Recognize Faces'):
            pred_img = RealTimePred.face_prediction(img, redis_face_db, 'facial_features')
            attendance_data = recognize_faces(images)
            df = pd.DataFrame(attendance_data)
            st.table(df)

if __name__ == "__main__":
    main()
