# V-Attend: Automated Attendance Management System 

---- Solve-A-Thon'24 ----

Traditional attendance methods, such as manual sign-in sheets or ID card tapping, are prone to errors and time-consuming processes, leading to inefficiencies in workforce management. These methods often suffer from issues like buddy punching and loss of cards, impacting accurate attendance tracking and organizational productivity.

V-Attend is an automated attendance management system. It utilizes facial recognition technology to mark attendance and provides real-time reports on attendance status.

## Features

- **Facial Recognition:** Utilizes facial recognition technology to identify individuals and mark their attendance.
- **Real-time Reporting:** Provides real-time reports on attendance status, including present, absent, and on leave.
- **Filtering Options:** Allows users to filter attendance reports based on various criteria such as present, absent, and on leave.
- **User Registration:** Enables users to register themselves into the system by providing their name and registration number.
- **Mobile Notifications:** Students receive daily notifications on their mobile devices, informing them about their attendance status.

## Technologies Used

- **Python:** The backend of the application is developed using Python programming language.
- **Redis:** Utilized as the database to store attendance logs and user registration data.
- **Streamlit:** Used for building the web application user interface with interactive features.
- **Insightface:** 
- **OpenCV:** Integrated for facial recognition capabilities.
- **NumPy and Pandas:** Utilized for data manipulation and analysis.
- **dotenv:** Employed for managing environment variables.

## Installation

To install and run the v-attend application locally, follow these steps:

1. Clone the repository:

   ```bash
   git clone https://github.com/Precoder365/v-attend.git
   ```

2. Create a virtual environment and install the required libraries.

    ```bash
    python -m venv venv
    venv\Scripts\activate
    pip install -r requirements.txt
    ```
   
2. Add .env file

    ```
    REDIS_HOST =
    REDIS_PORT =
    REDIS_PASSWORD =
    
    TWILIO_ACCOUNT_SID =
    TWILIO_AUTH_TOKEN =
       
    VONAGE_API_KEY = 
    VONAGE_API_SECRET = 
    VONAGE_PHONE_NUMBER = 
    ```

3. Run the streamlit app

   ```bash
   streamlit run Home.py
   ```

## Deployed link:

<a href="https://vattend-pren.streamlit.app/">https://vattend-pren.streamlit.app/</a>
