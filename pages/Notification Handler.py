import streamlit as st
import redis
import vonage
import os

# Function to send notifications
def send_notifications(absentee_status, absentee_phones):
    # Your Vonage API key and secret
    api_key = os.getenv('VONAGE_API_KEY')
    api_secret = os.getenv('VONAGE_API_SECRET')
    vonage_phone_number = os.getenv('VONAGE_PHONE_NUMBER')

    # Initialize Vonage client
    client = vonage.Client(key=api_key, secret=api_secret)
    sms_client = vonage.Sms(client)

    for key, status in absentee_status.items():
        name_regNo = key.split('@')
        name = name_regNo[0]
        regNo = name_regNo[1]
        phNumber = absentee_phones.get(key, None)
        print(name)
        print(phNumber)
        if phNumber:
            # Add +91 to phone number
            phNumber_with_country_code = '91' + phNumber

            # Compose SMS message
            message_body = f"Dear {name}, you are marked absent today. Please contact your supervisor if there's any issue."

            # Send SMS message
            response = sms_client.send_message({
                'from': vonage_phone_number,
                'to': phNumber_with_country_code,
                'text': message_body
            })

            st.write(f"Notification sent to {name} ({phNumber_with_country_code})")

# Streamlit app
def main():
    st.title('Send Notifications to Absentees')

    # Connect to Redis
    hostname = os.getenv('REDIS_HOST')
    portnumber = os.getenv('REDIS_PORT')
    password = os.getenv('REDIS_PASSWORD')
    r = redis.StrictRedis(host=hostname, port=portnumber, password=password)

    # Retrieve absentee status and phone numbers
    absentee_status = r.hgetall('vattend:status')
    absentee_phones = r.hgetall('vattend:phNumber')

    absentee_status = {key.decode('utf-8'): value.decode('utf-8') for key, value in absentee_status.items()}
    absentee_phones = {key.decode('utf-8'): value.decode('utf-8') for key, value in absentee_phones.items()}

    st.write("Absentee Status:", absentee_status)  # Debugging
    st.write("Absentee Phones:", absentee_phones)  # Debugging

    # Button to trigger sending notifications
    if st.button('Send Notifications'):
        st.write("Sending notifications...")
        send_notifications(absentee_status, absentee_phones)
        st.success("Notifications sent successfully!")

if _name_ == "_main_":
    main()