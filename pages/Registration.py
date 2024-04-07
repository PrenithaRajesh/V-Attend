import redis
import os
from dotenv import load_dotenv
from insightface.app import FaceAnalysis
from app import register

load_dotenv()

hostname = os.getenv('REDIS_HOST')
portnumber = os.getenv('REDIS_PORT')
password = os.getenv('REDIS_PASSWORD')

r = redis.StrictRedis(host=hostname, port=portnumber, password=password)

faceapp = FaceAnalysis(name='buffalo_l', root='./models', providers=['CPUExecutionProvider'])
faceapp.prepare(ctx_id=0, det_size=(640, 640))

if __name__ == "__main__":
    register()
