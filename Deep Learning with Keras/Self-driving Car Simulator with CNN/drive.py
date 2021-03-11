#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import  argparse
import  base64
from datetime import datetime
import os
import shutil
import numpy as np
import  socketio
import eventlet
import eventlet.wsgi
from PIL import Image
from  flask  import  Flask
from io import BytesIO

from tensorflow.keras.models import load_model

import  utils

#initialize our server
sio = socketio.Server()
#our flask (web) app
app = Flask(__name__)
#init our model and image array as empty
model = None
prev_image_array = None

# Minimum and maximum speed of the vehicle
MAX_SPEED = 25
MIN_SPEED = 10

# Speed ​​the original moment
speed_limit = MAX_SPEED

#registering event handler for the server
@sio.on('telemetry')
def telemetry(sid, data):
    if  data:
        # Get the current throttle value
        throttle = float(data["throttle"])
        # Current steering angle of car
        steering_angle = float(data["steering_angle"])
    	  # The current speed of the car
        speed = float(data["speed"])
        # Photo from middle camera
        image = Image.open(BytesIO(base64.b64decode(data["image"])))
        try:
			# Preprocessing photos, cropping, reshape
            image = np.asarray(image)       
            image = utils.preprocess(image)
            image = np.array([image])
            print('*****************************************************')
            steering_angle = float(model.predict(image, batch_size=1))
            
			# The speed we set is between 10 and 25
            global speed_limit
            if speed > speed_limit:
                speed_limit  =  MIN_SPEED   # slow down
            else:
                speed_limit = MAX_SPEED
            throttle = 1.0 - steering_angle**2 - (speed/speed_limit)**2

            print('{} {} {}'.format(steering_angle, throttle, speed))
			
			# Sending data about steering angle and speed to the software for self-driving car
            send_control(steering_angle, throttle)
        except Exception as e:
            print ( e )

        # save frame
        if args.image_folder != '':
            timestamp = datetime.utcnow().strftime('%Y_%m_%d_%H_%M_%S_%f')[:-3]
            image_filename = os.path.join(args.image_folder, timestamp)
            image.save('{}.jpg'.format(image_filename))
    else:
        
        sio.emit('manual', data={}, skip_sid=True)


@sio.on('connect')
def  connect ( sid , approximately ):
    print("connect ", sid)
    send_control(0, 0)


def send_control(steering_angle, throttle):
    sio.emit(
        "steer",
        data={
            'steering_angle': steering_angle.__str__(),
            'throttle': throttle.__str__()
        },
        skip_sid=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Remote Driving')
    parser.add_argument(
        'model',
        type=str,
        help='Path to model h5 file. Model should be on the same path.'
    )
    parser.add_argument(
        'image_folder',
        type=str,
        nargs = '?' ,
        default='',
        help='Path to image folder. This is where the images from the run will be saved.'
    )
    args = parser.parse_args()

    # Load the model that we have trained in the previous step
    model = load_model(args.model)

    if args.image_folder != '':
        print("Creating image folder at {}".format(args.image_folder))
        if not os.path.exists(args.image_folder):
            os.makedirs(args.image_folder)
        else:
            shutil . rmtree ( args . image_folder )
            os.makedirs(args.image_folder)
        print("RECORDING THIS RUN ...")
    else:
        print("NOT RECORDING THIS RUN ...")

    # wrap Flask application with engineio's middleware
    app = socketio.Middleware(sio, app)

    # deploy as an eventlet WSGI server
    eventlet.wsgi.server(eventlet.listen(('', 4567)), app)

