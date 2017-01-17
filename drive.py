import argparse
import base64
import json

import numpy as np
import socketio
import eventlet
import eventlet.wsgi
import time
from PIL import Image
from PIL import ImageOps
from flask import Flask, render_template
from io import BytesIO

from keras.models import model_from_json
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array

# Fix error with Keras and TensorFlow
import tensorflow as tf
tf.python.control_flow_ops = tf

from model import preprocessImg

sio = socketio.Server()
app = Flask(__name__)
model = None
prev_image_array = None
prev_steer_length = 1
prev_steer_array = [0]*prev_steer_length
prev_steer_i = 0
prev_throttle = 0.



@sio.on('telemetry')
def telemetry(sid, data):
    #CHANGE
    global prev_steer_i
    global prev_steer_array
    global prev_throttle
    throttleIncStep = 0.1

    # The current steering angle of the car
    steering_angle = data["steering_angle"]
    # The current throttle of the car
    throttle = data["throttle"]
    # The current speed of the car
    speed = data["speed"]
    # The current image from the center camera of the car
    imgString = data["image"]
    image = Image.open(BytesIO(base64.b64decode(imgString)))
    image = preprocessImg(image)
    image_array = np.asarray(image)
    transformed_image_array = image_array[None, :, :, :]
    # This model currently assumes that the features of the model are just the images. Feel free to change this.
    steering_angle = float(model.predict(transformed_image_array, batch_size=1))
    # The driving model currently just outputs a constant throttle. Feel free to edit this.
    #throttle = 0.3

    ## CHANGE
    # smoot out steering angle over time
    steering_angle_smooth = (steering_angle + np.sum(prev_steer_array)) / (1+prev_steer_length)
    prev_steer_array[prev_steer_i] = steering_angle_smooth
    prev_steer_i += 1
    if prev_steer_i >= prev_steer_length:
        prev_steer_i = 0

    # relate steering angle to a max speed
    targetSpeed = max((((abs(steering_angle_smooth*25))/-30.)+1.) * 30, 15.)
    if float(speed) > targetSpeed:
        throttle = max(prev_throttle - throttleIncStep, -0.4)
    else:
        throttle = min(prev_throttle + throttleIncStep, 0.4)
    # failsave so car doesn't stall out:
    if float(speed) < 10.:
        throttle = 0.8
    prev_throttle = throttle
    ## END CHANGE

    print(steering_angle, throttle)
    send_control(steering_angle_smooth, throttle)


@sio.on('connect')
def connect(sid, environ):
    print("connect ", sid)
    send_control(0, 0)


def send_control(steering_angle, throttle):
    sio.emit("steer", data={
    'steering_angle': steering_angle.__str__(),
    'throttle': throttle.__str__()
    }, skip_sid=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Remote Driving')
    parser.add_argument('model', type=str,
    help='Path to model definition json. Model weights should be on the same path.')
    args = parser.parse_args()
    with open(args.model, 'r') as jfile:
        # NOTE: if you saved the file by calling json.dump(model.to_json(), ...)
        # then you will have to call:
        #
        #   model = model_from_json(json.loads(jfile.read()))\
        #
        # instead.
        model = model_from_json(jfile.read())


    model.compile("adam", "mse")
    weights_file = args.model.replace('json', 'h5')
    model.load_weights(weights_file)

    # wrap Flask application with engineio's middleware
    app = socketio.Middleware(sio, app)

    # deploy as an eventlet WSGI server
    eventlet.wsgi.server(eventlet.listen(('', 4567)), app)