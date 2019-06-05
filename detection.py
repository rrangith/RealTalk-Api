import cv2
import tensorflow as tf
from keras.models import load_model
import numpy as np
from threading import Thread

from utils import detector_utils as detector_utils
from utils.datasets import get_labels
from utils.inference import detect_faces
from utils.inference import draw_bounding_box
from utils.inference import apply_offsets
from utils.inference import load_detection_model
from utils.preprocessor import preprocess_input

"""
Global variable for threads
"""
thread_storage = {} # need this instead of appending to array because you don't know which thread will finish first

"""
Hand
"""
# Constants
num_hands_detect = 2 # max number of hands we want to detect/track, can scale this up
min_threshold = 0.2

# Model
detection_graph, sess = detector_utils.load_inference_graph()

"""
Face
"""
# Files
detection_model_path = 'haarcascade_frontalface_default.xml'
emotion_model_path = 'fer2013_mini_XCEPTION.102-0.66.hdf5'
emotion_labels = get_labels('fer2013')

# Loading models
face_detection_model = load_detection_model(detection_model_path)
emotion_classifier = load_model(emotion_model_path, compile=False)
graph = tf.get_default_graph() # To avoid tensor flow graph not found error

# Emotion constants
emotion_target_size = emotion_classifier.input_shape[1:3]
emotion_offsets = (20, 40)

"""
Gets a PIL image and returns the structure that contains hand coordinates, emotion, and face coordinates
This will also output an image with boxes around the face and hands if they exist for testing purposes

image: the PIL image that should be analyzed

The hands will be an array of 2, containing data for both hands
The face will contain the emotion and coordinates
"""
def detectImage(image):
    im_width, im_height = image.size # Getting image height and width
    image_np = np.array(image) # Converting PIL image to numpy array

    hand_thread = Thread(target=getHandCoords, args=[image_np, im_width, im_height]) # Get coordinates of hands in an array

    face_thread = Thread(target=getFaceDetails, args=[image_np]) # Get emotion and coordinates of face

    hand_thread.start()
    face_thread.start()

    hand_thread.join()
    face_thread.join()

    hand_coords = thread_storage['hands']
    face_details = thread_storage['face']
    
    # JSON structure to return
    image_details = {
                        "hands": [
                            {
                                "x": hand_coords[0][0],
                                "y": hand_coords[0][1],
                                "width": hand_coords[0][2],
                                "height": hand_coords[0][3]
                            },
                            {
                                "x": hand_coords[1][0],
                                "y": hand_coords[1][1],
                                "width": hand_coords[1][2],
                                "height": hand_coords[1][3]
                            }
                        ],
                        "face": {
                                "emotion": face_details[0],
                                "x": face_details[1],
                                "y": face_details[2],
                                "width": face_details[3],
                                "height": face_details[4]
                        }
                    }

    cv2.imwrite('box.png', cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)) # Writing to an image for testing purposes to prove boxes draw properly
    
    return image_details

"""
Gets an image as a numpy array and returns coordinates of hands

image_np: image as numpy array
im_width: width of the image
im_height: height of the image

Returns an array of 2, each element in the array will be of the form [bottom x coordinate, bottom y coordinates, width, height]
If a hand or two are missing, then an array of None will be returned
"""
def getHandCoords(image_np, im_width, im_height):
    # Actual detection. Variable boxes contains the bounding box cordinates for hands detected,
    # while scores contains the confidence for each of these boxes.
    boxes, scores = detector_utils.detect_objects(image_np, detection_graph, sess)

    hand_coords = detector_utils.get_hand_coords(num_hands_detect, min_threshold, scores, boxes, im_width, im_height, image_np) # 0.2 is the min threshold

    # This is in case hands are not found
    if len(hand_coords) < 1:
        hand_coords[0] = [None, None, None, None]
    if len(hand_coords) < 2:
        hand_coords[1] = [None, None, None, None]

    thread_storage['hands'] = hand_coords # instead of returning, put it into dictionary

 
"""
Gets an image as a numpy array and returns emotion and coordinates of face

image_np: image as numpy array

Returns an array of the form [emotion, bottom x coordinate, bottom y coordinate, width, height]
"""
def getFaceDetails(image_np):
    gray_image = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY) # the model requires a grayscale image

    faces = detect_faces(face_detection_model, gray_image) # pass in the grayscale image

    if len(faces) > 0: # make sure a face was found
        draw_bounding_box(faces[0], image_np) # draws a square around the face, just for testing purposes
        
        # doing preprocessing on grayscale image, found from the model creator's example
        x1, x2, y1, y2 = apply_offsets(faces[0], emotion_offsets) 
        gray_face = gray_image[y1:y2, x1:x2]
        gray_face = cv2.resize(gray_face, (emotion_target_size))
        gray_face = preprocess_input(gray_face, True)
        gray_face = np.expand_dims(gray_face, 0)
        gray_face = np.expand_dims(gray_face, -1)
        
        with graph.as_default(): # to prevent graph not found error
            emotion_prediction = emotion_classifier.predict(gray_face) # predict the emotion

        #emotion_probability = np.max(emotion_prediction) # uncomment this line to see the accuracy of the emotion
        
        emotion_label_arg = np.argmax(emotion_prediction)
    
        face_details = [emotion_labels[emotion_label_arg], int(faces[0][0]), int(faces[0][1]), int(faces[0][2]), int(faces[0][3])] # structure to return emotion and face coords
    else:
        face_details = [None, None, None, None, None] # in case no face was found

    thread_storage['face'] = face_details # instead of returning, put it into dictionary
