from utils import detector_utils as detector_utils
import cv2
import tensorflow as tf
import datetime

from statistics import mode

from keras.models import load_model
import numpy as np

from utils.datasets import get_labels
from utils.inference import detect_faces
from utils.inference import draw_text
from utils.inference import draw_bounding_box
from utils.inference import apply_offsets
from utils.inference import load_detection_model
from utils.preprocessor import preprocess_input

detection_graph, sess = detector_utils.load_inference_graph()

# parameters for loading data and images
detection_model_path = 'haarcascade_frontalface_default.xml'
emotion_model_path = 'fer2013_mini_XCEPTION.102-0.66.hdf5'
emotion_labels = get_labels('fer2013')

# loading models
face_detection = load_detection_model(detection_model_path)
emotion_classifier = load_model(emotion_model_path, compile=False)

emotion_offsets = (20, 40)

 # getting input model shapes for inference
emotion_target_size = emotion_classifier.input_shape[1:3]


num_hands_detect = 2 # max number of hands we want to detect/track, can scale this up
min_threshold = 0.2

def detectImage(image):
    im_width, im_height = image.size

    image_np = np.array(image)


    image_np = cv2.flip(image_np, 1)
    gray_image = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)



    faces = detect_faces(face_detection, gray_image)

    draw_bounding_box(faces[0], image_np)


    # Actual detection. Variable boxes contains the bounding box cordinates for hands detected,
    # while scores contains the confidence for each of these boxes.
    # Hint: If len(boxes) > 1 , you may assume you have found atleast one hand (within your score threshold)
    boxes, scores = detector_utils.detect_objects(image_np, detection_graph, sess)

    hand_coords = detector_utils.get_coords(num_hands_detect, min_threshold, scores, boxes, im_width, im_height, image_np) #0.2 is the min threshold

    cv2.imwrite('box.png', cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR))


    print(str(hand_coords))

    face_details = []

    for face_coords in faces:
        x1, x2, y1, y2 = apply_offsets(face_coords, emotion_offsets)
        gray_face = gray_image[y1:y2, x1:x2]
        gray_face = cv2.resize(gray_face, (emotion_target_size))
        gray_face = preprocess_input(gray_face, True)
        gray_face = np.expand_dims(gray_face, 0)
        gray_face = np.expand_dims(gray_face, -1)
        emotion_prediction = emotion_classifier.predict(gray_face)
        emotion_probability = np.max(emotion_prediction)
        emotion_label_arg = np.argmax(emotion_prediction)
        face_details.append((emotion_labels[emotion_label_arg], face_coords))

    image_details = {
                        "hands": hand_coords,
                        "faces": face_details
                    }
    
    return image_details

    