import cv2
import numpy as np


import face_recognition
from keras.models import model_from_json

class_map = {
    0 : 'surprise',
    1 : 'fear',
    2 : 'angry',
    3 : 'neutral',
    4 : 'sad',
    5 : 'disgust',
    6 : 'happy'
}

# load json and create model
json_file = open('model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)

# load weights into new model
loaded_model.load_weights("model.h5")

def locate_face(image_path, vis=True):
    image = face_recognition.load_image_file(image_path)
    face_locations = face_recognition.face_locations(image)

    return image, face_locations

def crop_image(image, box):
    crop_img = []
    for i, c in enumerate(box):
        y = c[0]
        x = c[3]
        bottom = c[2]
        right = c[1]

        _y = c[0]-65
        _x = c[3]-50
        _bottom = c[2] + 50
        _right = c[1] + 50

        crop_img = image[y: bottom, x: right]
    
    return crop_img


def predict_emotion(image_path, trained_model, return_image=False):
    original_image, coordinate = locate_face(image_path)
    cropped_image = crop_image(original_image, coordinate)
    pp_cropped_image = preprocess_image(cropped_image)
    pp_cropped_image = pp_cropped_image.reshape((1, pp_cropped_image.shape[0], pp_cropped_image.shape[1], 1)) # reshape so it fit with pretrain model
    prediction = trained_model.predict(pp_cropped_image)
    
    if not return_image:
        return prediction
    
    return prediction, original_image, pp_cropped_image


image_size = (48, 48) # (width, height)
emotion = 'test'

def preprocess_image(image):
    if len(image.shape) > 2:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image = cv2.resize(image, image_size, interpolation = cv2.INTER_AREA) # resize (48, 48)
    image = np.array(image) # convert pixel to float
    image = image.astype('float32')
    image /= 255 # normalize
    
    return image

def plot_image_and_emotion(image, label, prediction):
    fig, axs = plt.subplots(1, 2, figsize=(20, 6), sharey=False)
    
    bar_label = class_map.keys()
    
    image = np.reshape(image, (image.shape[0], image.shape[1]))
    
    axs[0].imshow(image, "gray")
    axs[0].set_title(label)
    
    axs[1].bar(bar_label, prediction)
    axs[1].grid()
    

def predictionFun(testImagePath):
    prediction, original_image, using_image = predict_emotion(testImagePath.format(emotion), loaded_model, return_image=True)
    predicted_arr = prediction[0]
    result = (np.amax(predicted_arr))
    result = np.where(predicted_arr == result)
    return (class_map[result[0][0]])
