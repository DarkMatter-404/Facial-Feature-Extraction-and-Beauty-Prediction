from flask import Flask, render_template, request
import os
from flask import Flask, render_template, request, jsonify
import base64
import io
import cv2
import numpy as np
import pandas as pd
import dlib
from PIL import Image
import torch
import pickle
from skimage.feature import greycomatrix, greycoprops
from sklearn.preprocessing import StandardScaler
import torchvision.transforms as transforms

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'

if not os.path.exists(app.config['UPLOAD_FOLDER']):
    os.makedirs(app.config['UPLOAD_FOLDER'])

@app.route('/')
def index():
    return render_template('pls.html')

@app.route('/upload', methods=['POST'])
def upload_image():
    if 'image' not in request.files:
        return "No image part in the form!"

    image = request.files['image']

    if image.filename == '':
        return "No selected image!"

    image_path = os.path.join(app.config['UPLOAD_FOLDER'], image.filename)
    image.save(image_path)
    prediction = process_image_and_predict(image_path)

    return "Prediction: " + str(prediction)

model = pickle.load(open("LR_Model.pkl",'rb'))
scaler = pickle.load(open('scaler.sav','rb'))
predictor_path = 'shape_predictor_68_face_landmarks.dat'
predictor = dlib.shape_predictor(predictor_path)

def detect_landmarks(image):

  detector = dlib.get_frontal_face_detector()
  predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')

  gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

  faces = detector(gray)

  for face in faces:
    landmarks = predictor(gray,face)

    points = np.zeros((68,2),dtype=int)
    for i in range(0,68):
      points[i] = (landmarks.part(i).x, landmarks.part(i).y)
      x,y = (landmarks.part(i).x, landmarks.part(i).y)
      cv2.circle(image, (x, y), 1, (0, 0, 255), -1)

    
    return points

def calculate_golden_ratios(landmarks):

    under_eyes = np.linalg.norm(landmarks[41]-landmarks[46])
    interocular = np.linalg.norm(landmarks[39]-landmarks[42])
    nose_width = np.linalg.norm(landmarks[31]-landmarks[35])
    mouth_width = np.linalg.norm(landmarks[48]-landmarks[54])
    upper_lip_jaw = np.linalg.norm(landmarks[51]-landmarks[8])
    lip_height = np.linalg.norm(landmarks[51]-landmarks[57])
    nose_mouth_height = np.linalg.norm(landmarks[33]-landmarks[51])
    eyebrows_nose = np.linalg.norm((np.linalg.norm(landmarks[21]-landmarks[22]))-landmarks[33])/2
    nose_jaw = np.linalg.norm(landmarks[33]-landmarks[9])
    upper_lip_height = np.linalg.norm(landmarks[51]-landmarks[56])/2



    # Calculate the golden ratios
    ratio1 = under_eyes / interocular
    ratio2 = under_eyes / nose_width
    ratio3 = mouth_width / interocular
    ratio4 = upper_lip_jaw/interocular
    ratio5 = upper_lip_jaw / nose_width
    ratio6 = interocular / lip_height
    ratio7 = nose_width / interocular
    ratio8 = nose_width / upper_lip_height
    ratio9 = interocular/nose_mouth_height
    ratio10 = eyebrows_nose / nose_jaw
    ratio11 = interocular / nose_width

    return ratio1, ratio2, ratio3, ratio4, ratio5, ratio6, ratio7, ratio8, ratio9, ratio10, ratio11

def calculate_symmetric_ratios(landmarks):
    upper_eyebrow_numerator = np.linalg.norm(landmarks[21]-((np.linalg.norm(landmarks[22]-landmarks[21]))/2))
    upper_eyebrow_denominator = np.linalg.norm(landmarks[22]-((np.linalg.norm(landmarks[22]-landmarks[21]))/2))

    # Calculate the symmetric ratios
    lower_eyebrow_length = (np.linalg.norm(landmarks[17]-landmarks[21]))/(np.linalg.norm(landmarks[22]-landmarks[26]))
    lower_lip_length = (np.linalg.norm(landmarks[48]-landmarks[57]))/(np.linalg.norm(landmarks[54]-landmarks[57]))
    upper_eyebrow = upper_eyebrow_numerator/upper_eyebrow_denominator
    upper_lip = (np.linalg.norm(landmarks[48]-landmarks[51]))/(np.linalg.norm(landmarks[51]-landmarks[55]))
    nose = (np.linalg.norm(landmarks[31]-landmarks[33]))/(np.linalg.norm(landmarks[33]-landmarks[35]))

    return lower_eyebrow_length, lower_lip_length, upper_eyebrow, upper_lip, nose

def compute_glcm_features(image):

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


    glcm = greycomatrix(gray, [1], [0])

    # Calculate the GLCM properties
    homogeneity = greycoprops(glcm, 'homogeneity')[0, 0]
    contrast = greycoprops(glcm, 'contrast')[0, 0]
    energy = greycoprops(glcm, 'energy')[0, 0]
    correlation = greycoprops(glcm, 'correlation')[0, 0]

    return homogeneity, contrast, energy, correlation

def compute_hu_moments(image):

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Threshold the image to create a binary image
    _, binary = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY)

    # Calculate Hu's moments
    moments = cv2.moments(binary)
    hu_moments = cv2.HuMoments(moments)
    hu_moments = -np.sign(hu_moments) * np.log10(np.abs(hu_moments))

    return hu_moments.flatten()

def process_image_and_predict(image_path):
    img = Image.open(image_path)
    image_array = np.array(img)
    
    feature_data = []
    # Detect landmarks
    landmarks = detect_landmarks(image_array)
    #Golden Ratios
    ratio1, ratio2, ratio3, ratio4, ratio5, ratio6, ratio7, ratio8, ratio9, ratio10, ratio11 = calculate_golden_ratios(landmarks)

    #Symmetric Ratios
    lower_eyebrow_length, lower_lip_length, upper_eyebrow, upper_lip, nose = calculate_symmetric_ratios(landmarks)

    # Compute GLCM features
    glcm_features = compute_glcm_features(image_array)

    #Hu Moments
    hu_moments = compute_hu_moments(image_array)

    feature_data.append({'UE/IO': ratio1, 'UE/NW': ratio2, 'MW/IO': ratio3, 'ULJ/IO': ratio4, 'ULJ/NW': ratio5, 'IO/LH': ratio6,
                     'NW/IO': ratio7, 'NW/ULH': ratio8, 'IO/NMH': ratio9, 'EBN/NJ': ratio10, 'IO/NW': ratio11, 'LEL': lower_eyebrow_length,
                     'LLL': lower_lip_length, 'UEB': upper_eyebrow, 'UL': upper_lip, 'Nose': nose, 'Homogeneity': glcm_features[0],
                         'Contrast': glcm_features[1], 'Energy': glcm_features[2],
                         'Correlation': glcm_features[3], 'HuM1': hu_moments[0],
                         'HuM2': hu_moments[1], 'HuM3': hu_moments[2],
                         'HuM4': hu_moments[3], 'HuM5': hu_moments[4], 'HuM6': hu_moments[5], 'HuM7': hu_moments[6]  })

    # Convert the list of features to a dataframe
    df = pd.DataFrame.from_dict(feature_data)
    # Scale the Data
    df_scaled = scaler.transform(df)
    cols = df.columns
    df_scaled = pd.DataFrame(df_scaled, columns = cols)
    actual_rating = { 1.0: 0 , 1.5: 1 , 2.0: 2 , 2.5: 3 , 3.0: 4 , 3.5: 5 , 4.0: 6 , 4.5: 7 , 5.0: 8}
    key_list = list(actual_rating.keys())
    beauty_rating = key_list[int(model.predict(df_scaled))]


    return beauty_rating

if __name__ == '__main__':
    app.run(debug=True)

