from flask import Flask, jsonify, request
from flask_cors import CORS, cross_origin
from matplotlib import cm
import os
from tensorflow import keras
import cv2
import librosa
import librosa.display
import matplotlib
import numpy as np
import tensorflow as tf
import uuid 


matplotlib.use('Agg')
img_size = 200
app = Flask(__name__)
CORS(app)
model1 = None
model2 = None
model3 = None
model4 = None

model1_name_ = 'CNN Model'
model2_name_ = 'Le-Net Model'
model3_name_ = 'LSTM Model'
model4_name_ = 'Feed-Forward Model'

model1_name = './Models/CNN/'
model2_name = './Models/LETNET/'
model3_name = './Models/LTSM/'
model4_name = './Models/FeedForward/'

model1_metrics = './Models/CNN.txt'
model2_metrics = './Models/LETNET.txt'
model3_metrics = './Models/LTSM.txt'
model4_metrics = './Models/FeedForward.txt'
    
def load_model(name):
    json_file = open('./Models/' + name + ".json", 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = tf.keras.models.model_from_json(loaded_model_json)
    loaded_model.load_weights('./Models/' + name + ".h5")
    print(name + " loaded model from disk")
    return loaded_model

def load_models():
    global model1
    model1 = load_model(model1_name_)
    
    global model2
    model2 = load_model(model2_name_)
    
    global model3
    model3 = load_model(model3_name_)
    
    global model4
    model4 = load_model(model4_name_)

def song_to_spectogram(path, savepath):
    data, sr = librosa.load(path, sr=44100)    #Amplitude vs time
    data_by_frequency = librosa.stft(data)
    Xdb = librosa.amplitude_to_db(abs(data_by_frequency))
    librosa.display.specshow(Xdb, sr=sr, x_axis='time', y_axis='log', hop_length=1024, cmap=cm.Spectral)
    matplotlib.pyplot.axis('off')
    matplotlib.pyplot.savefig(savepath, bbox_inches='tight', pad_inches=0)

def predict_with_CNN_model(path, genres, model):
    img_array = cv2.imread(path)[...,::-1]
    image = cv2.resize(img_array, (img_size, img_size))
    tests_files = [image]
    tests_files = np.array(tests_files) / 255
    tests_files.reshape(-1, img_size, img_size, 1)
    prediction = model.predict(tests_files)
    result = {}
    for i in range(0, len(prediction[0])):
        result[genres[i]] = prediction[0][i]
    return(result)

def predict_with_model1(path):
    return predict_with_CNN_model(path, ['classical', 'country','metal','pop'], model1)

def predict_with_model2(path):
    return predict_with_CNN_model(path, ['classical', 'country','metal','pop'], model2)
                           
def predict_with_model3(path):
    return predict_with_CNN_model(path, ['classical', 'country','metal','pop'], model3)

def predict_with_model4(path):
    return predict_with_CNN_model(path, ['classical', 'country','metal','pop'], model4)
                           

def serialize(results):
    new_results = {}
    for result in results:
        new_results[result] = str(results[result])
    return new_results

def get_current_accuracy(model_metrics_path):
    metrics = {}
    with open(model_metrics_path) as f:
        lines = f.readlines()
        for line in lines:
            if ('accuracy' in line):
                metrics['accuracy'] = line.split(" : ")[-1]
            if ('loss' in line):
                metrics['loss'] = line.split(" : ")[-1]
    return metrics


@app.route('/', methods=['POST'])
@cross_origin()
def index():
    print("Request received")
    # Initiate returned structure
    data = {
        "success": False
        }
    
    name = uuid.uuid4()
    audio_name = './Files/' + str(name) +'.wav'
    spectogram_name = './Files/' + str(name) +'.png'
    if request.method == "POST":
        if('audio_file' in request.files):
            #Get the audio and save in a temporal file
            request.files['audio_file'].save(audio_name)
            
            #Generate the spectogram
            song_to_spectogram(audio_name, spectogram_name)
            
            #Predict with CNN Model 
            results1 = predict_with_model1(spectogram_name)
            
            #Predict with Let Net Model 
            results2 = predict_with_model2(spectogram_name)
            
            #Predict with LSTM Model 
            results3 = predict_with_model3(spectogram_name)
            
            #Predict with Feed Forward Model 
            results4 = predict_with_model4(spectogram_name)
            

            data = {}
            data[model1_name_] = {} 
            data[model1_name_]['data'] = serialize(results1)
            data[model1_name_]['metrics'] = get_current_accuracy(model1_metrics)
            
            data[model2_name_] = {} 
            data[model2_name_]['data'] = serialize(results2)
            data[model2_name_]['metrics'] = get_current_accuracy(model2_metrics)
            
            
            data[model3_name_] = {} 
            data[model3_name_]['data'] = serialize(results3)
            data[model3_name_]['metrics'] = get_current_accuracy(model3_metrics)
            
            data[model4_name_] = {} 
            data[model4_name_]['data'] = serialize(results4)
            data[model4_name_]['metrics'] = get_current_accuracy(model4_metrics)
            
            os.remove(audio_name)
            os.remove(spectogram_name)
            # return the data dictionary as a JSON response
    print("Data sent")
    return jsonify(data)

if __name__ == "__main__":
    print(("* Loading Keras model and Flask starting server..."
        "please wait until server has fully started"))
    load_models()
    app.run(host='localhost', port=8000)