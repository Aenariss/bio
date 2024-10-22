"""
BIO Project 2024
Author: 
    Vojtech Fiala <xfiala61>
    ChatGPT
"""

from flask import Flask, render_template, request

from src.Comparator import Comparator
from src.Pipeline import pipeline
from src.FeatureExtractor import FeatureExtractor

import numpy as np

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def index():

    if request.method != 'POST':
        return render_template('index.html', score=None)

    image1 = request.files.get('image1')
    image2 = request.files.get('image2')

    if not (image1 and image2):
        return render_template('index.html', score=None)
    
    # Convert it to np array
    image1_data = np.frombuffer(image1.read(), np.uint8)
    image2_data = np.frombuffer(image2.read(), np.uint8)

    try:
        img1 = pipeline(image1_data)
        img2 = pipeline(image2_data)
    except:
        return render_template('index.html', error=True)

    #descriptor1 = FeatureExtractor(img1).create_descriptor()
    #descriptor2 = FeatureExtractor(img2).create_descriptor()

    # The lower score the more similar they are
    #score = Comparator().compare_descriptors(descriptor1, descriptor2)

    score = Comparator(threshold=14).compare(img1, img2)

    result = score[0]
    score = score[1]

    return render_template('index.html', result=result, score=score)

if __name__ == '__main__':
    app.run()
