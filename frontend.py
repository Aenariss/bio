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
        # Process the images and obtain vein masks
        img1, mask1 = pipeline(image1_data)
        img2, mask2 = pipeline(image2_data)

        # Initialize comparator
        comparator = Comparator()

        # Align the second image with the first using ECC (Enhanced Correlation Coefficient)
        img2 = comparator.align_images(img1, img2, mask1, mask2)

        # Extract feature descriptors from both images
        features1 = FeatureExtractor(img1)
        descriptor1 = features1.get_features()

        features2 = FeatureExtractor(img2)
        descriptor2 = features2.get_features()

    except Exception as e: 
        print(e)
        return render_template('index.html', error=True)

    # Calculate and print the similarity score
    score = comparator.compare_descriptors(descriptor1, descriptor2)

    result = score < comparator.threshold

    return render_template('index.html', result=result, score=score, threshold=comparator.threshold)

if __name__ == '__main__':
    app.run()
