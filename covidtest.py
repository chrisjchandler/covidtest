import os
import cv2
import numpy
from PIL import Image
from flask import Flask, request, render_template

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/detect', methods=['POST'])
def detect():
    # Get the uploaded image
    file = request.files['image']
    img = Image.open(file.stream)

    # Convert the image to grayscale
    img = img.convert('L')

    # Convert the image to OpenCV format
    img_cv = cv2.cvtColor(numpy.array(img), cv2.COLOR_RGB2BGR)

    # Apply thresholding to the image
    ret, thresh = cv2.threshold(img_cv, 127, 255, cv2.THRESH_BINARY)

    # Detect lines in the image
    lines = cv2.HoughLines(thresh, 1, numpy.pi/180, 100)

    # Count the number of lines
    if lines is not None and len(lines) >= 2:
        result = 'Positive'
    else:
        result = 'Negative'

    # Return the result
    return render_template('result.html', result=result)

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 8001))
    app.run(host='0.0.0.0', port=port)
