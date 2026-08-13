from flask import Flask, render_template, request
import keras
import numpy as np
from PIL import Image
import io
from images import resize_grayscale
import os

app = Flask(__name__)


class PrefixMiddleware:
    """Lets Flask generate correct URLs when reverse-proxied under a path prefix (e.g. /distress)."""

    def __init__(self, wsgi_app, prefix=''):
        self.wsgi_app = wsgi_app
        self.prefix = prefix

    def __call__(self, environ, start_response):
        if self.prefix and environ['PATH_INFO'].startswith(self.prefix):
            environ['PATH_INFO'] = environ['PATH_INFO'][len(self.prefix):]
            environ['SCRIPT_NAME'] = self.prefix
        return self.wsgi_app(environ, start_response)


app.wsgi_app = PrefixMiddleware(app.wsgi_app, prefix=os.environ.get('APP_PREFIX', ''))

# Load the Keras model
model = keras.models.load_model('distressCNN.keras')

# Define the classes (types of potholes)
class_names = ['Longitudinal-traverse low severity', 'Longitudinal-traverse medium severity', 'Longitudinal-traverse high severity',
               'patch low severity', 'patch medium severity', 'patch high severity',
               'pothole low severity', 'pothole medium severity', 'pothole high severity',
               'Ravelling and weathering low severity', 'Ravelling and weathering medium severity', 'Ravelling and weathering high severity',
               'Rutting']


# Function to preprocess the image
def preprocess_image(image):
    # Resize the image to match the input size of your model
    image = resize_grayscale(image, 64)
    # Normalize pixel values
    img_array = image / 255.0
    # Expand dimensions to match the input shape expected by the model
    img_array = img_array.reshape(1, 64, 64, 1)
    return img_array


# Route for home page
@app.route('/')
def index():
    return render_template('home.html')


# Route for the upload/classify page
@app.route('/classify')
def classify():
    return render_template('classify.html')


# Route to handle file upload and prediction
@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        # Get the uploaded file
        file = request.files['file']
        if file:
            # Read image from file
            # img = Image.open(io.BytesIO(file.read()))
            img = Image.open(file)
            # Preprocess the image
            img_array = preprocess_image(img)
            # Make prediction
            prediction = model.predict(img_array)
            # Get the predicted class
            predicted_class = class_names[np.argmax(prediction)]
            img_path = os.path.join('static', 'uploaded_image.png')
            img.save(img_path)
            print("Predicted class:", predicted_class)
            print("Image path:", file)
            return render_template('result.html', predicted_class=predicted_class, image=img_path)


if __name__ == '__main__':
    app.run(debug=True)
