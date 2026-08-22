from flask import Flask, render_template, request, abort
import keras
import numpy as np
from PIL import Image
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

# Maps a class_names prefix to the same distress-type labels used on the
# homepage's "What it detects" cards, so the result page doesn't leak the
# raw training-label casing (e.g. "Longitudinal-traverse") to the UI.
DISTRESS_TYPE_LABELS = {
    'Longitudinal-traverse': 'Cracking',
    'patch': 'Patching',
    'pothole': 'Potholes',
    'Ravelling and weathering': 'Ravelling & weathering',
    'Rutting': 'Rutting',
}


def describe_prediction(predicted_class):
    """Splits a raw class_names entry into a display-ready (type, severity)
    pair. Severity is None for Rutting, the one class with no severity split."""
    severity = next((s for s in ('low', 'medium', 'high') if s in predicted_class), None)
    prefix = predicted_class
    for suffix in (' low severity', ' medium severity', ' high severity'):
        prefix = prefix.replace(suffix, '')
    distress_type = DISTRESS_TYPE_LABELS.get(prefix, prefix)
    return distress_type, severity


# Function to preprocess the image
def preprocess_image(image):
    # Resize the image to match the input size of your model
    image = resize_grayscale(image, 64)
    # Normalize pixel values
    img_array = image / 255.0
    # Expand dimensions to match the input shape expected by the model
    img_array = img_array.reshape(1, 64, 64, 1)
    return img_array


def classify_and_render(img):
    """Shared by the upload path and the sample-image path: runs the model
    on an already-opened PIL image, saves it as the displayed result image,
    and renders result.html."""
    img_array = preprocess_image(img)
    prediction = model.predict(img_array)
    predicted_class = class_names[np.argmax(prediction)]
    distress_type, severity = describe_prediction(predicted_class)
    img_path = os.path.join('static', 'uploaded_image.png')
    img.save(img_path)
    return render_template('result.html', distress_type=distress_type, severity=severity, image=img_path)


# Bundled example photos shown as "try a sample" thumbnails on /classify, so
# a visitor can see a real prediction without sourcing their own pavement
# photo. Keyed by the slug used in the URL, not the file's own name, so the
# route can't be pointed at an arbitrary path on disk. `file` is a static/
# filename, used both to open the image for prediction and to render the
# thumbnail via url_for('static', ...).
SAMPLE_IMAGES = {
    'pothole': {'label': 'Pothole', 'file': 'samples/pothole.png'},
    'patching': {'label': 'Patching', 'file': 'samples/patching.jpg'},
    'cracking': {'label': 'Cracking', 'file': 'samples/cracking.jpg'},
}


# Route for home page
@app.route('/')
def index():
    return render_template('home.html')


# Route for the upload/classify page
@app.route('/classify')
def classify():
    return render_template('classify.html', samples=SAMPLE_IMAGES)


# Runs a bundled sample image through the same model/render path as a real
# upload, so a visitor can see one real prediction with no file of their own.
@app.route('/classify/sample/<name>')
def classify_sample(name):
    sample = SAMPLE_IMAGES.get(name)
    if not sample:
        abort(404)
    img = Image.open(os.path.join('static', sample['file']))
    return classify_and_render(img)


# Route explaining the model architecture. Named model_page, not model, since
# `model` above is the loaded Keras model.
@app.route('/model')
def model_page():
    return render_template('model.html')


# Route to handle file upload and prediction
@app.route('/predict', methods=['POST'])
def predict():
    file = request.files.get('file')
    if not file or not file.filename:
        return render_template('classify.html', error='Choose an image before analyzing.', samples=SAMPLE_IMAGES)
    try:
        img = Image.open(file)
        return classify_and_render(img)
    except Exception:
        return render_template('classify.html', error="That file doesn't look like a valid image. Try a JPG or PNG.", samples=SAMPLE_IMAGES)


if __name__ == '__main__':
    app.run(debug=True)
