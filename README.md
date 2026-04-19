# Pavement Distress Detection

A web application that classifies road pavement distresses from uploaded images using a convolutional neural network.

## Overview

Upload a photo of a road surface and the app will identify the type and severity of pavement distress present. The model recognizes 13 categories across four distress types:

- **Longitudinal-traverse cracking** — low / medium / high severity
- **Patching** — low / medium / high severity
- **Potholes** — low / medium / high severity
- **Ravelling and weathering** — low / medium / high severity
- **Rutting**

## Setup

Install dependencies:

```bash
pip install flask keras tensorflow pillow numpy scikit-learn matplotlib
```

## Running

```bash
python backend.py
```

Open `http://127.0.0.1:5000` in your browser, upload a road image, and click **Upload and Predict**.

## Model

The CNN (`distressCNN.keras`) is a Keras Sequential model trained on the [Flexible Pavement Distresses](https://www.kaggle.com/) dataset. It takes 64×64 grayscale images as input and outputs a softmax distribution over the 13 classes.

To retrain the model, uncomment the training block in `main.py`, point the dataset path to your local copy, and run it.

## Tech Stack

- **Backend:** Flask, Keras / TensorFlow
- **Image processing:** Pillow, NumPy
- **Frontend:** HTML, CSS (no framework)

---

Created by Muaz Alagroudi — [LinkedIn](https://www.linkedin.com/in/muaz-alagroudi-3b00bb223) · [GitHub](https://github.com/Muaz-Alagroudi)
