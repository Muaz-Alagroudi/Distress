# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What This Project Does

A Flask web application that classifies road pavement distresses from uploaded images using a pre-trained Keras CNN model (`distressCNN.keras`). Users upload a photo; the app preprocesses it and returns one of 13 distress categories with severity levels.

## Running the App

```bash
python backend.py
```

Flask runs in debug mode on `http://127.0.0.1:5000` by default.

## Architecture

- **[backend.py](backend.py)** — Flask app with two routes: `GET /` (upload form) and `POST /predict` (inference). Loads the model at startup, preprocesses uploads using `resize_grayscale`, saves the uploaded image to `static/uploaded_image.png`, and renders the result.
- **[images.py](images.py)** — Image utilities: `resize_grayscale` (PIL → 64×64 grayscale numpy array), dataset loading/augmentation helpers, and `predict_sample` (for notebook-style exploration with matplotlib).
- **[main.py](main.py)** — Standalone script for ad-hoc model testing; most logic is commented out. Not part of the web app.
- **[distressCNN.keras](distressCNN.keras)** — Pre-trained Keras Sequential CNN (Conv2D → MaxPool × 2 → Dense). Input: `(1, 64, 64, 1)` grayscale. Output: softmax over 13 classes.
- **[templates/](templates/)** — Jinja2 templates (`index.html` upload form, `result.html` prediction display).
- **[static/styles/](static/styles/)** — CSS for the two pages.

## 13 Distress Classes

The model outputs one of: Longitudinal-traverse (low/med/high), patch (low/med/high), pothole (low/med/high), Ravelling and weathering (low/med/high), Rutting.

## Image Preprocessing Pipeline

Every image fed to the model must go through `resize_grayscale(image, 64)` → divide by 255.0 → reshape to `(1, 64, 64, 1)`. This is done in both `backend.py` (`preprocess_image`) and `main.py`.

## Retraining the Model

The training code is commented out in `main.py`. To retrain: point `get_leaf_directory_paths` at the dataset root, uncomment the training block, run `main.py`. The dataset expected is "Flexible Pavement Distresses" with one subdirectory per class. `images.py` contains `augment_dir` to expand the dataset before training.
