import keras
import tensorflow as tf
import numpy as np
from keras.src.layers import Conv2D, MaxPooling2D
from sklearn.model_selection import train_test_split
from images import *

class_names = ['L-T_l', 'L-T_m', 'L-T_h', 'patch_l', 'patch_m', 'patch_h',
               'pothole_l', 'pothole_m', 'pothole_h', 'R-W_l', 'R-W_m', 'R-W_h', 'Rutting']

# paths = get_leaf_directory_paths('C:\Programming\Python\datasets\Flexible Pavement Distresses - aug')
# image_array, labels = load_images_from_dirs(paths)
# # print("Shape of the image array:", image_array.shape)
# # print("label length:", len(labels))
#
# x_train, x_test, y_train, y_test = train_test_split(image_array, labels, test_size=0.2, shuffle=True, random_state=40)
# x_train = x_train / 255.0
# x_test = x_test / 255.0
#
sample = Image.open('patching_h.jpeg')
sample = resize_grayscale(sample, 64)
sample = sample / 255
sample = sample.reshape(1, 64, 64, 1)
print(sample.shape)

# def create_cnn_model(input_shape, num_classes):
#     model = tf.keras.Sequential([
#         Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=input_shape),
#         MaxPooling2D(pool_size=(2, 2)),
#         Conv2D(64, kernel_size=(3, 3), activation='relu'),
#         MaxPooling2D(pool_size=(2, 2)),
#         tf.keras.layers.Flatten(),
#         tf.keras.layers.Dense(128, activation='relu'),
#         tf.keras.layers.Dense(num_classes, activation='softmax')
#     ])
#     return model
#
#
# input_shape = (64, 64, 1)  # Example shape for grayscale images of size 64x64
# num_classes = 13  # Example number of classes
# model = create_cnn_model(input_shape, num_classes)
#
# model.compile(optimizer='adam',
#               loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
#               metrics=['accuracy'])
#
# model.fit(x_train, y_train, epochs=40, validation_split=0.1)
#
# test_loss, test_acc = model.evaluate(x_test, y_test, verbose=2)
#
# print('\nTest accuracy:', test_acc)
#
#
# probability_model = tf.keras.Sequential([model,
#                                          tf.keras.layers.Softmax()])
#
# probability_model.save('distressCNN.keras')
# print(sample.shape)
model = keras.models.load_model('distressCNN.keras')
# model.compile(optimizer='adam',
#               loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
#               metrics=['accuracy'])
# test_loss, test_acc = model.evaluate(x_test, y_test, verbose=2)

# print('\nTest accuracy:', test_acc)
# dim = x_test.shape
# print(dim)
# x_test = x_test.reshape(156, 64, 64, 1)
#
#
# predictions = model.predict(sample)
predict_sample(model, sample, class_names)

# def plot_image(i, predictions_array, true_label, img):
#     true_label, img = true_label[i], img[i]
#     plt.grid(False)
#     plt.xticks([])
#     plt.yticks([])
#
#     plt.imshow(img, cmap=plt.cm.binary)
#
#     predicted_label = np.argmax(predictions_array)
#     if predicted_label == true_label:
#         color = 'blue'
#     else:
#         color = 'red'
#
#     plt.xlabel("{} {:2.0f}% ({})".format(class_names[predicted_label],
#                                          100 * np.max(predictions_array),
#                                          class_names[true_label]),
#                color=color)
#
#
# def plot_value_array(i, predictions_array, true_label):
#     true_label = true_label[i]
#     plt.grid(False)
#     plt.xticks(range(13))
#     plt.yticks([])
#     thisplot = plt.bar(range(13), predictions_array, color="#777777")
#     plt.ylim([0, 1])
#     predicted_label = np.argmax(predictions_array)
#
#     thisplot[predicted_label].set_color('red')
#     thisplot[true_label].set_color('blue')
#
# num_rows = 5
# num_cols = 3
# num_images = num_rows*num_cols
# plt.figure(figsize=(2*2*num_cols, 2*num_rows))
# for i in range(num_images):
#   plt.subplot(num_rows, 2*num_cols, 2*i+1)
#   plot_image(i, predictions[i], y_test, x_test)
#   plt.subplot(num_rows, 2*num_cols, 2*i+2)
#   plot_value_array(i, predictions[i], y_test)
# plt.tight_layout()
# plt.show()
