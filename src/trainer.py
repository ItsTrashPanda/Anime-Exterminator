import numpy as np
import os
import cv2
import random
import pickle
import tensorflow as tf
from tensorflow import keras

IMG_SIZE = 64  # Resize Images

training_data = []
testing_data = []

#'''
def create_training_data():
   count = 0
   directory = r"F:/AE/Datasets"
   categories = ["Anime", "Humans"]
   for cat in categories:
       path = os.path.join(directory, cat)  # Path To folder containing images
       class_num = categories.index(cat)
       for img in os.listdir(path):
           print(count)
           count += 1
           try:
               img_array = cv2.imread(os.path.join(path, img),
                                      cv2.IMREAD_GRAYSCALE)  # Grabs images and converts them to greyscale
               new_images = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))  # new images will be a 64 by 64 image
               new_images = cv2.Sobel(new_images, cv2.CV_64F, 1, 0, ksize=5)
               training_data.append([new_images, class_num])
           except Exception as e:
               print("passing at " + str(class_num))
               pass



create_training_data()
random.shuffle(training_data)

training_images = []
training_labels = []

for features, labels in training_data:
   training_images.append(features)
   training_labels.append(labels)

# training_images = np.array(training_images)

training_images = np.array(training_images).reshape(-1, IMG_SIZE, IMG_SIZE)
# training_images = training_images / 255.0
training_labels = np.array(training_labels)


# To save our data so it doesn't always have to be loaded
'''
pickle_out = open("training_images.pickle", "wb")
pickle.dump(training_images, pickle_out)
pickle_out.close()

pickle_out = open("training_labels.pickle", "wb")
pickle.dump(training_labels, pickle_out)
pickle_out.close()

pickle_in = open("training_images.pickle", "rb")
training_images = pickle.load(pickle_in)

pickle_in = open("training_labels.pickle", "rb")
training_labels = pickle.load(pickle_in)
#'''
# End of pickle section (optional for now)
'''
def create_test_data():
   count = 0
   directory = r"F:/AE/Datasets"
   categories = ["AnimeT", "HumansT"]
   for cat in categories:
       path = os.path.join(directory, cat)  # Path To folder containing images
       class_num = categories.index(cat)
       for img in os.listdir(path):
           print(count)
           count += 1
           try:
               img_array = cv2.imread(os.path.join(path, img), cv2.IMREAD_GRAYSCALE)  # Grabs images and converts them to grayscale
               new_images = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))  # new images will be a 64 by 64 image
               new_images = cv2.Sobel(new_images, cv2.CV_64F, 1, 0, ksize=5)
               testing_data.append([new_images, class_num])
           except Exception as e:
               print("passing at " + str(class_num))
               pass


create_test_data()
random.shuffle(testing_data)

testing_images = []
testing_labels = []

for features, labels in testing_data:
   testing_images.append(features)
   testing_labels.append(labels)

testing_images = np.array(testing_images).reshape(-1, IMG_SIZE, IMG_SIZE)

testing_images = testing_images / 255.0
#'''
training_images = training_images / 255.0

model = keras.Sequential([keras.layers.Flatten(input_shape=(64, 64)), keras.layers.Dense(128, activation=tf.nn.relu),
                         keras.layers.Dense(10, activation=tf.nn.softmax)])
model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
model.fit(training_images, training_labels, epochs=10)

model.save('F:/AE/Anime-Exterminator/model')
##test_loss, test_acc = model.evaluate(testing_images, testing_labels)
##print(f"Test Accuracy: {test_acc}")
# '''
'''
average = 0
predictions = model.predict(testing_images)

for num in range(len(predictions)):
   if np.argmax(predictions[num]) == testing_labels[num]:
       average += 1
       print(average)
   else:
       print("passing at img: {num}")

average = average / len(predictions)
print(average)
#'''