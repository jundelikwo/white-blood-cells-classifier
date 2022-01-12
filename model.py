import numpy as np
import math, cv2, os
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPool2D, Flatten, Dense, Dropout
from tensorflow.keras.callbacks import ModelCheckpoint

CATEGORIES = ['EOSINOPHIL', 'LYMPHOCYTE', 'MONOCYTE', 'NEUTROPHIL']
DATASET_DIR = 'images'

# Plot Image
def plotImage(image_path):
    image = cv2.imread(image_path) # Reads image in blue green red
    image = image[:, :, [2, 1, 0]] # Reorders to read green blue
    image = image.astype('float32') / 255
    plt.imshow(image)

# Plot images of the different cell types
plt.figure(figsize = (15, 8))
plt.subplot(221)
plt.title('Eosinophil'); plt.axis('off'); plotImage('images/TRAIN/EOSINOPHIL/_0_207.jpeg')
plt.subplot(222)
plt.title('Lymphocyte'); plt.axis('off'); plotImage('images/TRAIN/LYMPHOCYTE/_0_204.jpeg')
plt.subplot(223)
plt.title('Monocyte'); plt.axis('off'); plotImage('images/TRAIN/MONOCYTE/_0_180.jpeg')
plt.subplot(224)
plt.title('Neutrophil'); plt.axis('off'); plotImage('images/TRAIN/NEUTROPHIL/_0_292.jpeg')

# Summary of training data
print('Training data:')
train_dir = os.path.join(DATASET_DIR, 'TRAIN')
num_data = 0
for sub_dir in os.listdir(train_dir):
    sub_dir_path = os.path.join(train_dir, sub_dir)
    if os.path.isdir(sub_dir_path) == False: continue;

    num_images = len(os.listdir(sub_dir_path))
    num_data += num_images
    print('Cell Type: {:15s} number of images: {:d}'.format(sub_dir, num_images))

print('Total training data: {:d}'.format(num_data))

# Summary of test data
print('Test data:')
test_dir = os.path.join(DATASET_DIR, 'TEST')
num_data = 0
for sub_dir in os.listdir(test_dir):
    sub_dir_path = os.path.join(test_dir, sub_dir)
    if os.path.isdir(sub_dir_path) == False: continue;

    num_images = len(os.listdir(sub_dir_path))
    num_data += num_images
    print('Cell Type: {:15s} number of images: {:d}'.format(sub_dir, num_images))

print('Total test data: {:d}'.format(num_data))


# Image Processing
train_datagen = ImageDataGenerator(rescale = 1./255)
test_datagen = ImageDataGenerator(rescale = 1./255)

training_set = train_datagen.flow_from_directory(train_dir,
                                                 target_size = (64, 64),
                                                 batch_size = 32,
                                                 color_mode = 'rgb',
                                                 shuffle = True,
                                                 seed = None,
                                                 class_mode = 'categorical')

test_set = test_datagen.flow_from_directory(test_dir,
                                            target_size = (64, 64),
                                            batch_size = 32,
                                            color_mode = 'rgb',
                                            shuffle = True,
                                            seed = None,
                                            class_mode = 'categorical')


# Initialising the CNN
classifier = Sequential()

# Step 1 - Convolution
classifier.add(Conv2D(filters = 32, kernel_size = 3, input_shape = (64, 64, 3), activation = 'relu'))

# Step 2 - Pooling
classifier.add(MaxPool2D(pool_size = 2))
classifier.add(Dropout(0.20)) # Randomly dropout 20% of the weights in this layer

# Add Second Convolutional layer
classifier.add(Conv2D(filters = 64, kernel_size = 3, activation = 'relu'))
classifier.add(MaxPool2D(pool_size = 2))
classifier.add(Dropout(0.20))

# Add Third Convolutional layer
classifier.add(Conv2D(filters = 128, kernel_size = 3, activation = 'relu'))
classifier.add(MaxPool2D(pool_size = 2))
classifier.add(Dropout(0.20))

# Step 3 - Flatten
classifier.add(Flatten())

# Step 4 - Full Connection
classifier.add(Dense(units = 64, activation = 'relu'))
classifier.add(Dense(units = 128, activation = 'relu'))
classifier.add(Dense(units = 64, activation = 'relu'))
classifier.add(Dense(units = 4, activation = 'softmax')) # Softmax ensures that the sum of th output is 1

# Compile the CNN
classifier.compile(optimizer = 'rmsprop', loss = 'categorical_crossentropy', metrics = ['accuracy'])

# Fitting the model
# history = classifier.fit(training_set,
#                                    steps_per_epoch = 100,
#                                    epochs = 25,
#                                    validation_data = test_set,
#                                    validation_steps = 100)

filepath = "best_weights.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='val_accuracy', verbose=1, save_best_only=True, mode='max')
callbacks_list = [checkpoint]

history = classifier.fit(x = training_set,
                         validation_data = test_set,
                         epochs = 32,
                         callbacks=callbacks_list)

print(history.history.keys())
classifier.summary()

# Plot the accuracy
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc = 'upper left')
plt.show()

# Plot the loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc = 'upper left')
plt.show()


# Export the model
filename = 'model.pkl'
# tf.keras.models.save_model(classifier, filename)
classifier.save(filename)

# Make a prediction
loaded_model = tf.keras.models.load_model(filename)
y_pred = loaded_model.predict(test_set)
y_pred = y_pred.argmax(axis = -1) # argmax returns the index of the highest value
# axis = -1 indicate that the y axis should be used

y_true = test_set.classes
y_true


# Evaluating the model
from sklearn.metrics import confusion_matrix, accuracy_score
cm = confusion_matrix(y_true, y_pred)

print(cm)
print(accuracy_score(y_true, y_pred))


import itertools
def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
  """
  This function prints and plots the confusion matrix.
  Normalization can be applied by setting `normalize=True`.
  """
  if normalize:
      cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
      print("Normalized confusion matrix")
  else:
      print('Confusion matrix, without normalization')

  print(cm)

  plt.imshow(cm, interpolation='nearest', cmap=cmap)
  plt.title(title)
  plt.colorbar()
  tick_marks = np.arange(len(classes))
  plt.xticks(tick_marks, classes, rotation=45)
  plt.yticks(tick_marks, classes)

  fmt = '.2f' if normalize else 'd'
  thresh = cm.max() / 2.
  for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
      plt.text(j, i, format(cm[i, j], fmt),
               horizontalalignment="center",
               color="white" if cm[i, j] > thresh else "black")

  plt.tight_layout()
  plt.ylabel('Actual label')
  plt.xlabel('Predicted label')
  plt.show()
  
labels = [None] * len(test_set.class_indices)
for k, v in test_set.class_indices.items():
  labels[v] = k
  
plot_confusion_matrix(cm, labels, title='Test confusion matrix')


# Validation data
validation_datagen = ImageDataGenerator(rescale = 1./255)

validation_set = validation_datagen.flow_from_directory(os.path.join(DATASET_DIR, 'TEST_SIMPLE'),
                                                 target_size = (64, 64),
                                                 batch_size = 32,
                                                 color_mode = 'rgb',
                                                 shuffle = True,
                                                 seed = None,
                                                 class_mode = 'categorical')
predictions = classifier.evaluate(validation_set)

print("LOSS:  " + "%.4f" % predictions[0])
print("ACCURACY:  " + "%.2f" % predictions[1])

predictions = np.argmax(classifier.predict(validation_set), axis=1)
CLASS_NAMES = list(training_set.class_indices.keys())
CLASS_NAMES

acc = accuracy_score(validation_set.labels, predictions)
cm = tf.math.confusion_matrix(validation_set.labels, predictions)

import seaborn as sns

plt.figure(figsize=(8, 8))
sns.heatmap(cm, annot=True, fmt='g', vmin=0, cmap='Blues', cbar=False)
plt.xticks(ticks= np.arange(4) + 0.5, labels=CLASS_NAMES)
plt.yticks(ticks= np.arange(4) + 0.5, labels=CLASS_NAMES)
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()

