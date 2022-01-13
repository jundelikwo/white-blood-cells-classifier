# Inspired by https://www.kaggle.com/mehmetlaudatekman/finetuned-resnet50-83-accuracy

import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Input, Lambda, GlobalAveragePooling2D, Dense
from tensorflow.keras.models import Sequential
from tensorflow.keras.applications import ResNet50, resnet50
from tensorflow.keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(vertical_flip = True,
                                   horizontal_flip = True,
                                   rotation_range = 0.2,
                                   validation_split = 0.2)

validation_datagen = ImageDataGenerator()

second_validation_datagen = ImageDataGenerator()


training_set = train_datagen.flow_from_directory('images/TRAIN',
                                                      subset = 'training',
                                                      target_size = (224, 224),
                                                      batch_size = 32,
                                                      color_mode = 'rgb',
                                                      shuffle = True,
                                                      class_mode = 'categorical')

test_set = train_datagen.flow_from_directory('images/TRAIN',
                                                      subset = 'validation',
                                                      target_size = (224, 224),
                                                      batch_size = 32,
                                                      color_mode = 'rgb',
                                                      shuffle = True,
                                                      class_mode = 'categorical')

validation_set = validation_datagen.flow_from_directory('images/TEST',
                                                        target_size = (224, 224),
                                                        color_mode = 'rgb',
                                                        shuffle = False,
                                                        class_mode = 'categorical')

second_validation_set = second_validation_datagen.flow_from_directory('images/TEST_SIMPLE',
                                                        target_size = (224, 224),
                                                        color_mode = 'rgb',
                                                        shuffle = False,
                                                        class_mode = 'categorical')



base_model = ResNet50(include_top = False, weights = 'imagenet')
"""
include_top = False => Don't add fully connected layers 
weights = "imagenet" => Download weights trained on imagenet (an image dataset that contain 1000 classes)
"""
base_model.summary()

print("There are {} layers in model".format(len(base_model.layers)))


# We will freeze the layers under 140
for layer in base_model.layers[:140]:
    layer.trainable = False

base_model.summary()


model = Sequential()

model.add(Input(shape = (224, 224, 3)))
model.add(Lambda(resnet50.preprocess_input))
model.add(base_model)
model.add(GlobalAveragePooling2D())
model.add(Dense(4, activation = 'softmax'))

model.summary()


model.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])

history = model.fit(x = training_set, epochs = 3, validation_data = test_set)


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
filename = 'resnet.pkl'
model.save(filename)


# Predictions with first validation set
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

y_pred = model.predict(validation_set)
y_pred = y_pred.argmax(axis = -1)


y_true = validation_set.labels

print("Final test accuracy is {}%".format(accuracy_score(y_pred=y_pred,y_true=y_true)))


print(classification_report(
    y_true, 
    y_pred, 
    target_names = CLASS_NAMES))

confMatrix = confusion_matrix(y_pred=y_pred,y_true=y_true)
CLASS_NAMES = list(training_set.class_indices.keys())

plt.subplots(figsize=(6,6))
sns.heatmap(confMatrix,annot=True,fmt=".1f",linewidths=1.5)
plt.xticks(ticks= np.arange(4) + 0.5, labels=CLASS_NAMES)
plt.yticks(ticks= np.arange(4) + 0.5, labels=CLASS_NAMES)
plt.xlabel("Predicted Label")
plt.ylabel("Actual Label")
plt.show()



# Predictions with second validation set
y_pred = model.predict(second_validation_set)
y_pred = y_pred.argmax(axis = -1)


y_true = second_validation_set.labels

print("Final test accuracy is {}%".format(accuracy_score(y_pred=y_pred,y_true=y_true)))


print(classification_report(
    y_true, 
    y_pred, 
    target_names = CLASS_NAMES))

confMatrix = confusion_matrix(y_pred=y_pred,y_true=y_true)

plt.subplots(figsize=(6,6))
sns.heatmap(confMatrix,annot=True,fmt=".1f",linewidths=1.5)
plt.xticks(ticks= np.arange(4) + 0.5, labels=CLASS_NAMES)
plt.yticks(ticks= np.arange(4) + 0.5, labels=CLASS_NAMES)
plt.xlabel("Predicted Label")
plt.ylabel("Actual Label")
plt.show()
