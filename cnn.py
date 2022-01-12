# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPool2D, Flatten, Dense
tf.__version__


# Preprocessing the training set
train_datagen = ImageDataGenerator(rescale = 1./255,
                                   shear_range = 0.2,
                                   zoom_range = 0.2,
                                   horizontal_flip = True)

training_set = train_datagen.flow_from_directory('images/TRAIN',
                                                 target_size = (64, 64),
                                                 batch_size = 32,
                                                 class_mode = 'categorical')

# Preprocessing the test set
test_datagen = ImageDataGenerator(rescale = 1./255)

test_set = test_datagen.flow_from_directory('images/TEST',
                                            target_size = (64, 64),
                                            batch_size = 32,
                                            class_mode = 'categorical')


# Initializing the CNN
cnn = Sequential()

# Step 1 - Convolution
cnn.add(Conv2D(filters = 32,
               kernel_size = 3,
               activation = 'relu',
               input_shape = [64, 64, 3]))

# Step 2 - Pooling
cnn.add(MaxPool2D(pool_size = 2, strides = 2))

# Add Second convolutional layer
cnn.add(Conv2D(filters = 32,
               kernel_size = 3,
               activation = 'relu'))

cnn.add(MaxPool2D(pool_size = 2, strides = 2))


# Step 3 - Flattening
cnn.add(Flatten())

# Step 4 - Full Connection
cnn.add(Dense(units = 128, activation = 'relu'))

# Step 5 - Output Layer
cnn.add(Dense(units = 4, activation = 'softmax'))
cnn.output_shape


# Compiling the CNN
cnn.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])


# Training the cnn on the training set and evaluating on the test set
history = cnn.fit(x = training_set, validation_data = test_set, epochs = 25)


# Plot the loss
plt.plot(history.history['loss'], label='Loss (training data)')
plt.plot(history.history['val_loss'], label='Loss (validation data)')
plt.title('Training and validation loss')
plt.ylabel('Loss')
plt.xlabel('No. epoch')
plt.legend(loc="upper left")
plt.show()

# Plot the accuracy
plt.plot(history.history['accuracy'], label='Loss (training data)')
plt.plot(history.history['val_accuracy'], label='Loss (validation data)')
plt.title('Training and validation accuracy')
plt.ylabel('Accuracy')
plt.xlabel('No. epoch')
plt.legend(loc="upper left")
plt.show()


# Make a single prediction
test_image = image.load_img('images/TEST_SIMPLE/NEUTROPHIL/_39_1103.jpeg', target_size = (64, 64))
test_image = image.img_to_array(test_image)
test_image = np.expand_dims(test_image, axis = 0)
result = cnn.predict(test_image)
result
training_set.class_indices



# # Predict function
# def predict(file_name):
#     loaded_image = image.load_img(file_name, target_size = (64, 64))
#     loaded_image = image.img_to_array(loaded_image)
#     loaded_image = np.expand_dims(loaded_image, axis = 0)
#     predicted_result = cnn.predict(loaded_image)
#     predicted_value = None
    
#     if predicted_result[0][0] > 0.5:
#         predicted_value = 0
#     elif predicted_result[0][1] > 0.5:
#         predicted_value = 1
#     elif predicted_result[0][2] > 0.5:
#         predicted_value = 2
#     elif predicted_result[0][3] > 0.5:
#         predicted_value = 3
        
#     return predicted_value


# # Build Confusion Matrix
# import glob

# real_values = []
# predicted_values = []
# for file in glob.glob("images/TEST_SIMPLE/EOSINOPHIL/*"):
#     real_values.append(0)
#     predicted_values.append(predict(file))

# for file in glob.glob("images/TEST_SIMPLE/LYMPHOCYTE/*"):
#     real_values.append(1)
#     predicted_values.append(predict(file))
    
# for file in glob.glob("images/TEST_SIMPLE/MONOCYTE/*"):
#     real_values.append(2)
#     predicted_values.append(predict(file))
    
# for file in glob.glob("images/TEST_SIMPLE/NEUTROPHIL/*"):
#     real_values.append(3)
#     predicted_values.append(predict(file))


# from sklearn.metrics import confusion_matrix
# cm = confusion_matrix(real_values, predicted_values)
# print(cm)
