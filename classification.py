import numpy as np
import matplotlib.pyplot as plt
import os
import cv2
import random
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, Flatten, Conv2D, MaxPooling2D, Dropout
from keras.callbacks import ReduceLROnPlateau
from keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers
from tensorflow.keras import regularizers

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import itertools

train_dir = "/Users/lenaschill/Desktop/Skin_Dataset/training"
test_dir = "/Users/lenaschill/Desktop/Skin_Dataset/testing"

CATEGORIES = ["Cancer", "Non_Cancer"]
IMG_SIZE = 224

training_data = []
testing_data = []

def create_data(dir, dataset):
    for category in CATEGORIES:
        path = os.path.join(dir, category)
        class_num = CATEGORIES.index(category)
        for img in os.listdir(path):
            tmp_path = os.path.join(path, img)
            src = cv2.imread(tmp_path)
            
            #convert original image to grayscale
            grayScale = cv2.cvtColor(src, cv2.COLOR_RGB2GRAY)
            
            # Kernel for the morphological filtering
            kernel = cv2.getStructuringElement(1,(17,17))
        
            # Perform the blackHat filtering on the grayscale image to find the 
            # hair countours
            blackhat = cv2.morphologyEx(grayScale, cv2.MORPH_BLACKHAT, kernel)
        
            # intensify the hair countours in preparation for the inpainting 
            # algorithm
            ret,thresh2 = cv2.threshold(blackhat,10,255,cv2.THRESH_BINARY)
            
            # inpaint the original image depending on the mask
            dst = cv2.inpaint(src,thresh2,1,cv2.INPAINT_TELEA)
            new_array = cv2.resize(dst, (IMG_SIZE, IMG_SIZE))
            dataset.append([new_array, class_num])
            

create_data(train_dir, training_data)
create_data(test_dir, testing_data)
            
random.shuffle(training_data)
random.shuffle(testing_data)

X_train = []
X_test = []

y_train = []
y_test = []

for features, label in training_data:
    X_train.append(features)
    y_train.append(label)
    
for features, label in testing_data:
    X_test.append(features)
    y_test.append(label)
    
X_train = np.array(X_train)
X_test = np.array(X_test)

y_train = tf.keras.utils.to_categorical(y_train, num_classes=2)
y_test = tf.keras.utils.to_categorical(y_test, num_classes=2)

#Normalizing all values of the pictures by dividing all the RGB values by 255
X_train = X_train/255.0
X_test = X_test/255.0

#Splitting data into validation and training set
X_train, X_validation, y_train, y_validation = train_test_split(X_train, y_train, test_size = 0.2, random_state = 2)

#Data augmentation to prevent overfitting 
datagen = ImageDataGenerator(
            rotation_range = 30, 
            width_shift_range=0.2, 
            height_shift_range=0.2,
            shear_range=0.2,
            zoom_range=0.2,
            vertical_flip=True,
            horizontal_flip=True)  


datagen.fit(X_train)

#L1 and L2 regularization to prevent overfitting
layer = layers.Dense(units=5,
                     kernel_initializer='ones',
                     kernel_regularizer=regularizers.L1(0.01),
                     activity_regularizer=regularizers.L2(0.01))

#Building the model
def build(input_shape= (224,224,3), lr = 1e-3, num_classes= 2,
          activ = 'relu', optim= 'adam'):
    model = Sequential()
    model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 3)))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    model.add(Conv2D(64, (3, 3), activation='relu', input_shape=(224, 224, 3)))
    model.add(MaxPooling2D(pool_size = (2, 2)))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dense(num_classes, activation='softmax'))

    model.summary()

    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

input_shape = (224,224,3)
lr = 1e-5
activ = 'relu'
optim = 'adam'
epochs = 50
batch_size = 64

model = build(lr=lr, activ= activ, optim=optim, input_shape= input_shape)

# Set a learning rate annealer
learning_rate_reduction = ReduceLROnPlateau(monitor='val_accuracy', 
                                            patience=3, 
                                            verbose=1, 
                                            factor=0.5, 
                                            min_lr=0.00001)


#training the model with validation data
history = model.fit_generator(datagen.flow(X_train,y_train, batch_size=batch_size),
                              epochs = epochs, validation_data = (X_validation,y_validation), callbacks=[learning_rate_reduction])


# list all data in history
print(history.history.keys())

# summarize history for accuracy
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.show()

# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

#testing the Model
#Fitting model to all data
model = build(lr=lr,  
              activ= activ, 
              optim=optim, 
              input_shape= input_shape)

history = model.fit_generator(datagen.flow(X_train,y_train, batch_size=batch_size),
                              epochs = epochs, callbacks=[learning_rate_reduction])


#Saving the model 
model.save('skin_cancer_classification.h5')

loss, accuracy = model.evaluate(X_test, y_test, verbose=1)
loss_v, accuracy_v = model.evaluate(X_validation, y_validation, verbose=1)
print("Validation: accuracy = %f  ;  loss_v = %f" % (accuracy_v, loss_v))
print("Test: accuracy = %f  ;  loss = %f" % (accuracy, loss))


# Function to plot confusion matrix    
def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

# Predict the values from the validation dataset
Y_pred = model.predict(X_validation)
# Convert predictions classes to one hot vectors 
Y_pred_classes = np.argmax(Y_pred,axis = 1) 
# Convert validation observations to one hot vectors
Y_true = np.argmax(y_validation,axis = 1) 
# compute the confusion matrix
confusion_mtx = confusion_matrix(Y_true, Y_pred_classes)

# plot the confusion matrix
plot_confusion_matrix(confusion_mtx, classes = range(2)) 




