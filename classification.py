import numpy as np
import matplotlib.pyplot as plt
import os
import cv2
import random
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, Flatten, Conv2D, MaxPooling2D, Dropout
from keras import optimizers
from keras.optimizers import Adam, RMSprop
from keras.callbacks import ReduceLROnPlateau
from sklearn.preprocessing import StandardScaler 
from sklearn.model_selection import train_test_split, KFold, cross_val_score, GridSearchCV
from sklearn.metrics import accuracy_score
from keras import backend as K 

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
            img_array = cv2.imread(os.path.join(path,img), -1)
            new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
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

history = model.fit(X_train, y_train, validation_split=0.2,
                    epochs= epochs, batch_size= batch_size)


# list all data in history
print(history.history.keys())
# summarize history for accuracy
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

K.clear_session()
del model
del history

#Cross Validation Model

# define 3-fold cross validation test harness
kfold = KFold(n_splits=3, shuffle=True, random_state=11)

cvscores = []
for train, test in kfold.split(X_train, y_train):
  # create model
    model = build(lr=lr,  
                  activ= activ, 
                  optim=optim, 
                  input_shape= input_shape)
    
    # Fit the model
    model.fit(X_train[train], y_train[train], epochs=epochs, batch_size=batch_size)
    # evaluate the model
    scores = model.evaluate(X_train[test], y_train[test], verbose=0)
    print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
    cvscores.append(scores[1] * 100)
    K.clear_session()
    del model
    
print("%.2f%% (+/- %.2f%%)" % (np.mean(cvscores), np.std(cvscores)))

#testing the Model

# Fitting model to all data
model = build(lr=lr,  
              activ= activ, 
              optim=optim, 
              input_shape= input_shape)

model.fit(X_train, y_train,
          epochs=epochs, batch_size= batch_size
         )

# Testing model on test data to evaluate
#y_pred = model.predict(X_test)



y_pred = model.predict(X_test)
#print(y_pred[:, 0])
for i in y_pred[:, 0]:
    if i < 0.5:
        i = 0
    else:
        i = 1
    
print(accuracy_score(y_test, y_pred))





