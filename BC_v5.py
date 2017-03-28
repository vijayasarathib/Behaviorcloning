import csv
import cv2
import numpy as np
from random import shuffle
import matplotlib.pyplot as plt
import h5py
from random import randint
lines =[]
with open('/home/carnd/data/driving_log.csv') as csvfile:
    reader=csv.reader(csvfile)
    for line in reader:
        lines.append(line)


from sklearn.model_selection import train_test_split
train_samples, validation_samples = train_test_split(lines, test_size=0.2)

import cv2
import numpy as np
import sklearn

def generator(samples, batch_size=32):
    #num_samples = len(samples)
    num_samples=randint(0,len(samples))
    while 1: # Loop forever so the generator never terminates
        shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]
            images = []
            angles = []
            for row in batch_samples:
                if float(row[3]) > 0. or float(row[3]) < 0.:
                    #print(row[3])
                    for i in range(3):
                        source_path =line[i]
                        filename = source_path.split('/')[-1]
                        current_path ='/home/carnd/data/IMG/'+filename
                        image = cv2.imread(current_path)
                        img = cv2.cvtColor(image,cv2.COLOR_BGR2HSV)
                        random_bright =.25+np.random.uniform()
                        #print(random_bright)
                        img[:,:,2] = img[:,:,2]*random_bright
                        img = cv2.cvtColor(img,cv2.COLOR_HSV2RGB)
                        images.append(img)
                    steering_center = float(row[3])
                    # create adjusted steering measurements for the side camera images
                    correction = 0.4 # this is a parameter to tune
                    steering_left = steering_center + correction
                    steering_right = steering_center - correction


                    angles.append(steering_center)
                    angles.append(steering_left)            
                    angles.append(steering_right)

                # trim image to only see section with road
                augmented_images,augmented_measurements=[],[]
                for image,measurement in zip(images,angles):
                    augmented_images.append(image)
                    augmented_measurements.append(measurement)
                    augmented_images.append(cv2.flip(image,1))
                    augmented_measurements.append(measurement*-1.0)
        
        
            X_train = np.array(augmented_images)
            y_train = np.array(augmented_measurements)
           
            yield sklearn.utils.shuffle(X_train, y_train)

# compile and train the model using the generator function
train_generator = generator(train_samples, batch_size=32)
validation_generator = generator(validation_samples, batch_size=32)



from keras.models import Sequential,Model
from keras.layers import Flatten,Dense,Lambda,Cropping2D
from keras.layers import Dropout,Activation
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D

#model = Sequential()


model = Sequential()
model.add(Lambda(lambda x:(x /255.0) - 0.5,input_shape=(160,320,3)))
model.add(Cropping2D(cropping=((70,25),(0,0))))

model.add(Convolution2D(24,5,5,subsample=(2,2),activation="relu"))
model.add(Convolution2D(36,5,5,subsample=(2,2),activation="relu"))
#model.add(MaxPooling2D())#
model.add(Convolution2D(48,5,5,subsample=(2,2),activation="relu"))
model.add(Convolution2D(64,3,3,activation="relu"))
model.add(Convolution2D(64,3,3,activation="relu"))


#model.add(Convolution2D(6,5,5,activation="relu"))
#model.add(MaxPooling2D())
#model.add(Convolution2D(6,5,5,activation="relu"))

model.add(Flatten())
#model.add(Dropout(0.5))
model.add(Dense(100))
model.add(Dense(50))
model.add(Dense(10))
model.add(Dropout(0.5))
model.add(Dense(1))

model.compile(loss='mse',optimizer='adam')




history=model.fit_generator(train_generator, samples_per_epoch=len(train_samples), validation_data=validation_generator,nb_val_samples=len(validation_samples), nb_epoch=14,verbose=1)


import h5py

model.save('model.h5')
        
        
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
#exit()

