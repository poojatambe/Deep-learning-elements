######################################################### 1 X 1 convolution #################################################
# padding same: As output size same as input size is required, accordingly padding is adjusted.
# Stride= 1.

from keras.models import Sequential
from tensorflow.keras.layers import Conv2D
# create model
model = Sequential()
model.add(Conv2D(512, (3,3), padding='same', activation='relu', input_shape=(256, 256, 3)))
# summarize model
model.summary()

# The no of filters in 1 X 1 conv decides increase or decrease of feature maps. 
# In this example, 512: feature map projection, 64(< 512): decrease no of feature maps, 1024(>512): increase no of feature maps.
# The no of parameters are less when 1x1 conv is used than 3x3 conv with same padding.
# create model
model = Sequential()
model.add(Conv2D(512, (3,3), padding='same', activation='relu', input_shape=(256, 256, 3)))
model.add(Conv2D(64, (1,1), activation='relu'))
# summarize model
model.summary() 


# create model
model = Sequential()
model.add(Conv2D(512, (3,3), padding='same', activation='relu', input_shape=(256, 256, 3)))
model.add(Conv2D(64, (3,3), activation='relu', padding='same'))
# summarize model
model.summary()