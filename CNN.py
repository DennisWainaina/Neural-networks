x = 7
y = 8
if x < y:
    print(x, 'is smaller than', y)

# Today we're going to be looking at convolutional neural networks.
# These are basically neural networks that look for localised patterns in an image.
# They are different from dense neural networks as these networks look for patterns in specific locations.
# If however the location of this pattern moves if lets say the image is flipped the sense neural network,
# cannot identify the pattern.
# Convolutional neural networks are then better in this regard
# These patterns in which the CNN is looking for are called filters and this may be as many as 32 or more.
# If an image is say a 5 by 5 grid it looks at specific filters(patterns) which are in a 3 by 3 grid.
# It then looks how much the pattern(filter) is present in the image and then maps a number to an output feature map.
# The number ranges from 0 to 1 based on how much they are alike.
# The number of output feature maps is equal to the number of filters being looked for.
# The 5 by 5 grid is split into 3 by 3 grid of the filter and in each 3 by 3 grid of the image a number is assigned a number.
# As stated before this number depends on how alike the 3 by 3 grid of the image is to the 3 by 3 grid of the filter.
# This number is then assigned to the output feature maps in each space in its 3 by 3 grid.
# After this is done the output feature map is made smaller in a process known as pooling where the average, max or min
# of the output feature map is stored in a 2 by 2 grid in a similar process to how the original image was done.
# Pooling is done to show the model what to look for
# Now trying to build the model.

# First importing important libraries
import tensorflow as tf
import numpy as np
import keras
from keras import datasets, layers, models
import matplotlib.pyplot as plt
from keras.preprocessing.image import ImageDataGenerator
import tensorflow_datasets as tfds

# Splitting the data into training and test data set
(train_images, train_labels), (test_images, test_labels) = datasets.cifar10.load_data()

# Normalising pixel values to be between 0 and 1
train_images, test_images = train_images / 255.0, test_images / 255.0
class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck']

# Looking at one image
IMG_INDEX = 7  # So as to look at different images
plt.imshow(train_images[IMG_INDEX], cmap=plt.cm.binary)
plt.xlabel(class_names[train_labels[IMG_INDEX][0]])
plt.show()

# Building the model which has the CNN layer and the pooling layer
model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))

# Adding dense layers
model.add(layers.Flatten())
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(10))

# Summarising the model
model.summary()

# Training the model
model.compile(optimizer='adam', loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])
history = model.fit(train_images, train_labels, epochs=6, validation_data=(test_images,
                                                                            test_labels))

# Evaluating model to see how it performs on the test data.
test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=2)
print(test_acc)  # 0.6982

y = 6
print('The first value is', y)
# To get an accurate model such that the model correctly predicts or classifies most of the time
# The model may need to be fed with millions of pictures or lots of data.
# This may not always be possible but there are methods that may be used to make a decent model with little data compared to the millions
# This works by using a data genrator object that splits lets say one image into many different images
# Lets say it may stretch rotate flip or even do things to the images so as to obtain different images.
# The process of altering these images as explained above to obtain different images is called Augmentation
# Doing this gives the model lots of data to work with and gives it a general view of the features.
# An example of this is:

# Creating data generator object that transforms images
datagen = ImageDataGenerator(
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest',
)

# Picking an image to transform
test_img = train_images[20]
img = np.array(test_img)  # convert img to numpy array
img = img.reshape((1,) + img.shape)  # reshaping image

i = 0

# For all the images
for batch in datagen.flow(img, save_prefix='test', save_format='jpeg'):
    plt.figure(i)
    plt.imshow(np.array(batch[0]))
    i = i + 1
    if i > 4:
        break
plt.show()

# Loading the cats vs dogs dataset
tfds.disable_progress_bar()
(train_raw, validation_raw, test_raw), metadata = tfds.load(
    'cats_vs_dogs',
    split=['train[:80%]', 'train[80%:90%]', 'train[90%:]'],
    with_info=True,
    as_supervised=True
)

# Creating function to get labels
get_label_name = metadata.features['label'].int2str

# Displaying 2 images from the dataset
for image, label in train_raw.take(2):
    plt.figure()
    plt.imshow(image)
    plt.title(get_label_name(label))
    plt.show()

# Rescaling the images
IMG_SIZE = 160
def format_example(image, label):
    image = tf.cast(image, tf.float32)
    image = (image / 127.5) - 1
    image = tf.image.resize(image, (IMG_SIZE, IMG_SIZE))
    return image, label

train = train_raw.map(format_example)
validation = validation_raw.map(format_example)
test = test_raw.map(format_example)

# Picking the pre-trained model
IMG_SHAPE = (IMG_SIZE, IMG_SIZE, 3)
base_model = tf.keras.applications.MobileNetV2(
    input_shape=IMG_SHAPE,
    include_top=False,
    weights='imagenet'
)
base_model.summary()

# Freezing the imported model
base_model.trainable = False

# Adding layers to the model
global_average_layer = tf.keras.layers.GlobalAveragePooling2D()
prediction_layer = keras.layers.Dense(1)

# Combining all layers together
model = tf.keras.Sequential([
    base_model,
    global_average_layer,
    prediction_layer
])
model.summary()

# Compiling and training the model
base_learning_rate = 0.0001
initial_epochs = 3
model.compile(optimizer=tf.keras.optimizers.RMSprop(lr=base_learning_rate),
              loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
              metrics=['accuracy'])
history = model.fit(train, epochs=initial_epochs, validation_data=validation)
acc = history.history['accuracy']
print(acc)
