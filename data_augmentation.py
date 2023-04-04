# Importing important libraries
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
