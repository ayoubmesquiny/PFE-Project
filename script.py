import os
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential, Model
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, Input
from keras.utils import plot_model
from keras.applications import InceptionV3, VGG16, VGG19, EfficientNetB7, ResNet50, DenseNet121, Xception, InceptionResNetV2
from keras.callbacks import EarlyStopping
from keras.optimizers import SGD, Adam, RMSprop
from keras.regularizers import l2


def load_base_model_InceptionResNetV2():
    # Load the pre-trained model
    base_model = InceptionResNetV2(
        weights='imagenet', # Load pre-trained weights from the ImageNet dataset
        include_top=False, # Set to False to exclude the top fully connected layers from the model
        input_shape=(img_width, img_height, 3) # The input shape of the images
    )

    # Freeze the weights of the pre-trained layers
    for layer in base_model.layers:
        layer.trainable = False

    return base_model

def load_base_model_Xception():
    # Load the pre-trained model
    base_model = Xception(
        weights='imagenet', # Load pre-trained weights from the ImageNet dataset
        include_top=False, # Set to False to exclude the top fully connected layers from the model
        input_shape=(img_width, img_height, 3) # The input shape of the images
    )

    # Freeze the weights of the pre-trained layers
    for layer in base_model.layers:
        layer.trainable = False

    return base_model

def load_base_model_DenseNet121():
    # Load the pre-trained model
    base_model = DenseNet121(
        weights='imagenet', # Load pre-trained weights from the ImageNet dataset
        include_top=False, # Set to False to exclude the top fully connected layers from the model
        input_shape=(img_width, img_height, 3) # The input shape of the images
    )

    # Freeze the weights of the pre-trained layers
    for layer in base_model.layers:
        layer.trainable = False

    return base_model

def load_base_model_InceptionV3():
    # Load the pre-trained model
    base_model = InceptionV3(
        weights='imagenet', # Load pre-trained weights from the ImageNet dataset
        include_top=False, # Set to False to exclude the top fully connected layers from the model
        input_shape=(img_width, img_height, 3) # The input shape of the images
    )

    # Freeze the weights of the pre-trained layers
    for layer in base_model.layers:
        layer.trainable = False

    return base_model

def load_base_model_ResNet50():
    # Load the pre-trained model
    base_model = ResNet50(
        weights='imagenet', # Load pre-trained weights from the ImageNet dataset
        include_top=False, # Set to False to exclude the top fully connected layers from the model
        input_shape=(img_width, img_height, 3) # The input shape of the images
    )

    # Freeze the weights of the pre-trained layers
    for layer in base_model.layers:
        layer.trainable = False

    return base_model

def load_base_model_VGG16():
    # Load the pre-trained model
    base_model = VGG16(
        weights='imagenet', # Load pre-trained weights from the ImageNet dataset
        include_top=False, # Set to False to exclude the top fully connected layers from the model
        input_shape=(img_width, img_height, 3) # The input shape of the images
    )

    # Freeze the weights of the pre-trained layers
    for layer in base_model.layers:
        layer.trainable = False

    return base_model

def load_base_model_VGG19():
    # Load the pre-trained model
    base_model = VGG19(
        weights='imagenet', # Load pre-trained weights from the ImageNet dataset
        include_top=False, # Set to False to exclude the top fully connected layers from the model
        input_shape=(img_width, img_height, 3) # The input shape of the images
    )

    # Freeze the weights of the pre-trained layers
    for layer in base_model.layers:
        layer.trainable = False

    return base_model

def load_base_model_EfficientNetB7():
    # Load the pre-trained model
    base_model = EfficientNetB7(
        weights='imagenet', # Load pre-trained weights from the ImageNet dataset
        include_top=False, # Set to False to exclude the top fully connected layers from the model
        input_shape=(img_width, img_height, 3) # The input shape of the images
    )

    # Freeze the weights of the pre-trained layers
    for layer in base_model.layers:
        layer.trainable = False

    return base_model

def build_model(base_model):
    # Build a new model on top of the pre-trained model
    model = Sequential()
    model.add(base_model)

    # Flatten the output of the base model
    model.add(Flatten())

    # Add dense layers with dropout and L2 regularization
    model.add(Dense(512, activation='relu', kernel_regularizer=l2(0.01)))
    model.add(Dense(256, activation='relu', kernel_regularizer=l2(0.01)))
    model.add(Dense(128, activation='relu', kernel_regularizer=l2(0.01)))
    model.add(Dense(64, activation='relu', kernel_regularizer=l2(0.01)))
    model.add(Dense(1, activation='sigmoid'))

    return model

def save_model_summary(model, filename):

    with open(filename, 'w') as f:
        # Redirect the output of print() to the file
        print_func = lambda x: print(x, file=f)
        model.summary(print_fn=print_func)

def plot_model_architecture(model):
    # Plot the model architecture
    plot_model(model, to_file='B07_model.png', show_shapes=True, show_layer_names=True)

def compile_model(model):
    # Compile the model
    optimizer = SGD(learning_rate=1e-4, momentum=0.9)
    model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])

    return model

def create_train_datagen(img_width, img_height, batch_size):
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=20,
        width_shift_range=0.1,
        height_shift_range=0.1,
        brightness_range=(0.8, 1.2),
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest'
    )

    train_data = train_datagen.flow_from_directory(
        train_dir,
        target_size=(img_width, img_height),
        batch_size=batch_size,
        class_mode="binary"
    )

    return train_data

def create_test_datagen(img_width, img_height, batch_size):
    test_datagen = ImageDataGenerator(rescale=1./255)

    test_data = test_datagen.flow_from_directory(
        test_dir,
        target_size=(img_width, img_height),
        batch_size=batch_size,
        class_mode="binary",
        shuffle=False
    )

    return test_data

def load_data(img_width, img_height, batch_size):
    train_data = create_train_datagen(img_width, img_height, batch_size)
    test_data = create_test_datagen(img_width, img_height, batch_size)

    return train_data, test_data

def train_model(model, train_data, test_data, epochs, early_stop_patience):
    # Define early stopping callback
    early_stop_loss = EarlyStopping(monitor='val_loss', patience=early_stop_patience)
    early_stop_acc = EarlyStopping(monitor='val_accuracy', patience=early_stop_patience)

    # Train the model
    history = model.fit(
        train_data, 
        epochs=epochs, 
        steps_per_epoch=train_data.n // train_data.batch_size,
        validation_data=test_data,
        validation_steps=test_data.n // test_data.batch_size,
        callbacks=[early_stop_loss, early_stop_acc]
    )

    return history

def plot_training_loss(history):
    # Plot the training and validation loss over the epochs
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')
    plt.title('Training and Validation Loss')

    # Save the plot as an image
    plt.savefig('loss_plot.png')

def plot_training_accuracy(history):
    # Plot the training and validation accuracy over the epochs
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('Model accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')
    plt.title('Training and Validation Accuracy')

    # Save the plot as an image
    plt.savefig('accuracy_plot.png')

def plot_confusion_matrix(y_true, y_pred_classes):
    # Compute the confusion matrix
    cm = confusion_matrix(y_true, y_pred_classes)

    # Plot the confusion matrix using seaborn
    sns.heatmap(cm, annot=True, cmap='Blues')

    # Set the axis labels
    plt.xlabel('Predicted')
    plt.ylabel('True')

    
    plt.savefig('confusion_matrix.png')




# Define the path to the data directory
data_dir = "C:/Users/DELL/Desktop/datatest/D-E"

# Define the paths to the train and test directories
train_dir = os.path.join(data_dir, "train")
test_dir = os.path.join(data_dir, "test")

# Define the input image size and number of classes
img_width = 224
img_height = 224
batch_size = 4


base_model = load_base_model_EfficientNetB7()
model = build_model(base_model)
save_model_summary(model,'model_summary.txt')
plot_model_architecture(model)
model = compile_model(model)


train_data, test_data =load_data(img_width, img_height, batch_size)
history = train_model(model, train_data, test_data, epochs=100, early_stop_patience=10)


# Predict the classes for the validation data
y_pred = model.predict(test_data)
y_pred_classes = np.argmax(y_pred, axis=1)
# Get the true classes for the validation data
y_true = test_data.classes
# Plot and save the confusion matrix
plot_confusion_matrix(y_true, y_pred_classes)


plot_training_loss(history)
plot_training_accuracy(history)
