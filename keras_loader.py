import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import splitfolders
from keras.preprocessing.image import ImageDataGenerator
import seaborn as sns
import tensorflow as tf


def custom_preprocessing_function(image):
    random_number = np.random.rand()
    if random_number < 0.25:
        image = tf.image.rot90(image, k=1)  # Rotate 90 degrees
    elif random_number < 0.5:
        image = tf.image.rot90(image, k=2)  # Rotate 180 degrees
    elif random_number < 0.75:
        image = tf.image.rot90(image, k=3)  # Rotate 270 degrees
    return image

class ImportData:
    def __init__(self, input_dir = "raw_data", output_dir = "data") -> None:
        self.input_dir = input_dir
        self.output_dir = output_dir
        self.training_set = None
    
    def train_test_split(self, ratio = (0.8, 0.1, 0.1), SEED = 1234):        
        splitfolders.ratio(self.input_dir, output=self.output_dir, seed=SEED, ratio=ratio)
        return
        
    def load_data(self, image_size = (64, 64, 3), augment = True, batch_size = 100):
        '''if augmentation_params is None:
            augmentation_params = {
                "zoom_range": 0,
                "horizontal_flip": False,
                "width_shift_range": 0,
                "height_shift_range": 0
            }'''
        train_dir = self.output_dir + "/train"
        val_dir = self.output_dir + "/val"
        test_dir = self.output_dir + "/test"

        image_size = image_size
        
        if augment:
            train_datagen = ImageDataGenerator(rescale=1./255,
                                        preprocessing_function=custom_preprocessing_function
                                        )
        else:
            train_datagen = ImageDataGenerator(rescale=1./255)
        
        test_datagen = ImageDataGenerator(rescale=1./255)
        
        training_set =  train_datagen.flow_from_directory(
                                                    train_dir,
                                                    target_size=image_size[:2],
                                                    batch_size=batch_size,
                                                    class_mode='categorical',
                                                    color_mode='rgb'

        )

        val_set =  test_datagen.flow_from_directory(
                                                    val_dir,
                                                    target_size=image_size[:2],
                                                    batch_size=batch_size,
                                                    class_mode='categorical',
                                                    color_mode='rgb',
                                                    shuffle=False

        )

        test_set =  test_datagen.flow_from_directory(
                                                    test_dir,
                                                    target_size=image_size[:2],
                                                    batch_size=batch_size,
                                                    class_mode='categorical',
                                                    color_mode='rgb',
                                                    shuffle=False
                                                    

        )
        
        self.training_set = training_set
            
        return training_set, val_set, test_set
    
    def plot_random(self):
        images, labels = next(self.training_set)

        class_indices = self.training_set.class_indices
        index_to_class = {v: k for k, v in class_indices.items()}

        select_images = np.random.randint(size = 3, low = 0, high = len(labels))

        fig, axs = plt.subplots( 1,3, figsize=(15,5))

        for i,v in enumerate(select_images):
            one_hot_label = labels[v]
            class_name = index_to_class[np.argmax(one_hot_label)]
            
            ax = axs[i]
            ax.imshow(images[v])
            ax.set_title(class_name)

        plt.axis("off")
        plt.tight_layout()
        plt.show()
        