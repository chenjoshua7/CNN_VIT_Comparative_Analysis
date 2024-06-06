import numpy as np
import tqdm as tqdm

from keras.preprocessing.image import ImageDataGenerator

def custom_preprocessing_function(image):
    rotations = [90, 180, 270]
    rotation = np.random.choice(rotations)
    
    if rotation == 90:
        image = np.rot90(image, k=1)
    elif rotation == 180:
        image = np.rot90(image, k=2)
    elif rotation == 270:
        image = np.rot90(image, k=3)
    
    return image

class InvarianceCheckerKeras:
    def __init__(self) -> None:
        self.augment_transform = ImageDataGenerator(rescale=1./255,
                                        preprocessing_function=custom_preprocessing_function
                                        )
        self.normal_transform = ImageDataGenerator(rescale=1./255)
    
    def transform_test_set(self, image_size: tuple,  path: str, batch_size: int = 64):
        augment_testset =  self.augment_transform.flow_from_directory(
                                                    path,
                                                    target_size=image_size[:2],
                                                    batch_size=batch_size,
                                                    class_mode='categorical',
                                                    color_mode='rgb',
                                                    shuffle = False
        )
        
        normal_testset =  self.normal_transform.flow_from_directory(
                                                    path,
                                                    target_size=image_size[:2],
                                                    batch_size=batch_size,
                                                    class_mode='categorical',
                                                    color_mode='rgb',
                                                    shuffle= False
        )
        
        
        return augment_testset, normal_testset
    
    def predict_models(self, model, augment_testset, normal_testset):
        augment_predictions = model.predict(augment_testset)
        normal_predictions = model.predict(normal_testset)
        
        return augment_predictions, normal_predictions
    
    def calculate_invariance(augment_predictions, normal_predictions):
        return (augment_predictions == normal_predictions)/len(augment_predictions)
    
    


