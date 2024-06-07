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

def predict_keras(model, testset):
    testset_predictions = model.predict(testset)
    testset_predictions = np.argmax(testset_predictions, axis = 1)
    accurracy= (np.sum(testset_predictions == testset.classes))/len(testset_predictions)
    print(f"Accuracy: {accurracy:.4f}")
    return testset_predictions

class InvarianceCheckerKeras:
    def __init__(self) -> None:
        pass
    
    def transform_test_set(self, image_size: tuple,  path: str, batch_size: int = 64):
        rotated_transform = ImageDataGenerator(rescale=1./255,
                                        preprocessing_function=custom_preprocessing_function
                                        )
        normal_transform = ImageDataGenerator(rescale=1./255)
        
        rotated_testset =  rotated_transform.flow_from_directory(
                                                    path,
                                                    target_size=image_size[:2],
                                                    batch_size=batch_size,
                                                    class_mode='categorical',
                                                    color_mode='rgb',
                                                    shuffle = False
        )
        
        normal_testset =  normal_transform.flow_from_directory(
                                                    path,
                                                    target_size=image_size[:2],
                                                    batch_size=batch_size,
                                                    class_mode='categorical',
                                                    color_mode='rgb',
                                                    shuffle= False
        )
        
        
        return rotated_testset, normal_testset
    
    def predict_models(self, model, rotated_testset, normal_testset):
        print(f"Rotated Testset:")
        print("-"* 50)
        rotated_predictions = predict_keras(model, rotated_testset)
        print("-"* 50)
        print("-"* 50)
        print(f"Normal Testset:")
        print("-"* 50)
        normal_predictions = predict_keras(model, normal_testset)
        
        return rotated_predictions, normal_predictions
    
    def calculate_invariance(self, rotated_predictions, normal_predictions):
        return np.sum(rotated_predictions == normal_predictions)/len(rotated_predictions)
    
    


