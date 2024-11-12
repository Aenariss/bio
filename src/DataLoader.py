"""
BIO Project 2024

This module contains a class to load all the images from the dataset
Author: 
    Vojtech Fiala <xfiala61>
    ChatGPT
"""

import os

class DataLoader:
    """
    DataLoader class for crawling through a given folder and loading all images
    into a 3D array structure (person -> finger -> images).
    """
    
    def __init__(self, original_finger=None, dataset_path="./data"):
        """
        Initializes the DataLoader with the dataset path.
        
        Args:
            original_finger (str): Path to the original image which will be compared against other loaded images. Must be in format '\number\finger\number.bmp'
            dataset_path (str): The root path to the dataset containing all persons and their finger images.
        """
        self.dataset_path = dataset_path
        self.original_finger = original_finger

    def get_original_image_data(self):
        """
            Return data of who the orgiinal image belongs to
        """
        if self.original_finger:
            def get_data_from_path(path):
                # Windows
                origo_img = path.split('\\')

                if len(origo_img) == 1:
                    # Linux
                    origo_img = path.split('/')

                if len(origo_img) == 1:
                    exit("Could not parse the given path to the image to compare with!")
                    

                origo_id_finger = origo_img[-1]
                origo_finger = origo_img[-2]
                origo_id_person = origo_img[-3]
            
                return origo_id_person, origo_finger, origo_id_finger
        return get_data_from_path(self.original_finger)

    def load_images(self):
        """
        Crawls through the dataset folder and loads images for each person and finger.
        
        Returns:
            dict: A nested dictionary where the keys are person IDs, 
                  each containing another dictionary of fingers and their corresponding image arrays.
        """
        data = {}
            
        # Traverse the directory structure
        for person_id in os.listdir(self.dataset_path):
            person_path = os.path.join(self.dataset_path, person_id)
            if os.path.isdir(person_path):
                data[person_id] = {}

                for finger in os.listdir(person_path):
                    finger_path = os.path.join(person_path, finger)
                    if os.path.isdir(finger_path):
                        images = []
                        
                        # Load all image files in the finger folder
                        for image_file in os.listdir(finger_path):
                            if image_file.lower().endswith(('.bmp')):
                                image_path = os.path.join(finger_path, image_file)
                                images.append(image_path)

                        # Store the images for the current finger
                        data[person_id][finger] = images

        return data
