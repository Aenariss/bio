"""
BIO Project 2024

This module contains a class to load all the images from the dataset
Author: 
    Vojtech Fiala <xfiala61>
    ChatGPT
"""

import os

class DataLoader:
    def __init__(self, original_finger: str = None, dataset_path: str = "./data"):
        """
        Initializes the DataLoader with paths to the original image and dataset directory.
        
        Args:
            original_finger (str): File path to the original image to be used for comparison, 
                                   expected format is '\\number\\finger\\number.bmp'.
            dataset_path (str): Path to the root dataset directory, containing subfolders for each person and their finger images.
        """
        self.dataset_path = dataset_path
        self.original_finger = original_finger

    def get_original_image_data(self) -> tuple:
        """
        Extracts the metadata (person ID, finger type, and image ID) from the path of the original image.
        
        Returns:
            tuple: A tuple (origo_id_person, origo_finger, origo_id_finger) where:
                   - origo_id_person is the ID of the person to whom the original image belongs,
                   - origo_finger is the specific finger type,
                   - origo_id_finger is the identifier of the image file.
                   
        Raises:
            SystemExit: If the path format of the original image does not match expected conventions.
        """
        if self.original_finger:
            def get_data_from_path(path: str) -> tuple:
                # Split the path to get metadata, supporting both Windows and Linux formats
                origo_img = path.split('\\')
                
                # Check for Linux format if Windows format failed
                if len(origo_img) == 1:
                    origo_img = path.split('/')

                # Exit if neither format matches
                if len(origo_img) == 1:
                    exit("Could not parse the given path to the image for comparison!")

                # Extract IDs for person, finger, and image
                origo_id_finger = origo_img[-1]
                origo_finger = origo_img[-2]
                origo_id_person = origo_img[-3]
            
                return origo_id_person, origo_finger, origo_id_finger

        return get_data_from_path(self.original_finger)

    def load_images(self) -> dict:
        """
        Traverses the dataset directory, loading images for each person and finger.
        
        Returns:
            dict: A nested dictionary structure:
                  - Keys are person IDs, each containing a dictionary of:
                  - Finger types, each mapping to a list of image file paths.
                  
                  Example format:
                  {
                      'person1': {
                          'finger1': ['image1.bmp', 'image2.bmp', ...],
                          'finger2': ['image1.bmp', 'image2.bmp', ...],
                          ...
                      },
                      ...
                  }
        """
        data = {}
        
        # Traverse through each person folder in the dataset directory
        for person_id in os.listdir(self.dataset_path):
            person_path = os.path.join(self.dataset_path, person_id)

            # Check if the person path is a directory
            if os.path.isdir(person_path):
                data[person_id] = {}

                # Traverse each finger folder within the person's folder
                for finger in os.listdir(person_path):
                    finger_path = os.path.join(person_path, finger)
                    
                    # Ensure the finger path is also a directory
                    if os.path.isdir(finger_path):
                        images = []
                        
                        # Load all .bmp images in the finger folder
                        for image_file in os.listdir(finger_path):
                            if image_file.lower().endswith('.bmp'):
                                image_path = os.path.join(finger_path, image_file)
                                images.append(image_path)

                        # Store the images for the current finger in the data dictionary
                        data[person_id][finger] = images

        return data
