"""
BIO Project 2024
Author: 
    Filip Brna <xbrnaf00>
    Vojtech Fiala <xfiala61>
"""

from src.DataLoader import DataLoader

# Dataset Mentioned at https://www.researchgate.net/publication/308128095_A_Review_of_Finger-Vein_Biometrics_Identification_Approaches, found at https://huggingface.co/datasets/luyu0311/MMCBNU_6000
# We use only a part of it 

data = DataLoader().load_images()
print(data)
