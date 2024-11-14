# BIO 2024/2025 - Extrakce krevního řečiště prstu za pomoci deterministických algoritmů
Description of our implementation is in the file dokumentace.pdf.

### Folders
``./data/``      - real scanned fingers from our dataset
``./src/``       - source code
``./templates/`` - HTML files for frontend

### Dependencies
To install the dependecies, you can use ``pip``. The requirements are listed in the file requirements.txt and can be installed with ``pip install ./requirements.txt``.

### Usage:
```
python3 ./main.py -ip <path/to/image> -- Visualizes the intermediate results of the processing pipeline  
python3 ./main.py -ip <path/to/image> -cw <path/to/image> -- Compares 2 images and determines whether they belong to the same person
python3 ./main.py -ip <path/to/image> -ca -- Calculate FMR, FNMR, TMR, TNMR. Each 30th image is taken as new image the following images are compared to. Requires dataset to be present in ./data/Person_ID/Finger/Photo_ID.bmp. The link to the dataset is in the documentation.
python3 ./frontend.py -- Launches frontend on address 127.0.0.1:5000.
```

### Frontend
Frontend allows comparison of two images of the user's choice. 
The comparison returns the matching score, match result and thresholded it was compared against. Inside, it does the same as ``python3 ./main.py -ip <path/to/image> -cw <path/to/image>``

### Comparison with BoB:
Our Maximum curvature implementation was compared with the [BOB](https://www.idiap.ch/software/bob/docs/bob/bob.bio.vein/master/sphinx/index.html) reference library. The test shows results we receive from our implementation and the results of the BOB library. User can then visually compare them.

Testing can be run as:
``python3 ./test.py <path/to/image>``