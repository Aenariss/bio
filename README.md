# BIO 2024/2025 - Extrakce krevního řečiště prstu za pomoci deterministických algoritmů

### Dependencies
To install the dependecies, you can use ``pip``. The requirements are listed in the file requirements.txt and can be installed with ``pip install ./requirements.txt``.

Other way is to use ``make build`` which runs the command for you.



### Usage:
```
python3 ./main.py -ip  <path/to/image> -- Visualizes the intermediate results of the processing pipeline  
python3 ./main.py -ip <path/to/image> -cw <path/to/image> -- Compares 2 images and determines whether they belong to the same person
python3 ./main.py -ip <path/to/image> -ca -- Compares given image to all in the database and returns the GAR/FAR/whatever > TODO: Add these statistics to result and MOVE this functionality into the test.py

OR

make run -ip ... (TODO: determine how hard it is to add argument support to makefile)
```

After running the file, you can use ``make clean`` to remove \_\_pycache\_\_ directories

### Tests:
Our Maximum curvature implementation was compared with the [BOB](https://www.idiap.ch/software/bob/docs/bob/bob.bio.vein/master/sphinx/index.html) reference library. The test shows results we receive from our implementation and the results of the BOB library. User can then visually compare them.

Testing can be run as:
```
python3 ./test.py -ip <path/to/image> 

OR

make test <path/to/image> 
```