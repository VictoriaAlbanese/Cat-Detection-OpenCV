# Cat-vs-Dog-Classifier
This repository contains our Computer Vision final project where we classify images of cats and dogs utilizing Tensorflow and Scikit-Image.
Project created by Victoria Albanese and Hannah Chiodo (2019)
-------------------

Dataset Source: https://www.kaggle.com/c/dogs-vs-cats/data

Dataset directory structure is "{project location}\dataset\test_data" and "{project location}\dataset\training_data"

Python version: 3.6.4

Required packages:
    -keras
    -sklearn
    -skimage

Code execution:
    1) Split dataset manually: test_data should contain cats and dogs 0-3124. The rest go in training_data
    2) Creating the models: run nn_functions.py for the desired extractor (i.e. EXTRACTOR = fe.HOG_extractor). For the HOG, set         PIXELS_PER_CELL to the desired value, i.e. "4x4"
    3) Testing the models: run test_nn.py for the desired model. The model type is the EXTRACTOR value set in nn_functions.py

We used this tutorial as a jumping off point for our neural net and data processing: https://www.pyimagesearch.com/2016/09/26/a-simple-neural-network-with-python-and-keras/

```
   ,';,               ,';,
 ,' , :;             ; ,,.;
 | |:; :;           ; ;:|.|
 | |::; ';,,,,,,,,,'  ;:|.|    ,,,;;;;;;;;,,,
 ; |''  ___      ___   ';.;,,''             ''';,,,
 ',:   /   \    /   \    .;.                      '';,
 ;    /    |    |    \     ;,                        ';,
;    |    /|    |\    |    :|                          ';,
|    |    \|    |/    |    :|     ,,,,,,,               ';,
|     \____| __ |____/     :;  ,''                        ;,
;           /  \          :; ,'                           :;
 ',        `----'        :; |'                            :|
   ',,  `----------'  ..;',|'                             :|
  ,'  ',,,,,,,,,,,;;;;''  |'                              :;
,'  ,,,,                  |,                              :;
| ,'   :;, ,,''''''''''   '|.   ...........                ';,
;       :;|               ,,';;;''''''                      ';,
 ',,,,,;;;|.............,'                          ....      ;,
           ''''''''''''|        .............;;;;;;;''''',    ':;
                       |;;;;;;;;'''''''''''''             ;    :|
                                                      ,,,'     :;
                                          ,,,,,,,,,,''       .;'
                                         |              .;;;;'
                                         ';;;;;;;;;;;;;;'
```