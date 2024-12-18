# Card Recognition Using OpenCV and CNN
## Description
This program is a card recognition with CNN method. Using Tensorflow and OpenCV. The accuracy of the model that has been created reaches 98% and can detect cards in a standard deck well in various conditions.

## How To Use
First you have to run train model.py to train the dataset with CNN and generate the h.5 model, after that you can run the Test model.py program to open the open cv window and start reading.

## Dataset
So I used a dataset from Kaggle with the following link :

https://www.kaggle.com/datasets/gpiosenka/cards-image-datasetclassification

## Feature
1. Can detect all four corners of the card
2. Can straighten cards
3. The reading is done when the cards are straightened, so that whatever the position of the cards, it can still be read.

## Things to note
1. Prone to reflections, so make sure the card is completely visible without any light reflections.

For further implementation, I already makes game with this method, you can check here : 

https://github.com/zhafarullah/GinRummyGame
