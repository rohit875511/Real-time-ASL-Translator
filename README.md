This repo contains files allowing you to use, create data for, and retrain a real-time American Sign Language translator.
How to Use:

HomeDeviceApp.py - File which contains the code to run when you want to use the ASL Translator.

Logger - File which allows you to add new data. To add new data run the file, enter what class number you are adding for (0-25 are currently taken by letters
A-Z), press enter, then hold up the hand-sign you want to be classify. Press space bar to add the data the keypoint.csv file.

keypoint_classifier_label.csv - The line number-1 corrosponds to the class it is represented by in keypoint.csv. The text on the line is what the sign will 
read as when classified.

Training - When you finish adding data and addding the label, save both files, then run this file. This file retrains the model on the new data and contains
the models architecture. 
