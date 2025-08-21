#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np
import tensorflow as tf

class KeyPointClassifier(object):
    def __init__(
        self,
        model_path='model/keypoint_classifier/asl_classifier.hdf5',
    ):
        self.model = tf.keras.models.load_model(model_path)

    def __call__(
        self,
        landmark_list,
    ):
        # Make prediction using the full TensorFlow model
        result = self.model.predict(np.array([landmark_list], dtype=np.float32), verbose=0)
        result_index = np.argmax(np.squeeze(result))
        
        return result_index