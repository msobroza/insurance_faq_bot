"""Config file"""
import os

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
"""This file contains configurations of ML models and data"""
# Relative path that contains the FAQ data
FAQ_DATA_PATH = os.path.join(ROOT_DIR, "./data/FAQ_maif.csv")
# Relative path that contains the model path
MODEL_PATH = os.path.join(ROOT_DIR, "./models/fine_tuning_negative_sampling")
# Yaml of selected questions of FAQ
FAQ_SELECTION_PATH = os.path.join(ROOT_DIR, "./data/selection.yaml")
