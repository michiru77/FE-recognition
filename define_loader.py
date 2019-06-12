#_ import
from keras.preprocessing.image import load_img, img_to_array
import cv2
import numpy as np
import csv

#_ define Label loader
class Label_loader:
    def __init__(self, arg_path):
        self.path = arg_path
        self.labels = []
        
    def get_labels(self):
        with open(self.path, "r") as f:
            reader = csv.reader(f)
            for row in reader:
                self.labels.append(row[1])

#_ define Image loader
class Image_loader:
    def __init__(self, arg_paths, arg_size):
        self.paths = arg_paths
        self.size = arg_size
        self.PILs = []
        self.grayscales = []
        self.colors = []

    def set_PILs(self, arg_list):
        self.PILs = arg_list
    
    def get_PILs(self):
        self.PILs = []
        for path in self.paths:
            get_PIL = load_img(path, target_size=(self.size, self.size))
            self.PILs.append(get_PIL)
    
    def get_grayscales(self):
        self.grayscales = []
        for PIL in self.PILs:
            grayscale = img_to_array(PIL.convert("L"))/255.
            self.grayscales.append(grayscale)
        self.grayscales = np.array(self.grayscales)

    def get_colors(self):
        self.colors = []
        for PIL in self.PILs:
            color = img_to_array(PIL)
            color = cv2.cvtColor(color, cv2.COLOR_BGR2RGB)
            self.colors.append(color)
        self.colors = np.array(self.colors)