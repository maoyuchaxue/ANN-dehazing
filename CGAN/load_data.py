import numpy as np
import pandas as pd

def load_data_mini():
	print("reading data...")
	origin = pandas.read_csv("../data/original_images_mini.csv")
	origin = np.array(origin).reshape([-1, 480, 640, 3])
	return origin	
