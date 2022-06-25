import numpy as np
import os
import json
import math
from PIL import Image

def round(x):
	"""
	x : float
		the value to be rounded

	returns : int
		rounded value of x
	"""
	return int(x + math.copysign(0.5, x))

def create_histogram(image_matrix, max_val):
	"""
	image_matrix : 2-d array
		a matrix of an image consisting of only 1 channel, [[ r ]]
	max_val : int
		maximum value that the image_matrix could have

	returns : array
		a normalized histogram array
	"""
	# create an array with length max_val initialized to 0
	histogram = [0] * max_val
	height, width = image_matrix.shape
	count = height * width

	for row in image_matrix:
		for val in row:
			if val < 0 or val > max_val - 1:
				raise Exception(f"Histogram value must be between 0 and {max_val}. Received {val}.")
			# increment the histogram value of the color value
			histogram[int(val)] += 1

	# normalize the histogram by dividing the values by the amount of pixels and return as a list
	return list(map(lambda x: x / count, histogram))

def rgb_to_hsv(rgb):
	""" 
	rgb : array
		array consisting of RGB values, [ r, g, b ]

	returns : array
		array consisting of HSV values, [ h, s, v ]
	"""
	r = rgb[0] / 255.
	g = rgb[1] / 255.
	b = rgb[2] / 255.
	cmax = max(r, g, b)
	cmin = min(r, g, b)
	delta = cmax - cmin
	h = 60

	if delta == 0:
		h = 0
	elif cmax == r:
		h *= ((g - b) / delta) % 6
	elif cmax == g:
		h *= (b - r) / delta + 2
	elif cmax == b:
		h *= (r - g) / delta + 4

	return [round(h), delta / cmax if cmax != 0 else 0, cmax]

def rgb_to_hsv_image(image):
	"""
	image : 3-d array
		a matrix of an image consisting of RGB channels, [[ [r, g, b] ]]

	returns : 3-d array
		a matrix of an image consisting of HSV channels, [[ [h, s, v] ]]
	"""
	return np.array([[rgb_to_hsv(rgb) for rgb in row] for row in image])

def read_images(image_labels, start, end):
	"""
	read training images stored as './img/<label>/<name>'
	calculate RGB and HUE histograms
	save the histograms to a dictionary

	image_labels : str
		label names located in the './img/' folder
	start : int
		the start index of the training images
	end : int
		the end index of the training images
	
	returns : dict
		contains the histograms of the images like:
		{
			'label': {
				'name': {
					'r': [ red_histogram ], 'g': [ green_histogram ], 'b': [ blue_histogram ], 'h': [ hue_histogram ]
				},
				...
			},
			...
		}
	"""
	data = {}

	for label in image_labels:
		data[label] = {}
		path = f"./img/{label}"
		# list images in the label folder
		image_names = os.listdir(path)[start:end]
		for name in image_names:
			data[label][name] = {}
			# read image as RGB
			image = Image.open(f"{path}/{name}").convert("RGB")
			# convert image to RGB array
			image_rgb = np.asarray(image)
			# calculate r-g-b-h histograms ([...,x]==[:,:,x] selects the x'th channel, [[ [r, g, b] ]] [...,0] -> [[ r ]])
			histogram_r = create_histogram(image_rgb[...,0], 256)
			histogram_g = create_histogram(image_rgb[...,1], 256)
			histogram_b = create_histogram(image_rgb[...,2], 256)
			histogram_h = create_histogram(rgb_to_hsv_image(image_rgb)[...,0], 361)
			data[label][name]["r"] = histogram_r
			data[label][name]["g"] = histogram_g
			data[label][name]["b"] = histogram_b
			data[label][name]["h"] = histogram_h
			print(f"{path}/{name}")

	return data

def euclidean(hist1, hist2):
	"""
	find end return the euclidean distance between 2 histograms

	hist1, hist2 : array
		1-d array of an image histogram

	returns : int
		euclidean distance of the given histograms
	"""
	len1, len2 = len(hist1), len(hist2)
	if len1 != len2:
		raise Exception(f"Histogram lengths must be equal. Received {len1} and {len2}.")

	result = 0
	for val1, val2 in zip(hist1, hist2):
		result += (val1 - val2) ** 2

	return math.sqrt(result)

def distances(image, training, mode):
	"""
	calculate the euclidean distances of the given image
	compared to the training data set

	image : dict
		a dictionary containing the r-g-b-h histogram values like:
		{ 'r': [ red_histogram ], 'g': [ green_histogram ], 'b': [ blue_histogram ], 'h': [ hue_histogram ] }
	training : dict
		contains the histograms of the training images like:
		{
			'label': {
				'name': {
					'r': [ red_histogram ], 'g': [ green_histogram ], 'b': [ blue_histogram ], 'h': [ hue_histogram ]
				},
				...
			},
			...
		}
	mode : 'rgb', 'h'
		defines which histograms are going to be included in the difference calculation
		'rgb' -> euclidean(r) + euclidean(g) + euclidean(b)
		'h' -> euclidean(h)

	returns : array
		contains the top 5 training images' informations with the least differences to the original image
	"""

	if mode not in ["rgb", "h"]:
		raise Exception(f"Mode must be one of 'rgb' or 'h'. Received {mode}.")

	dist = []
	for label in training:
		for name in training[label]:
			# calculate the sum of the euclidean distances for the histograms given as the characters of the mode parameter
			dist.append({
				"label": label,
				"name": name,
				"difference": sum([euclidean(image[i], training[label][name][i]) for i in mode])
			})

	# sort the distances array based on the difference values and take the first 5 elements
	return sorted(dist, key=lambda x: x["difference"])[:5]

def write_data(name, data):
	"""
	write given data to a file with the given name
	in JSON format

	name : str
		file name
	data : dict
		data to be written in JSON format
	"""
	with open(name, "w") as f:
		json.dump(data, f)

def read_data(name):
	"""
	read data from a file with the given name
	in JSON format

	name : str
		file name

	returns : dict
		the JSON data read from the file interpreted as a Python dictionary
	"""
	with open(name, "r") as f:
		return json.load(f)
