from util import read_images, write_data

if __name__ == "__main__":
	image_labels = ["elephant", "flamingo", "kangaroo", "Leopards", "octopus", "sea_horse"]
	# use first 20 images as training images
	training = read_images(image_labels, 0, 20)
	# save training histogram values to 'histograms.json' in JSON format
	write_data("histograms.json", training)
