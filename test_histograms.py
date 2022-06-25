from util import read_data, read_images, distances, write_data

if __name__ == "__main__":
	image_labels = ["elephant", "flamingo", "kangaroo", "Leopards", "octopus", "sea_horse"]
	# use the next 10 images after the first 20 images as test images
	test = read_images(image_labels, 20, 30)
	# read training images' histogram data from the 'histograms.json' JSON file
	training = read_data("histograms.json")
	# dictionary to keep track of the correct guesses for every image label for 'rgb' and 'h' histograms
	guesses = { i: { "rgb": 0, "h": 0 } for i in image_labels }

	images = { f"{label}/{name}": { "rgb": [], "h": [] } for label in test for name in test[label] }

	# go through every test image
	for label in test:
		for name in test[label]:
			print(f"--> {label}/{name}")
			print("------> rgb")
			# calculate the distances of the test image to the training data set for 'rgb' values
			rgb = distances(test[label][name], training, "rgb")
			# if one of the 5 guesses include the label of the test image, increase the correct guesses
			if label in map(lambda x: x["label"], rgb):
				guesses[label]["rgb"] += 1
			for i in rgb:
				print(f"----------> {i['label']}/{i['name']} {i['difference']}")
				images[f"{label}/{name}"]["rgb"].append(f"{i['label']}/{i['name']}")
			print("------> h")
			# calculate the distances of the test image to the training data set for 'h' values
			h = distances(test[label][name], training, "h")
			# if one of the 5 guesses include the label of the test image, increase the correct guesses
			if label in map(lambda x: x["label"], h):
				guesses[label]["h"] += 1
			for i in h:
				print(f"----------> {i['label']}/{i['name']} {i['difference']}")
				images[f"{label}/{name}"]["h"].append(f"{i['label']}/{i['name']}")

	write_data("images.json", images)

	for i in guesses:
		print(f"{i}: rgb: {guesses[i]['rgb'] * 10}%, h: {guesses[i]['h'] * 10}%")
	# calculate the percentages for all the labels combined
	rgb, h = 0, 0
	for i in guesses.values():
		rgb += i["rgb"]
		h += i["h"]
	print("total: rgb: {:.2f}%, h: {:.2f}%".format(rgb / 60 * 100, h / 60 * 100))