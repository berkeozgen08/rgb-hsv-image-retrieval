import os
import json
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from PIL import Image

f = open("images.json", "r")
images = json.load(f)
f.close()

for original_name in images:
	os.makedirs(f"generated/{original_name.split('/')[0]}", exist_ok=True)
	original = Image.open(f"img/{original_name}")
	gs = GridSpec(2, 3)
	plt.suptitle(f"RGB")
	plt.subplot(gs[0, 0])
	plt.imshow(original)
	plt.title(f"ORIGINAL\n{original_name}", fontdict={ "fontsize": 8 })
	plt.axis("off")
	for index, (name, i, j) in enumerate(zip(images[original_name]["rgb"], [0, 0, 1, 1, 1], [1, 2, 0, 1, 2])):
		plt.subplot(gs[i, j])
		plt.imshow(Image.open(f"img/{name}"))
		plt.title(f"{index + 1}. IMAGE\n{name}", fontdict={ "fontsize": 8 })
		plt.axis("off")

	plt.tight_layout()
	plt.savefig(f"generated/{original_name.split('.jpg')[0]}_1.jpg")
	plt.close()
	
	gs = GridSpec(2, 3)
	plt.suptitle(f"HSV")
	plt.subplot(gs[0, 0])
	plt.imshow(original)
	plt.title(f"ORIGINAL\n{original_name}", fontdict={ "fontsize": 8 })
	plt.axis("off")
	for index, (name, i, j) in enumerate(zip(images[original_name]["h"], [0, 0, 1, 1, 1], [1, 2, 0, 1, 2])):
		plt.subplot(gs[i, j])
		plt.imshow(Image.open(f"img/{name}"))
		plt.title(f"{index + 1}. IMAGE\n{name}", fontdict={ "fontsize": 8 })
		plt.axis("off")
		
	plt.tight_layout()
	plt.savefig(f"generated/{original_name.split('.jpg')[0]}_2.jpg")
	plt.close()
