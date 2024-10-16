from PIL import Image
import numpy as np

def lambda_handler(event, context):
	image_path = "img.jpg" 
	img = Image.open(image_path)

	# Convert image to a numpy array
	img_array = np.array(img)

	resized_img = img.resize((500, 500))
	resized_img_array = np.array(resized_img)

	# Rotate image
	for _ in range(10):
		rotated_img = resized_img.rotate(45)
		rotated_img_array = np.array(rotated_img)

	# Combine numpy arrays but with smaller arrays
	combined_img_array = np.concatenate((resized_img_array, rotated_img_array), axis=1)

	# Apply color manipulations with arrays
	img_gray = img.convert("L")  # Convert to grayscale
	gray_img_array = np.array(img_gray)

	# Create a repeated pattern
	# repeated_array = np.tile(gray_img_array, (10, 10))  

	return {
	'statusCode': 200
	}