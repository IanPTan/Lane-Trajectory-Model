from transformers import pipeline

generator = pipeline("mask-generation", model="facebook/sam2.1-hiera-tiny", device=0)
image_url = "truck.jpg"
outputs = generator(image_url, points_per_batch=64)

len(outputs["masks"])  # Number of masks generated

