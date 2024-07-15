# AI-projects
!pip install torch torchvision transformers diffusers pillow
import torch
from diffusers import StableDiffusionPipeline

def generate_image(prompt, num_inference_steps=400, guidance_scale=7.5, output_path="generated_image.png"):
    # Load the Stable Diffusion model
    model_id = "CompVis/stable-diffusion-v1-4"
    pipe = StableDiffusionPipeline.from_pretrained(model_id)
    pipe = pipe.to("cuda")

    # Generate the image
    with torch.no_grad():
        image = pipe(prompt, num_inference_steps=num_inference_steps,strength=0.8, guidance_scale=guidance_scale).images[0]

    # Save the generated image
    image.save(output_path)
    print(f"Image saved to {output_path}")

# Example usage
prompt = "Create a stunning and fashionable saree worn by a woman, showcasing intricate patterns and vibrant colors. The saree should feature a blend of traditional and modern elements, with floral and geometric designs that complement each other. The color palette should include rich shades like deep blue, bright red, and gold accents, exuding elegance and style. The scene should capture the grace of the woman as she drapes the saree beautifully"
generate_image(prompt)
!pip install accelerate

import torch
from PIL import Image
import numpy as np
from diffusers import StableDiffusionImg2ImgPipeline

def preprocess_image(image_path, target_size=(512, 512)):
    image = Image.open(image_path).convert("RGB")
    image = image.resize(target_size)
    # Return a PIL Image instead of a tensor
    return image

def generate_image_from_image(input_image_path, prompt, strength=0.8, num_inference_steps=500, guidance_scale=7.5, output_path="generated_image.png"):
    # Load the Stable Diffusion model
    model_id = "CompVis/stable-diffusion-v1-4"
    pipe = StableDiffusionImg2ImgPipeline.from_pretrained(model_id)
    pipe = pipe.to("cuda")

    # Preprocess the input image
    init_image = preprocess_image(input_image_path) # Pass the PIL Image directly

    # Generate the image
    with torch.no_grad():
        images = pipe(prompt=prompt, image=init_image, strength=strength, num_inference_steps=num_inference_steps, guidance_scale=guidance_scale).images # Use 'image' instead of 'init_image'

    # Save the generated image
    generated_image = images[0]
    # Convert to numpy array and scale before creating PIL Image
    generated_image = Image.fromarray((np.array(generated_image) * 255).astype(np.uint8))
    generated_image.save(output_path)
    print(f"Image saved to {output_path}")

# Example usage
input_image_path = "input_image.png"  # Path to your input image
prompt = "Create a color full Fashion Design cloths on that model"
generate_image_from_image(input_image_path, prompt)

!pip install flask-ngrok torch torchvision transformers diffusers pillow numpy
%mkdir templates

%%writefile templates/index.html
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Stable Diffusion Image Generator</title>
</head>
<body>
    <h1>Stable Diffusion Image Generator</h1>
    <form action="/" method="post" enctype="multipart/form-data">
    <label for="image">Select image (optional):</label>
    <input type="file" id="image" name="image" accept="image/*">
    <br>
    <label for="prompt">Enter prompt:</label>
    <input type="text" id="prompt" name="prompt" required>
    <br>
    <button type="submit">Generate Image</button>
</form>

{% if output_image %}
    {% if input_image %}
        <h2>Original Image:</h2>
        <img src="{{ url_for('static', filename=input_image.split('/')[-1]) }}" alt="Input Image" style="max-width: 100%;">
    {% endif %}
    <h2>Generated Image:</h2>
    <img src="{{ url_for('static', filename=output_image.split('/')[-1]) }}" alt="Generated Image" style="max-width: 100%;">
    <p>Prompt: {{ prompt }}</p>
{% endif %}
</body>
</html>

%mkdir static

!pip install pyngrok

# Step 1: Install required packages
# Uncomment and run this line in a Colab cell
# !pip install flask flask-ngrok torch torchvision transformers diffusers pillow numpy

# Step 2: Import necessary libraries
import os
from flask import Flask, request, render_template
from pyngrok import ngrok  # Use pyngrok for ngrok integration
from PIL import Image
import torch
import numpy as np
from diffusers import StableDiffusionImg2ImgPipeline

# Step 3: Initialize Flask app
app = Flask(__name__)
port = "5000"

# Step 4: Open a ngrok tunnel to the HTTP server
public_url = ngrok.connect(port).public_url
print(f" * ngrok tunnel \"{public_url}\" -> \"http://127.0.0.1:{port}\"")

# Step 5: Define image preprocessing function
def preprocess_image(image_path, target_size=(512, 512)):
    image = Image.open(image_path).convert("RGB")
    image = image.resize(target_size)
    return image

# Step 6: Define the image generation function
def generate_image_from_image(input_image_path=None, prompt=None, strength=0.8, num_inference_steps=300, guidance_scale=7.5, output_path="/content/static/generated_image.png"):
    model_id = "CompVis/stable-diffusion-v1-4"
    pipe = StableDiffusionImg2ImgPipeline.from_pretrained(model_id)
    pipe = pipe.to("cuda")  # Make sure to use GPU

    if input_image_path:
        init_image = preprocess_image(input_image_path)
    else:
        init_image = Image.new('RGB', (512, 512), color=(255, 255, 255))  # White image

    with torch.no_grad():
        images = pipe(prompt=prompt, image=init_image, strength=strength, num_inference_steps=num_inference_steps, guidance_scale=guidance_scale).images

    generated_image = images[0]
    generated_image.save(output_path)
    return output_path

# Step 7: Define Flask routes
@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        prompt = request.form["prompt"]

        input_image_path = None
        if "image" in request.files and request.files["image"].filename != "":
            image_file = request.files["image"]
            input_image_path = os.path.join("static", image_file.filename)
            image_file.save(input_image_path)

        output_image_path = generate_image_from_image(input_image_path, prompt)
        print(f"Image saved to {output_image_path}")
        return render_template("index.html", input_image=input_image_path, output_image=output_image_path, prompt=prompt)

    return render_template("index.html")

# Step 8: Run the Flask app (no threading)
if __name__ == "__main__":
    app.run(port=int(port))


import getpass

from pyngrok import ngrok, conf

print("Enter your authtoken, which can be copied from https://dashboard.ngrok.com/auth")
conf.get_default().auth_token = getpass.getpass()

# Open a TCP ngrok tunnel to the SSH server
connection_string = ngrok.connect("22", "tcp").public_url

ssh_url, port = connection_string.strip("tcp://").split(":")
print(f" * ngrok tunnel available, access with `ssh root@{ssh_url} -p{port}`")

