#! /usr/bin/env python3

import os
import replicate
import requests

def genImage(prompt: str, output_full_path: str) -> None:
    """
    Generate an image using the Flux model on Replicate.
    
    Args:
        prompt (str): The text prompt to generate the image from
        output_full_path (str): Full path where the generated image will be saved
    """
    # Read API key from file
    api_key_path = os.path.expanduser("~/.mingdaoai/replicate.key")
    with open(api_key_path, "r") as f:
        os.environ["REPLICATE_API_TOKEN"] = f.read().strip()
    
    # Generate image using Flux model
    output = replicate.run(
        "black-forest-labs/flux-schnell",
        input={
            "prompt": prompt,
            "go_fast": True,
            "megapixels": "1",
            "num_outputs": 1,
            "aspect_ratio": "1:1",
            "output_format": "webp",
            "output_quality": 80,
            "num_inference_steps": 4
        }
    )
    
    # Download and save the generated image
    if isinstance(output, list) and len(output) > 0:
        image_url = output[0]
        response = requests.get(image_url)
        if response.status_code == 200:
            with open(output_full_path, "wb") as f:
                f.write(response.content)
        else:
            raise Exception(f"Failed to download image. Status code: {response.status_code}")
    else:
        raise Exception("No image URL returned from the model")

if __name__ == "__main__":
    # Test the function
    test_prompt = "black forest gateau cake spelling out the words \"FLUX SCHNELL\", tasty, food photography, dynamic shot"
    output_path = os.path.join(os.path.dirname(__file__), "output.png")
    genImage(test_prompt, output_path)
    print(f"Image generated and saved to: {output_path}")
