#! /usr/bin/env python3

import os
import replicate
import requests
import logging
from mcp.server.fastmcp import FastMCP

# Initialize logger
logger = logging.getLogger(__name__)

# Initialize MCP server
mcp = FastMCP("Image Generator")

@mcp.tool()
def gen_image(prompt: str, output_full_path: str) -> str:
    """
    Generate an image using the Flux model on Replicate.
    
    Args:
        prompt (str): The text prompt to generate the image from
        output_full_path (str): Full path where the generated image will be saved
        
    Returns:
        str: Path to the generated image
    """
    logger.info(f"Generating image with prompt: {prompt}")
    
    # Read API key from file
    api_key_path = os.path.expanduser("~/.mingdaoai/replicate.key")
    logger.debug(f"Reading API key from: {api_key_path}")
    with open(api_key_path, "r") as f:
        os.environ["REPLICATE_API_TOKEN"] = f.read().strip()
    
    # Generate image using Flux model
    logger.info("Calling Replicate API to generate image")
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
        logger.info(f"Downloading image from URL: {image_url}")
        response = requests.get(image_url)
        if response.status_code == 200:
            with open(output_full_path, "wb") as f:
                f.write(response.content)
            logger.info(f"Image successfully saved to: {output_full_path}")
            return f"Image generated and saved to: {output_full_path}"
        else:
            error_msg = f"Failed to download image. Status code: {response.status_code}"
            logger.error(error_msg)
            raise Exception(error_msg)
    else:
        error_msg = "No image URL returned from the model"
        logger.error(error_msg)
        raise Exception(error_msg)

if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('step2_image_mcp.log'),
            logging.StreamHandler()
        ]
    )
    
    # Run the MCP server
    logger.info("Starting MCP server...")
    mcp.run(transport="stdio")