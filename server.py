import argparse
import logging
import os
import time
from typing import List, Literal
import boto3
import numpy as np
import rembg
import torch
import xatlas
from PIL import Image

from tsr.system import TSR
from tsr.utils import remove_background, resize_foreground, save_video
from tsr.bake_texture import bake_texture
from fastapi import FastAPI, Request
import uvicorn, json, datetime
from pydantic import BaseModel
import base64
import io
import random
import string
import sagemaker
import uuid
model = None
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s", level=logging.INFO
)
DEFAULT_BUCKET = sagemaker.session.Session().default_bucket() 
default_output_s3uri = f's3://{DEFAULT_BUCKET}/sagemaker-endpoint/meshoutput/'
class InferenceArgs(BaseModel):
    image: List[str]
    mc_resolution: int = 256
    no_remove_bg : bool = False
    foreground_ratio:float = 0.85
    model_save_format: Literal["obj", "glb"] = 'obj'
    bake_texture : bool = False
    texture_resolution :int = 2048
    render : bool = False


def get_bucket_and_key(s3uri):
    """
    get_bucket_and_key is helper function
    """
    pos = s3uri.find('/', 5)
    bucket = s3uri[5: pos]
    key = s3uri[pos + 1:]
    return bucket, key

def write_to_s3(file):
    """
    update file to s3 bucket
    """
    dir_prefix= file.split('/')[-2]
    output_s3uri = default_output_s3uri + datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S") + dir_prefix + "/"

    bucket, key = get_bucket_and_key(output_s3uri)
    base_name = os.path.basename(file)
    key = key + base_name
    s3 = boto3.client('s3')
    try:
        s3.upload_file(file, bucket, key)
        logging.info(f"upload to s3://{bucket}/{key}")
        return f"s3://{bucket}/{key}"
    except Exception as e:
        logging.error(f"upload to s3://{bucket}/{key} failed: {e}")
        return ''





def random_string_name(length=12):
    return ''.join(random.choices(string.ascii_lowercase + string.digits, k=length))

def base64_to_image(base64_string):
    # Remove the data URL prefix if present
    if 'data:image' in base64_string:
        base64_string = base64_string.split(',')[1]
    
    # Decode base64 string
    img_data = base64.b64decode(base64_string)
    
    # Create an image object
    img = Image.open(io.BytesIO(img_data))
    
    return img


DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
OUPUT_DIR = '/tmp/output_dir'

class Timer:
    def __init__(self):
        self.items = {}
        self.time_scale = 1000.0  # ms
        self.time_unit = "ms"

    def start(self, name: str) -> None:
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        self.items[name] = time.time()
        logging.info(f"{name} ...")

    def end(self, name: str) -> float:
        if name not in self.items:
            return
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        start_time = self.items.pop(name)
        delta = time.time() - start_time
        t = delta * self.time_scale
        logging.info(f"{name} finished in {t:.2f}{self.time_unit}.")

timer = Timer()

app = FastAPI()

@app.get("/ping")
def ping():
    return {'status': 'Healthy'}


   
def process(args):
    global model
    output_dir = OUPUT_DIR
    os.makedirs(output_dir, exist_ok=True)

    temp_dir = random_string_name(16)
    if not os.path.exists(os.path.join(output_dir, temp_dir)):
        os.makedirs(os.path.join(output_dir, temp_dir))
    timer.start("Processing images")
    images = []

    if args.no_remove_bg:
        rembg_session = None
    else:
        rembg_session = rembg.new_session()

    for i, image_base64 in enumerate(args.image):
        if args.no_remove_bg:
            image = np.array(base64_to_image(image_base64).convert("RGB"))
        else:
            image = remove_background(base64_to_image(image_base64), rembg_session)
            image = resize_foreground(image, args.foreground_ratio)
            image = np.array(image).astype(np.float32) / 255.0
            image = image[:, :, :3] * image[:, :, 3:4] + (1 - image[:, :, 3:4]) * 0.5
            image = Image.fromarray((image * 255.0).astype(np.uint8))
            # if not os.path.exists(os.path.join(output_dir, str(i))):
            #     os.makedirs(os.path.join(output_dir, str(i)))
            # image.save(os.path.join(output_dir, str(i), f"input.png"))
        images.append(image)
    timer.end("Processing images")

    results = []
    for i, image in enumerate(images):
        logging.info(f"Running image {i + 1}/{len(images)} ...")

        timer.start("Running model")
        with torch.no_grad():
            scene_codes = model([image], device=DEVICE)
        timer.end("Running model")

        if args.render:
            timer.start("Rendering")
            render_images = model.render(scene_codes, n_views=30, return_type="pil")
            render_image_ret = []
            for ri, render_image in enumerate(render_images[0]):
                render_image.save(os.path.join(output_dir, temp_dir, f"{str(i)}_render_{ri:03d}.png"))
                s3_path1 = write_to_s3(os.path.join(output_dir, temp_dir, f"{str(i)}_render_{ri:03d}.png"))
                render_image_ret.append(s3_path1)
            save_video(
                render_images[0], os.path.join(output_dir, temp_dir, f"{str(i)}_render.mp4"), fps=30
            )

            s3_path2 = write_to_s3(os.path.join(output_dir, temp_dir, f"{str(i)}_render.mp4"))
            results.append({"render_images":render_image_ret,"render_video":s3_path2})
            timer.end("Rendering")

        timer.start("Extracting mesh")
        meshes = model.extract_mesh(scene_codes, not args.bake_texture, resolution=args.mc_resolution)
        timer.end("Extracting mesh")

        out_mesh_path = os.path.join(output_dir, temp_dir,  f"{str(i)}_mesh.{args.model_save_format}")
        if args.bake_texture:
            out_texture_path = os.path.join(output_dir, temp_dir, f"{str(i)}_texture.png")

            timer.start("Baking texture")
            bake_output = bake_texture(meshes[0], model, scene_codes[0], args.texture_resolution)
            timer.end("Baking texture")

            timer.start("Exporting mesh and texture")
            xatlas.export(out_mesh_path, meshes[0].vertices[bake_output["vmapping"]], bake_output["indices"], bake_output["uvs"], meshes[0].vertex_normals[bake_output["vmapping"]])
            Image.fromarray((bake_output["colors"] * 255.0).astype(np.uint8)).transpose(Image.FLIP_TOP_BOTTOM).save(out_texture_path)
            timer.end("Exporting mesh and texture")
            s3_path1 = write_to_s3(out_mesh_path)
            s3_path2 = write_to_s3(out_texture_path)
            results.append({"mesh_path":s3_path1,"texture_path":s3_path2})
        else:
            timer.start("Exporting mesh")
            meshes[0].export(out_mesh_path)
            timer.end("Exporting mesh")
            s3_path1 = write_to_s3(out_mesh_path)
            results.append({"mesh_path":s3_path1})

    return dict(results=results)
        

@app.on_event("startup")
async def startup_event():
    global model
    if model is None:
        timer.start("Initializing model")
        model = TSR.from_pretrained(
            "stabilityai/TripoSR",
            config_name="config.yaml", 
            weight_name="model.ckpt",
        )
        chunk_size = 8192
        model.renderer.set_chunk_size(chunk_size)
        model.to(DEVICE)
        timer.end("Initializing model")


@app.post("/invocations")
async def invocations(request: Request):
    body = await request.json()
    args = InferenceArgs(**body)
    return process(args)

def init_model():
    global model
    model = TSR.from_pretrained(
        "stabilityai/TripoSR",
        config_name="config.yaml",
        weight_name="model.ckpt",
    )
    chunk_size = 8192
    model.renderer.set_chunk_size(chunk_size)
    model.to(DEVICE)
    return model

if __name__ == '__main__':
    argparse = argparse.ArgumentParser()
    argparse.add_argument("--workers", type=int, default=1)
    args = argparse.parse_args()
    
    # timer.start("Initializing model")
    # init_model()
    # timer.end("Initializing model")
    uvicorn.run("server:app", host='0.0.0.0', port=8080, workers=args.workers)

    # read a image file as a  base64 string
    # image_path = "examples/chair.png"
    # with open(image_path, "rb") as f:
    #     binary_data = f.read()
    #     base_64_encoded_data = base64.b64encode(binary_data)
    #     base64_string = base_64_encoded_data.decode("utf-8")
    # ret = process(InferenceArgs(image=[base64_string]))
    # logging.info(ret)



 
