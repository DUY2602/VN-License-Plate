import os
from roboflow import Roboflow
from dotenv import load_dotenv
load_dotenv()

rf = Roboflow(api_key=os.getenv("API_KEY"))
project = rf.workspace("vietnam-license").project("vietnam-license-plate-hjswj")
version = project.version(2)
dataset = version.download("yolov8")