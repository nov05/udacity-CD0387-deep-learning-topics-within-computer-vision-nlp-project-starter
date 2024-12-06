## inference.py
'''
https://github.com/aws/sagemaker-pytorch-inference-toolkit/blob/master/src/sagemaker_pytorch_serving_container/default_pytorch_inference_handler.py
'''



import numpy as np
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import os
import json
import base64
from PIL import Image
from io import BytesIO



class ModelLoadError(Exception):
    pass



def model_fn(model_dir):
    """Loads a model. For PyTorch, a default function to load a model only if Elastic Inference is used.
    In other cases, users should provide customized model_fn() in script.
    Args:
        model_dir: a directory where model is saved.
    Returns: A PyTorch model.
    """
    print("🟢 Loading model...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"👉 Device: {device}")
    print(f"👉 Model dir: {model_dir}, type: {type(model_dir)}")
    for obj in os.listdir(model_dir):
        if os.path.isfile(os.path.join(model_dir, obj)):
            print(f"File: {obj}")
        elif os.path.isdir(os.path.join(model_dir, obj)):
            print(f"Directory: {obj}")
    model_file = model_dir + "/model.pth"
    try:
        # model = torch.jit.load(model_file, map_location=device)  ## model saved by TorchScript 
        model_type = 'resnet50'
        num_classes = 133
        model = getattr(torchvision.models, model_type)(pretrained=True)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        model.load_state_dict(torch.load(model_file, map_location=device))
        print(f"🟢 Model {model_file} loaded")
    except RuntimeError as e:
        raise ModelLoadError(
            f"⚠️ Failed to load {model_file}."
        ) from e
    model = model.to(device)
    return model



# Input function to handle image input
def input_fn(input_data, content_type):
    """A input_fn that can handle JSON, CSV and NPZ formats.
    Args:
        input_data: the request payload serialized in the content_type format
        content_type: the request content_type
    Returns: input_data deserialized into torch.FloatTensor or torch.cuda.FloatTensor,
        depending if cuda is available.
    """
    print("🟢 Processing input data...")
    print(f"👉 Input data type: {type(input_data)}, " ## str
          f"content type: {content_type}")  ## "application/json" in this case
    if content_type!='application/json':
        raise TypeError("⚠️ Expected content type 'application/json'")
    data_bytes = base64.b64decode(
        json.loads(input_data)
    )
    data_image = Image.open(BytesIO(data_bytes)).convert('RGB')  # Ensure 3-channel RGB
    preprocess = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406], 
            std=[0.229, 0.224, 0.225]
        ),
    ])
    # PyTorch models typically expect shape (batch_size, channels, height, width)
    data_tensor = preprocess(data_image).unsqueeze(0)  # Add batch dimension
    print(f"👉 Data tensor shape: {data_tensor.shape}")
    return data_tensor



def predict_fn(data, model):
    """A default predict_fn for PyTorch. Calls a model on data deserialized in input_fn.
    Runs prediction on GPU if cuda is available.
    Args:
        data: input data (torch.Tensor) for prediction deserialized by input_fn
        model: PyTorch model loaded in memory by model_fn
    Returns: a prediction
    """
    print("🟢 Making prediction...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, data = model.to(device), data.to(device)
    model.eval()
    with torch.no_grad():
        output = model(data)
    print(f"👉 Model output shape: {output.shape}")
    return output



## Output function to format the output
def output_fn(prediction, accept):
    """A default output_fn for PyTorch. Serializes predictions from predict_fn to JSON, CSV or NPY format.
    Args:
        prediction: a prediction result from predict_fn
        accept: type which the output data needs to be serialized
    Returns: output data serialized
    """
    print("🟢 Processing output data...")
    if type(prediction) is torch.Tensor:
        prediction = prediction.argmax(dim=1)
        prediction = prediction.detach().cpu().numpy().tolist()
        print(f"👉 Prediction: {prediction}")
        return json.dumps(prediction)
    raise TypeError("⚠️ Expected prediction data type 'torch.Tensor'")