# inference.py
from PIL import Image
import io
import torch
import torchvision.transforms as transforms


# Input function to handle image input
def input_fn(request_body, content_type='image/jpeg'):
    """Deserialize and preprocess the input image."""
    if content_type == 'image/jpeg':
        image = Image.open(io.BytesIO(request_body))
        preprocess = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        image = preprocess(image).unsqueeze(0)  # Add batch dimension
        return image
    else:
        raise ValueError(f"⚠️ Unsupported content type: {content_type}")


# Predict function that makes predictions using the model
def predict_fn(input_object, model):
    """Make a prediction using the trained model."""
    model.eval()
    with torch.no_grad():
        prediction = model(input_object)
    return prediction


# Output function to format the output
def output_fn(prediction, content_type='application/json'):
    """Format the prediction as JSON."""
    result = torch.argmax(prediction, dim=1).item()  # Extract predicted class
    return {" 🟢 Predicted class": result}
