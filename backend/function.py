import open_clip
import torch
import torch.nn as nn
from PIL import Image
from pathlib import Path
import torchvision.transforms as transforms
import warnings

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
num_classes = 6
class_names = ['Calculus', 'Caries', 'Gingivitis',
               'Mouth Ulcer', 'Tooth Discoloration', 'hypodontia']

BASE_DIR = Path(__file__).resolve(strict=True).parent.parent


class CustomCLIPVisionTransformer(nn.Module):
    def __init__(self, num_classes: int, model_name: str = "ViT-B-16", pretrained: str = "openai"):
        super(CustomCLIPVisionTransformer, self).__init__()

        # 1️⃣ Load pretrained CLIP model (ViT backbone only)
        # The `open_clip` factory may warn when the requested pretrained tag
        # (e.g. 'openai') expects QuickGELU but the default model config
        # doesn't enable it. Choose quick_gelu to match the pretrained tag to
        # avoid the UserWarning and ensure compatibility.
        quick_gelu = True if pretrained == "openai" else False
        # Suppress known QuickGELU mismatch warning emitted by open_clip.factory
        # by temporarily filtering that specific UserWarning. This avoids noisy
        # logs while keeping other warnings visible.

        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message=r"QuickGELU mismatch between final model config.*",
                category=UserWarning,
            )
            self.clip_model, _, _ = open_clip.create_model_and_transforms(
                model_name,
                pretrained=pretrained,
                quick_gelu=quick_gelu,
            )
        self.visual_encoder = self.clip_model.visual  # ViT visual encoder

        # 2️⃣ Unfreeze for fine-tuning
        for param in self.visual_encoder.parameters():
            param.requires_grad = True

        # 3️⃣ Get CLIP ViT embedding dimensions
        # The visual encoder's output_dim is the dimension of the pooled feature
        self.embed_dim = self.visual_encoder.output_dim  # e.g. 512 or 768

        # Removed additional self-attention layer as it requires patch tokens
        # and accessing them directly from open_clip's ViT is proving difficult.
        # We will use the standard pooled feature from the visual encoder.

        # 4️⃣ Classification head — flexible MLP
        self.classifier = nn.Sequential(
            nn.Linear(self.embed_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )

        # Removed attn_maps as we are not explicitly capturing attention weights in this simplified version.
        # If attention visualization is needed, a more advanced approach with hooks would be required.

    def forward(self, x):
        """
        x: [B, 3, H, W]
        returns: logits for num_classes
        """
        # Use the standard forward pass of the visual encoder
        # open_clip's VisionTransformer forward method typically returns the pooled output (CLS token feature)
        pooled_features = self.visual_encoder(x)  # Shape [B, embed_dim]

        # Pass the pooled features through the classification MLP
        logits = self.classifier(pooled_features)

        # Return logits
        return logits


best_model_clip = CustomCLIPVisionTransformer(num_classes)
path = f"{BASE_DIR}/models/best_model_clip.pth"
best_model_clip.load_state_dict(torch.load(
    path, map_location=torch.device('cpu')))
best_model_clip.to(device)
best_model_clip.eval()


def predict(image_path):
    """
    Run inference on a single image or batch.

    Args:
        x: one of:
            - torch.Tensor: shape [C,H,W] or [B,C,H,W] (assumed to be in model input space)
            - numpy.ndarray: HxWxC (uint8 BGR or RGB) or CxHxW
            - PIL.Image.Image: will be transformed with the CLIP preprocessing for ViT-B-16
        top_k: (optional) number of top predictions to return (default 1)

    Returns:
        dict with keys:
            - "indices": list of predicted class indices (length top_k)
            - "probs": list of probabilities (0-100) corresponding to indices
    """
    try:
        # Load and preprocess image
        input_image = Image.open(image_path).convert('RGB')
        preprocess = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])
        input_tensor = preprocess(input_image).unsqueeze(0).to(device)

        print("Image preprocessed\n\n\n", input_tensor.shape, "\n\n\n")

        # Get prediction - handle different output formats
        with torch.no_grad():
            outputs = best_model_clip(input_tensor)

            if isinstance(outputs, tuple):
                if len(outputs) >= 1:
                    logits = outputs[0]
                else:
                    raise ValueError("Empty tuple output")
            else:
                # Single tensor output
                logits = outputs

            probabilities = torch.softmax(logits, dim=1)
            _, predicted = torch.max(logits, 1)
            predicted_class = class_names[predicted.item()]
            confidence = probabilities[0, predicted.item()].item()

            print("Prediction computed\n\n\n",
                  predicted_class, confidence, "\n\n\n")

        return predicted_class, confidence

    except Exception as e:
        print(f"Error processing {image_path}: {e}")
        return None, e
