"""
DiNOv2 embedder for extracting image embeddings.

Uses pretrained DiNOv2 model to extract 768-dimensional embeddings.
"""

import torch
import numpy as np
from pathlib import Path
from typing import List, Union
from PIL import Image
import torchvision.transforms as transforms


class DinoV2Embedder:
    """Extract embeddings using DiNOv2 model"""

    def __init__(self, model_name: str = "dinov2_vitb14"):
        """
        Initialize DiNOv2 embedder.

        Args:
            model_name: DiNOv2 model variant. Options:
                - dinov2_vits14: Small (384-dim)
                - dinov2_vitb14: Base (768-dim) - default
                - dinov2_vitl14: Large (1024-dim)
                - dinov2_vitg14: Giant (1536-dim)
        """
        self.model_name = model_name

        # Detect device
        if torch.backends.mps.is_available():
            self.device = torch.device("mps")
            print(f"Using MPS (Metal Performance Shaders) device")
        elif torch.cuda.is_available():
            self.device = torch.device("cuda")
            print(f"Using CUDA device")
        else:
            self.device = torch.device("cpu")
            print(f"Using CPU device")

        # Load pretrained model
        print(f"Loading {model_name} model...")
        self.model = torch.hub.load('facebookresearch/dinov2', model_name)
        self.model.eval()
        self.model.to(self.device)
        print(f"Model loaded successfully")

        # Get embedding dimension
        self.embedding_dim = self._get_embedding_dim()
        print(f"Embedding dimension: {self.embedding_dim}")

        # Define image preprocessing
        self.transform = transforms.Compose([
            transforms.Resize(256, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def _get_embedding_dim(self) -> int:
        """Determine embedding dimension by running a test image"""
        with torch.no_grad():
            test_input = torch.randn(1, 3, 224, 224).to(self.device)
            test_output = self.model(test_input)
            return test_output.shape[1]

    def embed_batch(
        self,
        images: List[np.ndarray],
        batch_size: int = 32,
        verbose: bool = True
    ) -> np.ndarray:
        """
        Extract embeddings for a batch of images.

        Args:
            images: List of images as numpy arrays (H, W) or (H, W, C)
            batch_size: Batch size for processing
            verbose: Print progress

        Returns:
            embeddings: (N, embedding_dim) array
        """
        embeddings = []
        n_images = len(images)

        if verbose:
            print(f"Extracting embeddings for {n_images} images (batch_size={batch_size})...")

        for i in range(0, n_images, batch_size):
            batch = images[i:i + batch_size]
            batch_tensors = []

            # Preprocess images
            for img in batch:
                # Convert grayscale to RGB if needed
                if len(img.shape) == 2:
                    img = np.stack([img] * 3, axis=-1)
                elif img.shape[2] == 1:
                    img = np.repeat(img, 3, axis=-1)

                # Convert to PIL Image
                pil_img = Image.fromarray(img.astype(np.uint8))

                # Apply transforms
                tensor = self.transform(pil_img)
                batch_tensors.append(tensor)

            # Stack batch
            batch_tensor = torch.stack(batch_tensors).to(self.device)

            # Extract embeddings
            with torch.no_grad():
                batch_embeddings = self.model(batch_tensor)
                embeddings.append(batch_embeddings.cpu().numpy())

            if verbose and (i // batch_size + 1) % 5 == 0:
                print(f"  Processed {min(i + batch_size, n_images)}/{n_images} images")

        # Concatenate all batches
        embeddings = np.vstack(embeddings)

        if verbose:
            print(f"Extracted embeddings shape: {embeddings.shape}")

        return embeddings

    def embed_single(self, image: np.ndarray) -> np.ndarray:
        """
        Extract embedding for a single image.

        Args:
            image: Image as numpy array (H, W) or (H, W, C)

        Returns:
            embedding: (embedding_dim,) array
        """
        embeddings = self.embed_batch([image], batch_size=1, verbose=False)
        return embeddings[0]
