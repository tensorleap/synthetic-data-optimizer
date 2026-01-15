import numpy as np
import onnx
import onnxruntime as ort
from pathlib import Path
from typing import List, Union
from PIL import Image


class UNetEmbedder:

    def __init__(self, model_path: Union[str, Path] = None, embedding_layer: str = None):
        if model_path is None:
            model_path = Path(__file__).parent / "All_3ChipTypes_seg_model_deployed.onnx"

        self.model_path = Path(model_path)

        if not self.model_path.exists():
            raise FileNotFoundError(f"Model not found at {self.model_path}")

        if embedding_layer is None:
            embedding_layer = 'Conv__247:0'

        self.embedding_layer = embedding_layer

        model = onnx.load(str(self.model_path))

        intermediate_layer_value_info = onnx.helper.ValueInfoProto()
        intermediate_layer_value_info.name = self.embedding_layer
        model.graph.output.append(intermediate_layer_value_info)

        self.sess = ort.InferenceSession(model.SerializeToString())

        self.input_name = self.sess.get_inputs()[0].name
        self.input_shape = self.sess.get_inputs()[0].shape

        self.input_height = 250
        self.input_width = 220

        test_input = np.random.randn(1, self.input_height, self.input_width, 1).astype(np.float32)
        test_outputs = self.sess.run(None, {self.input_name: test_input})
        test_embedding = test_outputs[-1]

        self.embedding_shape = test_embedding.shape[1:]
        self.embedding_dim = int(np.prod(self.embedding_shape))

        print(f"UNet embedder initialized")
        print(f"  Model: {self.model_path.name}")
        print(f"  Input shape: (batch, {self.input_height}, {self.input_width}, 1)")
        print(f"  Embedding layer: {self.embedding_layer}")
        print(f"  Embedding shape: {self.embedding_shape}")
        print(f"  Embedding dim (flattened): {self.embedding_dim}")

    def embed_batch(
        self,
        images: List[np.ndarray],
        batch_size: int = 32,
        verbose: bool = True
    ) -> np.ndarray:
        embeddings = []
        n_images = len(images)

        if verbose:
            print(f"Extracting embeddings for {n_images} images (batch_size={batch_size})...")

        for i in range(0, n_images, batch_size):
            batch = images[i:i + batch_size]
            batch_arrays = []

            for img in batch:
                preprocessed = self._preprocess_image(img)
                batch_arrays.append(preprocessed)

            batch_array = np.stack(batch_arrays, axis=0)

            outputs = self.sess.run(None, {self.input_name: batch_array})
            batch_embeddings = outputs[-1]

            batch_embeddings_flat = batch_embeddings.reshape(len(batch), -1)
            embeddings.append(batch_embeddings_flat)

            if verbose and (i // batch_size + 1) % 5 == 0:
                print(f"  Processed {min(i + batch_size, n_images)}/{n_images} images")

        embeddings = np.vstack(embeddings)

        if verbose:
            print(f"Extracted embeddings shape: {embeddings.shape}")

        return embeddings

    def embed_single(self, image: np.ndarray) -> np.ndarray:
        embeddings = self.embed_batch([image], batch_size=1, verbose=False)
        return embeddings[0]

    def _preprocess_image(self, image: np.ndarray) -> np.ndarray:
        if len(image.shape) == 2:
            h, w = image.shape
            channels = 1
        elif len(image.shape) == 3:
            h, w, channels = image.shape
        else:
            raise ValueError(f"Invalid image shape: {image.shape}")

        if channels > 1:
            from PIL import Image
            pil_img = Image.fromarray(image.astype(np.uint8))
            pil_img = pil_img.convert('L')
            image = np.array(pil_img)

        if image.shape != (self.input_height, self.input_width):
            from PIL import Image
            pil_img = Image.fromarray(image.astype(np.uint8))
            pil_img = pil_img.resize((self.input_width, self.input_height), Image.Resampling.BILINEAR)
            image = np.array(pil_img)

        image = image.astype(np.float32)

        if image.max() > 1.0:
            image = image / 255.0

        image = image.reshape(self.input_height, self.input_width, 1)

        return image
