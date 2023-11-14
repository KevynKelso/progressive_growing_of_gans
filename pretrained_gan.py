import os
import sys
import pickle
import numpy as np
import tensorflow.compat.v1 as tf
import PIL.Image
sys.path.append(os.path.dirname(__file__))

class TF1Unpickler(pickle.Unpickler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def find_class(self, module, name):
        if module == 'tensorflow':
            return tf
        return super().find_class(module, name)


def run_model(pickle_file, latents):
    try:
        # Initialize TensorFlow session.
        tf.InteractiveSession()

        # Import official CelebA-HQ networks.
        with open(pickle_file, "rb") as file:
            _, _, Gs = TF1Unpickler(file).load()

        # Generate dummy labels (not used by the official networks).
        labels = np.zeros([latents.shape[0]] + Gs.input_shapes[1][1:])

        # Run the generator to produce a set of images.
        images = Gs.run(latents, labels)

        # Convert images to PIL-compatible format.
        images = np.clip(np.rint((images + 1.0) / 2.0 * 255.0), 0.0, 255.0).astype(
            np.uint8
        )  # [-1,1] => [0,255]
        images = images.transpose(0, 2, 3, 1)  # NCHW => NHWC

        del Gs
        del labels

        tf.InteractiveSession().close()

        return images

    except Exception as e:
        tf.InteractiveSession().close()
        raise e



if __name__ == "__main__":
    # Generate latent vectors.
    latents = np.random.RandomState(901).randn(
        10, 512 # *Gs.input_shapes[0][1:]
    )  # 10 random latents

    images = run_model(sys.argv[1], latents)

    # Save images as PNG.
    for idx in range(images.shape[0]):
        PIL.Image.fromarray(images[idx], "RGB").save("img%d.png" % idx)

