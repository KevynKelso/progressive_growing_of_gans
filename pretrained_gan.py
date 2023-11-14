import pickle
import numpy as np
import tensorflow.compat.v1 as tf
import PIL.Image

class TF1Unpickler(pickle.Unpickler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def find_class(self, module, name):
        print(f"hit {module}, {name}")
        if module == 'tensorflow':
            print("hit")
            return tf
        return super().find_class(module, name)
    def load_inst():
        module = self.readline()[:-1].decode("ascii")
        name = self.readline()[:-1].decode("ascii")
        print(f"hit2 {module} {name}")
        return super().load_inst()


def main():
    # Initialize TensorFlow session.
    tf.InteractiveSession()

    # Import official CelebA-HQ networks.
    with open("karras2018iclr-celebahq-1024x1024.pkl", "rb") as file:
        G, D, Gs = TF1Unpickler(file).load()

    # Generate latent vectors.
    latents = np.random.RandomState(999).randn(
        1000, *Gs.input_shapes[0][1:]
    )  # 1000 random latents
    latents = latents[
        [477, 56, 83, 887, 583, 391, 86, 340, 341, 415]
    ]  # hand-picked top-10

    # Generate dummy labels (not used by the official networks).
    labels = np.zeros([latents.shape[0]] + Gs.input_shapes[1][1:])

    # Run the generator to produce a set of images.
    images = Gs.run(latents, labels)

    # Convert images to PIL-compatible format.
    images = np.clip(np.rint((images + 1.0) / 2.0 * 255.0), 0.0, 255.0).astype(
        np.uint8
    )  # [-1,1] => [0,255]
    images = images.transpose(0, 2, 3, 1)  # NCHW => NHWC

    # Save images as PNG.
    for idx in range(images.shape[0]):
        PIL.Image.fromarray(images[idx], "RGB").save("img%d.png" % idx)


if __name__ == "__main__":
    main()
