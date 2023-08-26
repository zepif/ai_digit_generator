import os
import argparse
import tensorflow as tf

def set_cuda():
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    physical_devices = tf.config.list_physical_devices('GPU')
    if len(physical_devices) == 0:
        print("No GPUs available. Make sure that TensorFlow-GPU is installed and the graphics card is CUDA compatible.")
    else:
        tf.config.experimental.set_memory_growth(physical_devices[0], True)

def main():
    parser = argparse.ArgumentParser(description="Choose GAN architecture")
    parser.add_argument("--architecture", choices=["gan1", "gan2"], required=True, help="Specify GAN architecture")
    args = parser.parse_args()

    if args.architecture == "gan1":
        from gan import GAN
        gan1 = GAN(architecture='gan1')  # Create GAN instance for 'gan1' architecture
        gan1.train_gan(epochs=100, batch_size=128)
    elif args.architecture == "gan2":
        from dcgan import DCGAN
        gan2 = DCGAN(architecture='gan2')  # Create GAN instance for 'gan2' architecture
        gan2.train_gan(epochs=200, batch_size=128)
    else:
        print("Invalid architecture choice. Use 'gan1' or 'gan2'.")


if __name__ == "__main__":
    set_cuda()
    main()
