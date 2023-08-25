# ai_number_generator
 
**GAN for Digit Generation**

This project implements a Generative Adversarial Network for generating handwritten digit images. GANs consist of two neural networks, a generator, and a discriminator, which are trained in a competitive manner. The generator tries to produce realistic data, while the discriminator aims to distinguish between real and generated data.


**Dependencies:**

Python 3.x

TensorFlow

NumPy

Matplotlib


**Usage**

Clone this repository to your local machine:

`git clone https://github.com/zepif/ai_number_generator.git`

Ensure you have the required dependencies installed. You can install them using pip:

`pip install tensorflow numpy matplotlib`

Run the gan.py script to train the GAN:

`python gan.py`


The GAN will begin training and generate digit images over epochs. Generated images will be saved to the numbers directory. 

You can adjust the number of training epochs, batch size, and other parameters in the gan.py script to experiment with the training process.


**Project Structure**

gan.py: The main script that defines the GAN class, loads data, trains the GAN, and generates images.

data_loader.py: Contains a DataLoader class for loading and preprocessing the MNIST dataset.

discriminator.py: Defines the Discriminator neural network.

generator.py: Defines the Generator neural network.

numbers/: A directory to store generated digit images.
