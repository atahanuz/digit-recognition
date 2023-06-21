# Digit Recognition in Python
Easy-to-use Python digit recognition model you can personally test !

I've designed a digit recognition model in Python using PyTorch, a popular machine learning library.
The model takes an image as input and predicts the digit in the image

My model achieved 99.3% accuracy on MNIST dataset, a popular dataset of handwritten digits.
In this program, you'll be able to test it by drawing digits yourself. You can use tools like Paint or Paintbrush to draw on your computer.

## Requirements:
You need to have these Python libraries installed. You can install them by running these commands in terminal (assuming Python and pip are already installed on your system.)
```
pip install Pillow
pip install torch
pip install torchvision
pip install matplotlib
```

## How to use:
Usage is extremely simple. Everything works in a single Python file (main.py), just run it from terminal or any editor.
In the same folder as code, you need two files.
```
model.pth : The machine learning model, download it.
image.png   You can download the example, then you can edit or replace it with your drawing to test the model
```

After the code is run, it will print its prediction to the console. It will also show the image you drew alongside the prediction in a separate window.

## For accurate predictions

The model should be able to correctly recognize your drawings. If it doesn't, I suggest you to look at these worked examples and draw it like this. Use a thick brush, the numbers should be large relative the image's dimensions

![Image Description](https://i.imgur.com/VxGU4oV.png)








