# Digit Recognition in Python
Easy-to-use Python digit recognition model you can personally test !

I designed a digit recognition model in Python using PyTorch, a popular machine learning libray.
The model takes an image as input and predicts the digit in the image

My model achieved 99.3% accuracy on MNIST dataset, a popular dataset of handwritten digits.
In this program, you'll be able to test it by drawing digits yourself. You can use tools like Paint, Paintbrush to draw on your computer.

## Requirements:
You need to have these Python libraries installed. You can install them by running these commands in terminal(assuming Python and pip are already installed on your system.)
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
model.pth : Includes the machine learning model, download it.
image.png   You can edit or replace this with your drawing to test the model
```

After the code is runned, it will print its prediction to the console. It will also show the image you drew alongside the prediction.

## For accurate predictions

The model should be able to correctly recognize your drawings. If it won't, I suggest you to look at these worked examples and draw it like this. Use a tick brush, the numbers should be big relative the image's dimensions

![Image Description](https://i.imgur.com/VxGU4oV.png)








