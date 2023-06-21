from PIL import Image, ImageOps
import torch
from torchvision import transforms
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.nn.functional as F
import os
import sys

def main():
    #You can modify the path to image
    image_path="image.png"    
    
    
    #Don't touch the rest of the code !
    ###################################
    
    if not os.path.exists(image_path):
        print("Error: 'image.png' not found.")
        sys.exit(1)

    if not os.path.exists("model.pth"):
        print("Error: 'model.pth' not found.")
        sys.exit(1)

    img = Image.open(image_path) 
    checkpoint = torch.load('model.pth')
    model= getNetwork()
    model.load_state_dict(checkpoint['model_state_dict'])
        

    # Define the transformation
    transform = transforms.Compose([
        transforms.Grayscale(),  # Convert the image to grayscale
        transforms.Resize((28, 28)),  # Resize to 28x28 pixels
        transforms.ToTensor(),  # Convert to PyTorch tensor
        transforms.Normalize((0.1307,), (0.3081,)),  # Normalize to match MNIST dataset
    ])
    img = img.convert("L")

    img = ImageOps.invert(img)

    # Apply the transformation and add an extra dimension to match the expected input shape
    img_tensor = transform(img).unsqueeze(0)

    # Display the transformed image
    

    # Ensure the model is in evaluation mode
    model.eval()

    # Make the prediction
    output = model(img_tensor)

    # Get the predicted class
    _, predicted_class = torch.max(output.data, 1)
    
    
    prediction= predicted_class.item()
    print(f"The model predicts the digit is: {prediction}")
    plt.imshow(img_tensor[0].numpy().squeeze(), cmap='gray')
    plt.title(f"Prediction: {prediction}")
    plt.show()



if __name__ == '__main__':
    #Neural Network is defined here, don't touch unless you know what are you doing !
    class Net(nn.Module):
        def __init__(self):
            super(Net, self).__init__()
            self.conv1 = nn.Conv2d(1, 32, 3, 1)
            self.conv2 = nn.Conv2d(32, 64, 3, 1)
            self.dropout1 = nn.Dropout(0.25)
            self.dropout2 = nn.Dropout(0.25)
            self.fc1 = nn.Linear(9216, 128)
            self.fc2 = nn.Linear(128, 64)
            self.fc3 = nn.Linear(64, 64)

        def forward(self, x):
            x = self.conv1(x)
            x = F.relu(x)
            x = self.conv2(x)
            x = F.max_pool2d(x, 2)
            x = self.dropout1(x)
            x = torch.flatten(x, 1)
            x = self.fc1(x)
            x = F.relu(x)
            x = self.dropout2(x)
            x = self.fc2(x)
            x = F.relu(x)
            x= self.fc3(x)
            output = F.log_softmax(x, dim=1)
            return output

    def getNetwork():
        model = Net()
        return model

    main()