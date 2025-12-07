# this file is for creating a machine learning model based on categorical data 
#without the need of a camera image classifier, rather looking at dataset and labels directly
# This was developed with the assistance of artificial intelligence
# will be using pytorch and importing the needed libraries

import torch, os , numpy as np
from PIL import Image

#get the images into red green blue layout and accurate pixel amount otherwise getting mat mult error and in form of arr
loadImages = torch.stack([torch.tensor(np.array(Image.open(os.path.join("train/images",f)).convert("RGB").resize((64,64)))).permute(2,0,1).float()/255
                 for f in os.listdir("train/images")])
loadLabels = torch.tensor([int(open(os.path.join("train/labels", f.replace(".jpg", ".txt"))).read().split()[0])
                  for f in os.listdir("train/images")])

m = torch.nn.Linear(loadImages .view(len(loadImages), -1).shape[1], 2)

# this method is needed in python in order to actually get how far off
# based on the interval classfying if its more of a 1 closer or 0 further
torch.nn.CrossEntropyLoss()(m(loadImages .view(len(loadImages), -1)), loadLabels).backward();

torch.optim.Adam(m.parameters()).step() # now actually modify those exact weightages

# storing the model weight here
torch.save(m.state_dict(), "model.pth");   

print("successful w the training")

# now for the predicition

m.load_state_dict(torch.load("model.pth"))
m.eval()

testingImage = Image.open("train/images/GOPR0492_MP4-0_jpg.rf.00209cfdd7156b1316a2064cd640c595.jpg").convert("RGB").resize((64,64))
#train/images/GOPR0492_MP4-520_jpg.rf.63d37a6c1c72b083f9f63631c5fb4f2c.jpg answer for drowsy
#train/images/GOPR0492_MP4-0_jpg.rf.00209cfdd7156b1316a2064cd640c595.jpg for awake classfying

newLoadingImage = torch.tensor(np.array(testingImage)).permute(2,0,1).float()/255
newLoadingImage = newLoadingImage.reshape(1, -1)

with torch.no_grad():
    pred = m(newLoadingImage) #getting that new val

    label = torch.argmax(pred)

print("Based on model image is classified as: ", "Drowsy" if label==1 else "Awake")
  
