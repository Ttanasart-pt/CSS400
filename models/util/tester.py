from PIL import Image
import cv2
import torch
from torchvision.utils import save_image
from util.landmarkDrawer import drawPointArray
from util.transforms import preprocess

def testModel(model, path, size, device, outputPath):
    preprocessor = preprocess(size)
    
    img = Image.open(path)
    img = preprocessor(img)
    
    res = model(torch.unsqueeze(img.to(device), 0)).cpu().detach().numpy()
    
    img = img.numpy().transpose(1, 2, 0) * 255
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    img, points = drawPointArray(img, res[0].tolist())
    
    cv2.imwrite(outputPath + ".png", img)
    
    with open(outputPath + ".txt", "w") as f:
        f.write(str(points))
        
def runModel(model, img, size, device):
    preprocessor = preprocess(size)
    
    img = Image.fromarray(img)
    img = preprocessor(img)
    
    res = model(torch.unsqueeze(img.to(device), 0)).cpu().detach().numpy()
    
    landmark = res[0].tolist()
    points = []
    for i in range(int(len(landmark) / 2)):
        c = (landmark[i * 2], landmark[i * 2 + 1])
        points.append(c)
    return points