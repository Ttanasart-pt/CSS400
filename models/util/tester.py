from PIL import Image
import cv2
import torch
from torchvision.utils import save_image
from util.landmarkDrawer import drawPointArray
from util.transforms import preprocess

def testModel(model, path, size, device, outputPath):
    proprocessor = preprocess(size)
    
    img = Image.open(path)
    img = proprocessor(img)
    
    res = model(torch.unsqueeze(img.to(device), 0)).cpu().detach().numpy()
    
    img = img.numpy().transpose(1, 2, 0) * 255
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    img, points = drawPointArray(img, res[0].tolist())
    
    cv2.imwrite(outputPath + ".png", img)
    
    with open(outputPath + ".txt", "w") as f:
        f.write(str(points))