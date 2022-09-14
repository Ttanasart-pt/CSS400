import torchvision.transforms as T

def preprocess(size):
    return T.Compose([
        T.Resize(size),
        T.CenterCrop(size),
        T.ToTensor()
    ])