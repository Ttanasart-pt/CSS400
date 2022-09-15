import torchvision.transforms as T

def preprocess(size):
    return T.Compose([
        T.Resize((size, size)),
        T.ToTensor()
    ])