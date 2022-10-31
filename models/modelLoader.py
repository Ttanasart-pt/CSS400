def loadFromConfig(config):
    if config['model'] == "colorHandPose_handSeg":
        from colorHandPose.handSegNet import HandSegNet as Model
    elif config['model'] == "colorHandPose_postNet":
        from colorHandPose.posenet import PoseNet as Model
    elif config['model'] == "homebrew":
        from homebrew.CNN import Model
    elif config['model'] == "hourglass":
        from hourglass.model import Model
    elif config['model'] == "mobilenet":
        from mobilenet.model import Model
    
    return Model()