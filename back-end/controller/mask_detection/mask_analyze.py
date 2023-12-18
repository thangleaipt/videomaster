import time
import cv2
import torch 
import torchvision.transforms as transforms
from PIL import Image

class MaskDetector:

    def __init__(self, modelPath):
        self.modelPath = modelPath
        self.setup = True
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.maskSetup()
        
    def maskSetup(self):
        if self.setup:
            checkpoint = torch.load(self.modelPath)
           
            model = checkpoint['model']
            model.to(self.device)
            model.load_state_dict(checkpoint['state_dict'])
            for parameter in model.parameters():
                parameter.requires_grad = False
            
            self.model = model.eval()

            self.transforms = transforms.Compose([
                                       transforms.Resize((224,224)),
                                       transforms.ToTensor(),
                                       transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))
                                       ])
        
            self.setup = False


    def maskProcess(self, face):
        try:
            temp_image = Image.fromarray(face, mode = "RGB")
            temp_image = self.transforms(temp_image)
            image = temp_image.unsqueeze(0)
            image = image.to(self.device)
            result = self.model(image)
            _, maximum = torch.max(result.data, 1)
            prediction = maximum.item()

            label = 1 if prediction < 0.5 else 0
            return label
        except Exception as e:
            print("[PersonAndFaceResult][mask_detection]: ", e)
            return 0