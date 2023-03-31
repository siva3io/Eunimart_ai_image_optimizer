import torch
from .resolution_test import checkpoint,Trainer
from .resolution_model import Model
from .resolution_input import Data
torch.manual_seed(1)

class ResolutionEnhancement(object):

    def main(self,image):
        check = checkpoint(None)
        global model
        if check.ok:
            loader = Data(None,image)
            _model = Model(None, check)
            _loss=None
            trainer = Trainer(None, loader, _model, _loss, check)
            image_=trainer.terminate(image)
        return image_




