from .models import PixelNeRFNet
from .decoder import Decoder
from .neural_renderer import NeuralRenderer
from .discriminator import DCDiscriminator

def make_model(conf, *args, **kwargs):
    """ Placeholder to allow more model types """
    model_type = conf.get_string("type", "pixelnerf")  # single
    if model_type == "pixelnerf":
        decoder = Decoder()
        net = PixelNeRFNet(conf, decoder, *args, **kwargs)
    else:
        raise NotImplementedError("Unsupported model type", model_type)
    return net
