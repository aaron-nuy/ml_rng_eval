import patch_random
import random
import torch_extra_utils

__lcg__ = patch_random.LCG()
__pcg__ = patch_random.PCG()
__philox__ = patch_random.Philox()
__mt19937__ = random.Random()

def init_rng(seed: int = 42):
    global __lcg__, __pcg__, __philox__
    __lcg__.seed(seed)
    __pcg__.seed(seed)
    __philox__.seed(seed)
    __mt19937__.seed(seed)

def set_rng(type: str):
    if type.lower() == "lcg":
        patch_random.patch(__lcg__)
        torch_extra_utils.change_rng("LCG")
    elif type.lower() == "png":
        patch_random.patch(__pcg__)
        torch_extra_utils.change_rng("PCG")
    elif type.lower() == "philox":
        patch_random.patch(__philox__)
        torch_extra_utils.change_rng("PHILOX")
    else:
        patch_random.unpatch()
        torch_extra_utils.change_rng("MT19937")