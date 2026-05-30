import patch_random
import random
import torch

__lcg__ = patch_random.LCG()
__pcg__ = patch_random.PCG()
__philox__ = patch_random.Philox()

def change_rng_type(type: str, seed: int):
    if type.lower() == "lcg":
        patch_random.patch(__lcg__)
        torch.change_rng_type("LCG")
    elif type.lower() == "pcg":
        patch_random.patch(__pcg__)
        torch.change_rng_type("PCG")
    elif type.lower() == "philox":
        patch_random.patch(__philox__)
        torch.change_rng_type("PHILOX")
    else:
        patch_random.unpatch()
        torch.change_rng_type("MT19937")

    random.seed(seed)
    torch.manual_seed(seed)

def change_rng_type_without_resetting_seed(type: str):
    r"""
        This is used to change the prng engine mid training process.
        It ensures that the starting state after the change is different
        from the initial state.
    """
    next_seed = int(torch.randint(0, 2**31, (1,)).item())
    change_rng_type(type, next_seed)
