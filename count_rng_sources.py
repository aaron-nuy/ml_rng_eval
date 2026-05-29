import patch_random
import torch_extra_utils

lcg = patch_random.LCG()
patch_random.patch(lcg)

import mnist_train

mnist_train.main()

numpy_rng_calls = patch_random.__counter_to_rand_np__
python_rng_calls = patch_random.__counter_to_rand__
pytorch_rng_calls = torch_extra_utils.get_count()

all_rng_calls = numpy_rng_calls + python_rng_calls + pytorch_rng_calls

print(f"all_rng_calls: {all_rng_calls}")
print(f"numpy_rng_calls: {numpy_rng_calls} --> {numpy_rng_calls / all_rng_calls * 100:.2f}% of all calls")
print(f"python_rng_calls: {python_rng_calls} --> {python_rng_calls / all_rng_calls * 100:.2f}% of all calls")
print(f"pytorch_rng_calls: {pytorch_rng_calls} --> {pytorch_rng_calls / all_rng_calls * 100:.2f}% of all calls")
