import random as _random_module
import time

__counter_to_rand__ = 0

class LCG(_random_module.Random):
    _M = 1 << 31
    _A = 65539

    def seed(self, a=None, version=2):
        if a is None:
            a = int(time.time() * 1000)
        a = int(a) & 0x7FFFFFFF
        self._s = a if a else 1

    def _next(self):
        self._s = (self._A * self._s) % self._M
        return self._s

    def random(self):
        global __counter_to_rand__
        __counter_to_rand__ += 1
        return self._next() / self._M

    def getstate(self):
        return (self._s,)

    def setstate(self, state):
        self._s = state[0] % self._M

    def getrandbits(self, k):
        global __counter_to_rand__
        __counter_to_rand__ += 1
        if k < 0:  raise ValueError("number of bits must be non-negative")
        if k == 0: return 0
        out, bits = 0, 0
        while bits < k:
            out = (out << 31) | self._next()
            bits += 31
        return out & ((1 << k) - 1)


class Philox(_random_module.Random):
    _M = 0xD256D193
    _W = 0x9E3779B9

    def seed(self, a=None, version=2):
        if a is None:
            a = int(time.time() * 1000)
        self._key = int(a) & 0xFFFFFFFF
        self._ctr = 0
        self._buf = None

    def _generate(self):
        c0 = self._ctr & 0xFFFFFFFF
        c1 = (self._ctr >> 32) & 0xFFFFFFFF
        k = self._key
        for _ in range(10):
            prod = self._M * c0
            hi = (prod >> 32) & 0xFFFFFFFF
            lo = prod & 0xFFFFFFFF
            c0 = (hi ^ k ^ c1) & 0xFFFFFFFF
            c1 = lo
            k = (k + self._W) & 0xFFFFFFFF
        self._ctr = (self._ctr + 1) & 0xFFFFFFFFFFFFFFFF
        return c0, c1

    def _next(self):
        if self._buf is not None:
            val, self._buf = self._buf, None
            return val
        r0, r1 = self._generate()
        self._buf = r1
        return r0

    def random(self):
        global __counter_to_rand__
        __counter_to_rand__ += 1
        return self._next() / 0x100000000

    def getstate(self):
        return (self._key, self._ctr, self._buf)

    def setstate(self, state):
        self._key, self._ctr, self._buf = state

    def getrandbits(self, k):
        global __counter_to_rand__
        __counter_to_rand__ += 1
        if k < 0:  raise ValueError("number of bits must be non-negative")
        if k == 0: return 0
        out, bits = 0, 0
        while bits < k:
            out = (out << 32) | self._next()
            bits += 32
        return out & ((1 << k) - 1)


class PCG(_random_module.Random):
    _MULT = 6364136223846793005
    _INC = 1442695040888963407
    _MASK = (1 << 64) - 1

    def seed(self, a=None, version=2):
        if a is None:
            a = int(time.time() * 1000)
        self._state = ((int(a) + self._INC) * self._MULT + self._INC) & self._MASK

    def _rotr32(x, r):
        return ((x >> r) | (x << ((32 - r) & 31))) & 0xFFFFFFFF

    def _next(self):
        s = self._state
        xorshifted = (((s >> 18) ^ s) >> 27) & 0xFFFFFFFF
        out = self._rotr32(xorshifted, s >> 59)
        self._state = (s * self._MULT + self._INC) & self._MASK
        return out

    def random(self):
        global __counter_to_rand__
        __counter_to_rand__ += 1
        return self._next() / 0x100000000

    def getstate(self):
        return (self._state,)

    def setstate(self, state):
        self._state = state[0] & self._MASK

    def getrandbits(self, k):
        global __counter_to_rand__
        __counter_to_rand__ += 1
        if k < 0:  raise ValueError("number of bits must be non-negative")
        if k == 0: return 0
        out, bits = 0, 0
        while bits < k:
            out = (out << 32) | self._next()
            bits += 32
        return out & ((1 << k) - 1)


__functions_to_patch__ = [
    "seed", "random", "getstate", "setstate", "getrandbits",
    "randrange", "randint", "choice", "choices", "shuffle", "sample",
    "uniform", "triangular", "expovariate", "gammavariate", "gauss",
    "normalvariate", "lognormvariate", "vonmisesvariate", "paretovariate",
    "weibullvariate", "randbytes",
]

__original_functions__: dict = {}


def patch(gen: _random_module.Random) -> None:
    if gen is None or not isinstance(gen, _random_module.Random):
        return None

    if not __original_functions__:
        __original_functions__["_inst"] = _random_module._inst

        for name in __functions_to_patch__:
            if hasattr(_random_module, name):
                __original_functions__[name] = getattr(_random_module, name)

    _random_module._inst = gen

    for name in __functions_to_patch__:
        if hasattr(_random_module, name):
            setattr(_random_module, name, getattr(gen, name))
    return None


def unpatch() -> None:
    if not __original_functions__:
        return

    _random_module._inst = __original_functions__["_inst"]

    for name in __functions_to_patch__:
        if name in __original_functions__:
            setattr(_random_module, name, __original_functions__[name])

    __original_functions__.clear()