import unittest
import numpy as np
import numpy_utility as npu


class MyTestCase(unittest.TestCase):
    def test1(self):
        a1 = np.empty((0,), dtype=[("index", "i4"), ("b", "f8")])
        a2 = np.empty((0,), dtype=[("index", "i4"), ("c", "f8")])
        npu.merge_arrays([a1, a2])

    def test2(self):
        a1 = np.empty((10,), dtype=[("index", "U10"), ("b", "f8")])
        a2 = np.empty((10,), dtype=[("index", "U2"), ("c", "f8")])
        a1["index"] = np.arange(10)
        a2["index"] = np.arange(10)
        npu.merge_arrays([a1, a2])


if __name__ == '__main__':
    unittest.main()
