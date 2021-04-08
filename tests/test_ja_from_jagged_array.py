import unittest
import numpy_utility as npu
import numpy as np


class MyTestCase(unittest.TestCase):
    def test1(self):
        a = [
            [1, 2],
            [3]
        ]
        a = npu.ja.from_jagged_array(a)
        self.assertEqual(a.dtype, np.int64)
        self.assertEqual(a.shape, (2, 2))

    def test2(self):
        a = [
            [1, None],
            [3]
        ]
        a = npu.ja.from_jagged_array(a)
        self.assertEqual(a.dtype, np.object_)
        self.assertEqual(a.shape, (2, 2))



if __name__ == '__main__':
    unittest.main()
