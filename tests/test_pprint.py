import numpy as np
import numpy_utility.io.pretty_printing

if __name__ == "__main__":
    a = np.empty(
        10,
        dtype=[("parent", [("child1", "i8"), ("child2", "i8")]), ("only_child1", "f8"), ("only_child2", "f8")]
    )
    numpy_utility.io.pretty_printing.print_structured_array(a)

    a = np.ma.empty(
        10,
        dtype=[("only_child1", "f8"), ("only_child2", "f8")]
    )
    a.mask["only_child1"][::2] = True
    numpy_utility.io.pretty_printing.print_structured_array(a)
