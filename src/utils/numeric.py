import os.path
import os, sys
current_dir = os.path.split(os.path.abspath(__file__))[0]
proj_dir = current_dir.rsplit('/', 2)[0]
sys.path.append(proj_dir)


def float_approximately_equal(a, b, epsilon=1e-5):
    return abs(a - b) < epsilon