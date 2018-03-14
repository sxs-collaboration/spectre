# Distributed under the MIT License.
# See LICENSE.txt for details.

def test_none():
    return None


def test_numeric(a, b):
    return a * b


def test_vector(a, b):
    test = list()
    test.append(a[0] * b[0])
    test.append(a[1] * b[1])
    return test


def test_unordered_map(a, b):
    test = {}
    test["0"] = a["0"] * b["0"]
    test["1"] = a["1"] * b["1"]
    return test


def test_tuple(a, b):
    if a[0] != 0.35 or a[1] != "test 1" or b[0] != "test 2" or b[1] != 100:
        raise RuntimeError('Failed test_tuple')
    return 0.48, "test 3"


def test_tuple_list(a, b):
    if a[0] != 0.35 or a[1] != "test 1" or b[0] != "test 2" or b[1] != 100:
        raise RuntimeError('Failed test_tuple')
    return 0.48, [0.98, "test 4"]


def test_string(a):
    if a != "test string":
        raise RuntimeError('Failed string test')
    return "back test string"
