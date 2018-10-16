// Distributed under the MIT License.
// See LICENSE.txt for details.

#include <boost/python.hpp>

// These macros are required so that the NumPy API will work when used in
// multiple cpp files.  See
// https://docs.scipy.org/doc/numpy/reference/c-api.array.html#importing-the-api
#define PY_ARRAY_UNIQUE_SYMBOL SPECTRE_IO_H5_PYTHON_BINDINGS
// Code is clean against Numpy 1.7, see
// https://docs.scipy.org/doc/numpy-1.15.1/reference/c-api.deprecations.html
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <numpy/arrayobject.h>

// We do not use the antipattern:
// #define NO_IMPORT_ARRAY
// #include "DataStructures/Python/Numpy.hpp"
// because
// 1. This means we are controlling code in the header file with a
//    local macro, which can be very confusing and should generally be avoided.
// 2. When C++ modules land this type of pattern will no longer really be
//    possible since it goes against exactly what modules is trying to do: make
//    things modular.

namespace py_bindings {
void bind_h5file();
void bind_h5dat();
}  // namespace py_bindings

BOOST_PYTHON_MODULE(_H5) {
  Py_Initialize();
  import_array();
  py_bindings::bind_h5file();
  py_bindings::bind_h5dat();
}
