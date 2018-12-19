// Distributed under the MIT License.
// See LICENSE.txt for details.

#include <boost/python.hpp>

namespace py_bindings {
void bind_datavector();
void bind_matrix();
}  // namespace py_bindings

BOOST_PYTHON_MODULE(_DataStructures) {
  Py_Initialize();
  py_bindings::bind_datavector();
  py_bindings::bind_matrix();
}
