// Distributed under the MIT License.
// See LICENSE.txt for details.

#include <boost/python.hpp>

namespace py_bindings {
void bind_datavector();
}  // namespace py_bindings

BOOST_PYTHON_MODULE(_DataStructures) {
  Py_Initialize();
  py_bindings::bind_datavector();
}
