// Distributed under the MIT License.
// See LICENSE.txt for details.

#include <boost/python.hpp>

namespace py_bindings {
void bind_info_at_compile();
}  // namespace py_bindings

BOOST_PYTHON_MODULE(_PyInformer) {
  Py_Initialize();
  py_bindings::bind_info_at_compile();
}
