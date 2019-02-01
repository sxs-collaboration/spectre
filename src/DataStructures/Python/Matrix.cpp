// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "DataStructures/Python/Matrix.hpp"

#include <array>
#include <boost/python.hpp>
#include <boost/python/reference_existing_object.hpp>
#include <boost/python/return_value_policy.hpp>
#include <boost/python/tuple.hpp>
#include <cstddef>
#include <sstream>
#include <string>
#include <utility>

#include "DataStructures/Matrix.hpp"
#include "DataStructures/Python/ToNumpy.hpp"
#include "PythonBindings/BoundChecks.hpp"
#include "Utilities/MakeString.hpp"

namespace bp = boost::python;

namespace py_bindings {

void bind_matrix() {
  // Wrapper for basic Matrix operations
  bp::class_<Matrix>("Matrix", bp::init<size_t, size_t>())
      .add_property("shape",
                    +[](const Matrix& self) {
                      bp::tuple a = bp::make_tuple<size_t, size_t>(
                          self.rows(), self.columns());
                      return a;
                    })
      // __getitem__ and __setitem__ are the subscript operators (M[*,*]).
      .def("__getitem__",
           +[](const Matrix& self, bp::tuple x) -> double {
             matrix_bounds_check(self, bp::extract<size_t>(x[0]),
                                 bp::extract<size_t>(x[1]));
             return self(bp::extract<size_t>(x[0]), bp::extract<size_t>(x[1]));
           })
      // Need __str__ for converting to string/printing
      .def(
          "__str__",
          +[](const Matrix& self) { return std::string(MakeString{} << self); })
      .def("__setitem__",
           +[](Matrix& self, bp::tuple x, const double val) {
             matrix_bounds_check(self, bp::extract<size_t>(x[0]),
                                 bp::extract<size_t>(x[1]));
             self(bp::extract<size_t>(x[0]), bp::extract<size_t>(x[1])) = val;
           })
      .def("to_numpy", +[](const Matrix& self) { return to_numpy(self); },
           "Convert Matrix to a Numpy Array. Always creates a copy");
}

}  // namespace py_bindings
