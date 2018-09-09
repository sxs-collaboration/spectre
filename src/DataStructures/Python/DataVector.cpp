// Distributed under the MIT License.
// See LICENSE.txt for details.

#include <boost/python.hpp>
#include <sstream>
#include <string>
#include <utility>

#include "DataStructures/DataVector.hpp"
#include "PythonBindings/BoundChecks.hpp"
#include "Utilities/GetOutput.hpp"

namespace bp = boost::python;

namespace py_bindings {
void bind_datavector() {
  // Wrapper for basic DataVector operations
  bp::class_<DataVector>("DataVector", bp::init<size_t>())
      .def(bp::init<size_t, double>())
      // __len__ is for being able to write len(my_data_vector) in python
      .def_readonly("__len__", &DataVector::size)
      // __getitem__ and __setitem__ are the subscript operators (operator[]).
      // To define (and overload) operator() use __call__
      .def("__getitem__",
           +[](const DataVector& t, const size_t i) {
             bounds_check(t, i);
             return t[i];
           })
      .def("__setitem__",
           +[](DataVector& t, const size_t i, const double v) {
             bounds_check(t, i);
             t[i] = v;
           })
      // Need __str__ for converting to string/printing
      .def("__str__", +[](const DataVector& t) { return get_output(t); })
      // repr allows you to output the object in an interactive python terminal
      // using obj to get the "string REPResenting the object".
      .def("__repr__", +[](const DataVector& t) { return get_output(t); })
      .def(bp::self += bp::self)
      .def(bp::self += bp::other<double>{})
      // Need to do math explicitly converting to DataVector because we don't
      // want to represent all the possible expression template types
      .def("__add__",
           +[](const DataVector& self, const double other) {
             return DataVector{self + other};
           })
      .def("__radd__",
           +[](const DataVector& self, const double other) {
             return DataVector{other + self};
           })
      .def("__sub__",
           +[](const DataVector& self, const double other) {
             return DataVector{self - other};
           })
      .def("__rsub__",
           +[](const DataVector& self, const double other) {
             return DataVector{other - self};
           })
      .def("__mul__",
           +[](const DataVector& self, const double other) {
             return DataVector{self * other};
           })
      .def("__rmul__",
           +[](const DataVector& self, const double other) {
             return DataVector{other * self};
           })
      // Need __div__ for python 2 and __truediv__ for python 3.
      .def("__div__",
           +[](const DataVector& self, const double other) {
             return DataVector{self / other};
           })
      .def("__truediv__",
           +[](const DataVector& self, const double other) {
             return DataVector{self / other};
           })
      .def("__rdiv__",
           +[](const DataVector& self, const double other) {
             return DataVector{other / self};
           })
      .def("__rtruediv__",
           +[](const DataVector& self, const double other) {
             return DataVector{other / self};
           })

      // DataVector-DataVector math
      .def("__add__",
           +[](const DataVector& self, const DataVector& other) {
             return DataVector{self + other};
           })
      .def("__sub__",
           +[](const DataVector& self, const DataVector& other) {
             return DataVector{self - other};
           })
      .def("__mul__",
           +[](const DataVector& self, const DataVector& other) {
             return DataVector{self * other};
           })
      .def("__div__",
           +[](const DataVector& self, const DataVector& other) {
             return DataVector{self / other};
           })
      .def("__truediv__",
           +[](const DataVector& self, const DataVector& other) {
             return DataVector{self / other};
           })
      // NOLINTNEXTLINE(misc-redundant-expression)
      .def(bp::self == bp::self)
      // NOLINTNEXTLINE(misc-redundant-expression)
      .def(bp::self != bp::self)
      .def("__neg__", +[](const DataVector& t) { return DataVector{-t}; });
}
}  // namespace py_bindings
