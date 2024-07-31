// Distributed under the MIT License.
// See LICENSE.txt for details.

#include <pybind11/operators.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <string>

#include "DataStructures/Index.hpp"
#include "DataStructures/Python/DataVector.hpp"
#include "DataStructures/Python/Matrix.hpp"
#include "DataStructures/Python/ModalVector.hpp"
#include "PythonBindings/BoundChecks.hpp"
#include "Utilities/ErrorHandling/SegfaultHandler.hpp"
#include "Utilities/GetOutput.hpp"

namespace py = pybind11;

namespace {
template <size_t Dim>
void bind_index_impl(py::module& m) {  // NOLINT
  py::class_<Index<Dim>>(m, ("Index" + std::to_string(Dim) + "D").c_str())
      .def(py::init<const size_t>(), py::arg("i0"))
      .def(py::init<std::array<size_t, Dim>>(), py::arg("i"))
      .def(
          "__iter__",
          [](const Index<Dim>& t) {
            return py::make_iterator(t.begin(), t.end());
          },
          // Keep object alive while iterator exists
          py::keep_alive<0, 1>())
      // __len__ is for being able to write len(my_data_vector) in python
      .def("__len__", [](const Index<Dim>& t) { return t.size(); })
      // __getitem__ and __setitem__ are the subscript operators (operator[]).
      // To define (and overload) operator() use __call__
      .def("__getitem__",
           [](const Index<Dim>& t, const size_t i) {
             py_bindings::bounds_check(t, i);
             return t[i];
           })
      .def("__setitem__",
           [](Index<Dim>& t, const size_t i, const double v) {
             py_bindings::bounds_check(t, i);
             t[i] = v;
           })
      // Need __str__ for converting to string/printing
      .def(
          "__str__", [](const Index<Dim>& t) { return get_output(t); })
      // repr allows you to output the object in an interactive python terminal
      // using obj to get the "string REPResenting the object".
      .def(
          "__repr__", [](const Index<Dim>& t) { return get_output(t); })
      .def("product", &Index<Dim>::template product<Dim>,
           "The product of the extents.")
      .def(
          "slice_away",
          [](const Index<Dim>& t, const size_t d) {
            if constexpr (Dim == 0) {
              (void)d;
              (void)t;
              return Index<0>{};
            } else {
              return t.slice_away(d);
            }
          },
          py::arg("d"), "Slice away the specified dimension.")
      // NOLINTNEXTLINE(misc-redundant-expression)
      .def(py::self == py::self)
      // NOLINTNEXTLINE(misc-redundant-expression)
      .def(py::self != py::self)
      .def(py::pickle(
          [](const Index<Dim>& index) {
            return py::make_tuple(index.indices());
          },
          [](const py::tuple& state) {
            if (state.size() != 1) {
              throw std::runtime_error("Invalid state for Index!");
            }
            return Index<Dim>(state[0].cast<std::array<size_t, Dim>>());
          }));
  m.def("collapsed_index", &collapsed_index<Dim>, py::arg("index"),
        py::arg("extents"));
}

template <size_t Dim>
void bind_impl(py::module& m) {  // NOLINT
  bind_index_impl<Dim>(m);
}
}  // namespace

PYBIND11_MODULE(_Pybindings, m) {  // NOLINT
  enable_segfault_handler();
  py_bindings::bind_datavector(m);
  py_bindings::bind_matrix(m);
  py_bindings::bind_modalvector(m);
  bind_impl<0>(m);
  bind_impl<1>(m);
  bind_impl<2>(m);
  bind_impl<3>(m);
}
