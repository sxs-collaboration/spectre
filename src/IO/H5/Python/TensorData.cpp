// Distributed under the MIT License.
// See LICENSE.txt for details.

#include <pybind11/operators.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <string>
#include <vector>

#include "DataStructures/DataVector.hpp"
#include "IO/H5/TensorData.hpp"
#include "NumericalAlgorithms/Spectral/Spectral.hpp"
#include "Utilities/GetOutput.hpp"
#include "Utilities/StdHelpers.hpp"

namespace py = pybind11;

namespace py_bindings {
void bind_tensordata(py::module& m) {  // NOLINT
  // Wrapper for TensorComponent
  py::class_<TensorComponent>(m, "TensorComponent")
      .def(py::init<std::string, DataVector>(), py::arg("name"),
           py::arg("data"))
      .def_readwrite("name", &TensorComponent::name)
      .def_readwrite("data", &TensorComponent::data)
      .def("__str__", get_output<TensorComponent>)
      .def("__repr__", get_output<TensorComponent>)
      // NOLINTNEXTLINE(misc-redundant-expression)
      .def(py::self == py::self)
      // NOLINTNEXTLINE(misc-redundant-expression)
      .def(py::self != py::self);

  py::class_<ElementVolumeData>(m, "ElementVolumeData")
      .def(py::init<std::string, std::vector<TensorComponent>,
                    std::vector<size_t>, std::vector<Spectral::Basis>,
                    std::vector<Spectral::Quadrature>>(),
           py::arg("element_name"), py::arg("components"), py::arg("extents"),
           py::arg("basis"), py::arg("quadrature"))
      .def_readwrite("element_name", &ElementVolumeData::element_name)
      .def_readwrite("tensor_components", &ElementVolumeData::tensor_components)
      .def_readwrite("extents", &ElementVolumeData::extents)
      .def_readwrite("basis", &ElementVolumeData::basis)
      .def_readwrite("quadrature", &ElementVolumeData::quadrature)
      // NOLINTNEXTLINE(misc-redundant-expression)
      .def(py::self == py::self)
      // NOLINTNEXTLINE(misc-redundant-expression)
      .def(py::self != py::self);
}
}  // namespace py_bindings
