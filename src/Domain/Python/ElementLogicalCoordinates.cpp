// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Domain/Python/ElementLogicalCoordinates.hpp"

#include <array>
#include <cstddef>
#include <memory>
#include <optional>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <string>
#include <unordered_map>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/IdPair.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Domain/BlockLogicalCoordinates.hpp"
#include "Domain/Domain.hpp"
#include "Domain/ElementLogicalCoordinates.hpp"
#include "Domain/FunctionsOfTime/FunctionOfTime.hpp"
#include "Domain/Structure/BlockId.hpp"
#include "Domain/Structure/ElementId.hpp"
#include "Utilities/GetOutput.hpp"
#include "Utilities/MakeArray.hpp"

namespace py = pybind11;

namespace domain::py_bindings {

namespace {
template <size_t Dim>
void bind_element_logical_coordinates_impl(py::module& m) {  // NOLINT
  py::class_<ElementLogicalCoordHolder<Dim>>(
      m, ("ElementLogicalCoordHolder" + get_output(Dim) + "D").c_str())
      .def_readonly("element_logical_coords",
                    &ElementLogicalCoordHolder<Dim>::element_logical_coords)
      .def_readonly("offsets", &ElementLogicalCoordHolder<Dim>::offsets);
  m.def("element_logical_coordinates", &element_logical_coordinates<Dim>,
        py::arg("element_ids"), py::arg("block_coord_holders"));
}
}  // namespace

void bind_element_logical_coordinates(py::module& m) {  // NOLINT
  bind_element_logical_coordinates_impl<1>(m);
  bind_element_logical_coordinates_impl<2>(m);
  bind_element_logical_coordinates_impl<3>(m);
}

}  // namespace domain::py_bindings
