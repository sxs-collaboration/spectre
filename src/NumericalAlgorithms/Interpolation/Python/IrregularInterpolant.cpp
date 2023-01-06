// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "NumericalAlgorithms/Interpolation/Python/IrregularInterpolant.hpp"

#include <array>
#include <cstddef>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "DataStructures/DataVector.hpp"
#include "NumericalAlgorithms/Interpolation/IrregularInterpolant.hpp"
#include "NumericalAlgorithms/Spectral/Mesh.hpp"

namespace py = pybind11;

namespace intrp::py_bindings {
namespace {
template <size_t Dim>
void bind_irregular_impl(py::module& m) {  // NOLINT
  py::class_<Irregular<Dim>>(m,
                             ("Irregular" + std::to_string(Dim) + "D").c_str())
      .def(py::init(
               [](const Mesh<Dim>& source_mesh,
                  const std::array<DataVector, Dim>& target_logical_coords) {
                 return Irregular<Dim>{
                     source_mesh,
                     tnsr::I<DataVector, Dim, Frame::ElementLogical>{
                         target_logical_coords}};
               }),
           py::arg("source_mesh"), py::arg("target_logical_coords"))
      .def("interpolate",
           static_cast<DataVector (Irregular<Dim>::*)(const DataVector&) const>(
               &Irregular<Dim>::interpolate),
           py::arg("input"));
}
}  // namespace

void bind_irregular(py::module& m) {
  bind_irregular_impl<1>(m);
  bind_irregular_impl<2>(m);
  bind_irregular_impl<3>(m);
}

}  // namespace intrp::py_bindings
