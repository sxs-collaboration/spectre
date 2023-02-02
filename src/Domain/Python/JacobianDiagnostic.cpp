// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Domain/Python/JacobianDiagnostic.hpp"

#include <cstddef>
#include <pybind11/pybind11.h>
#include <string>
#include <type_traits>

#include "DataStructures/Tensor/TypeAliases.hpp"
#include "Domain/JacobianDiagnostic.hpp"
#include "NumericalAlgorithms/Spectral/Mesh.hpp"
#include "Utilities/Gsl.hpp"

namespace py = pybind11;

namespace domain::py_bindings {

namespace {
template <size_t Dim, typename TargetFrame>
void bind_jacobian_diagnostic_impl(py::module& m) {  // NOLINT
  m.def(
      "jacobian_diagnostic",
      static_cast<tnsr::i<DataVector, Dim, typename Frame::ElementLogical> (*)(
          const ::Jacobian<DataVector, Dim, Frame::ElementLogical,
                           TargetFrame>&,
          const tnsr::I<DataVector, Dim, TargetFrame>&, const ::Mesh<Dim>&)>(
          &::domain::jacobian_diagnostic<Dim, TargetFrame>),
      py::arg("jacobian"),
      py::arg(std::is_same_v<TargetFrame, Frame::Inertial> ? "inertial_coords"
                                                           : "grid_coords"),
      py::arg("mesh"));
}
}  // namespace

void bind_jacobian_diagnostic(py::module& m) {  // NOLINT
  bind_jacobian_diagnostic_impl<1, Frame::Grid>(m);
  bind_jacobian_diagnostic_impl<2, Frame::Grid>(m);
  bind_jacobian_diagnostic_impl<3, Frame::Grid>(m);
  bind_jacobian_diagnostic_impl<1, Frame::Inertial>(m);
  bind_jacobian_diagnostic_impl<2, Frame::Inertial>(m);
  bind_jacobian_diagnostic_impl<3, Frame::Inertial>(m);
}

}  // namespace domain::py_bindings
