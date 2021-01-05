// Distributed under the MIT License.
// See LICENSE.txt for details.

#include <pybind11/pybind11.h>

#include "NumericalAlgorithms/Spectral/Spectral.hpp"

namespace py = pybind11;

namespace Spectral::py_bindings {

void bind_basis(py::module& m) {  // NOLINT
  py::enum_<Spectral::Basis>(m, "Basis")
      .value("Legendre", Spectral::Basis::Legendre)
      .value("Chebyshev", Spectral::Basis::Chebyshev)
      .value("FiniteDifference", Spectral::Basis::FiniteDifference);
}

void bind_quadrature(py::module& m) {  // NOLINT
  py::enum_<Spectral::Quadrature>(m, "Quadrature")
      .value("Gauss", Spectral::Quadrature::Gauss)
      .value("GaussLobatto", Spectral::Quadrature::GaussLobatto)
      .value("CellCentered", Spectral::Quadrature::CellCentered)
      .value("FaceCentered", Spectral::Quadrature::FaceCentered);
}

}  // namespace Spectral::py_bindings
