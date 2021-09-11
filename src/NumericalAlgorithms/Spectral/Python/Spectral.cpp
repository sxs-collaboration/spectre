// Distributed under the MIT License.
// See LICENSE.txt for details.

#include <pybind11/pybind11.h>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Matrix.hpp"
#include "NumericalAlgorithms/Spectral/Mesh.hpp"
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

void bind_collocation_points(py::module& m) {  // NOLINT
  m.def("collocation_points",
        static_cast<const DataVector& (*)(const Mesh<1>&)>(
            &Spectral::collocation_points),
        py::arg("mesh"), py::return_value_policy::reference,
        "Collocation points for a one-dimensional mesh.");
}

}  // namespace Spectral::py_bindings
