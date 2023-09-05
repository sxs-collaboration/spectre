// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "NumericalAlgorithms/Spectral/Python/Spectral.hpp"

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Matrix.hpp"
#include "NumericalAlgorithms/Spectral/Basis.hpp"
#include "NumericalAlgorithms/Spectral/Mesh.hpp"
#include "NumericalAlgorithms/Spectral/Quadrature.hpp"
#include "NumericalAlgorithms/Spectral/Spectral.hpp"

namespace py = pybind11;

namespace Spectral::py_bindings {

void bind_spectral(py::module& m) {
  py::enum_<Spectral::Basis>(m, "Basis")
      .value("Legendre", Spectral::Basis::Legendre)
      .value("Chebyshev", Spectral::Basis::Chebyshev)
      .value("FiniteDifference", Spectral::Basis::FiniteDifference)
      .value("SphericalHarmonic", Spectral::Basis::SphericalHarmonic);
  py::enum_<Spectral::Quadrature>(m, "Quadrature")
      .value("Gauss", Spectral::Quadrature::Gauss)
      .value("GaussLobatto", Spectral::Quadrature::GaussLobatto)
      .value("CellCentered", Spectral::Quadrature::CellCentered)
      .value("FaceCentered", Spectral::Quadrature::FaceCentered)
      .value("Equiangular", Spectral::Quadrature::Equiangular);
  m.def("collocation_points",
        static_cast<const DataVector& (*)(const Mesh<1>&)>(
            &Spectral::collocation_points),
        py::arg("mesh"), py::return_value_policy::reference,
        "Collocation points for a one-dimensional mesh.");
  m.def("quadrature_weights",
        static_cast<const DataVector& (*)(const Mesh<1>&)>(
            &Spectral::quadrature_weights),
        py::arg("mesh"), py::return_value_policy::reference);
  m.def("differentiation_matrix",
        static_cast<const Matrix& (*)(const Mesh<1>&)>(
            &Spectral::differentiation_matrix),
        py::arg("mesh"), py::return_value_policy::reference);
  m.def("interpolation_matrix",
        static_cast<Matrix (*)(const Mesh<1>&, const std::vector<double>&)>(
            &Spectral::interpolation_matrix),
        py::arg("mesh"), py::arg("target_points"));
  m.def("modal_to_nodal_matrix",
        static_cast<const Matrix& (*)(const Mesh<1>&)>(
            &Spectral::modal_to_nodal_matrix),
        py::arg("mesh"), py::return_value_policy::reference,
        "Transformation matrix from modal to nodal coefficients for a "
        "one-dimensional mesh.");
  m.def("nodal_to_modal_matrix",
        static_cast<const Matrix& (*)(const Mesh<1>&)>(
            &Spectral::nodal_to_modal_matrix),
        py::arg("mesh"), py::return_value_policy::reference,
        "Transformation matrix from nodal to modal coefficients for a "
        "one-dimensional mesh.");
}

}  // namespace Spectral::py_bindings
