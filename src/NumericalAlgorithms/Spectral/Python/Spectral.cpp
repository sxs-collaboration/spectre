// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "NumericalAlgorithms/Spectral/Python/Spectral.hpp"

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Matrix.hpp"
#include "NumericalAlgorithms/SpatialDiscretization/Basis.hpp"
#include "NumericalAlgorithms/SpatialDiscretization/Mesh.hpp"
#include "NumericalAlgorithms/SpatialDiscretization/Quadrature.hpp"
#include "NumericalAlgorithms/Spectral/Spectral.hpp"

namespace py = pybind11;

namespace Spectral::py_bindings {

void bind_spectral(py::module& m) {
  py::enum_<SpatialDiscretization::Basis>(m, "Basis")
      .value("Legendre", SpatialDiscretization::Basis::Legendre)
      .value("Chebyshev", SpatialDiscretization::Basis::Chebyshev)
      .value("FiniteDifference",
             SpatialDiscretization::Basis::FiniteDifference);
  py::enum_<SpatialDiscretization::Quadrature>(m, "Quadrature")
      .value("Gauss", SpatialDiscretization::Quadrature::Gauss)
      .value("GaussLobatto", SpatialDiscretization::Quadrature::GaussLobatto)
      .value("CellCentered", SpatialDiscretization::Quadrature::CellCentered)
      .value("FaceCentered", SpatialDiscretization::Quadrature::FaceCentered);
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
