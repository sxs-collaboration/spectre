// Distributed under the MIT License.
// See LICENSE.txt for details.

#include <cstddef>
#include <pybind11/pybind11.h>
#include <type_traits>

#include "DataStructures/Tensor/TypeAliases.hpp"
#include "PointwiseFunctions/GeneralRelativity/Psi4Real.hpp"

namespace py = pybind11;

namespace GeneralRelativity {

PYBIND11_MODULE(_PyGeneralRelativity, m) {  // NOLINT
  py::module_::import("spectre.DataStructures");
  py::module_::import("spectre.DataStructures.Tensor");
  py::module_::import("spectre.Spectral");
  m.def("psi4real",
        static_cast<Scalar<DataVector> (*)(
            const tnsr::ii<DataVector, 3>&, const tnsr::ii<DataVector, 3>&,
            const tnsr::ijj<DataVector, 3>&, const tnsr::ii<DataVector, 3>&,
            const tnsr::II<DataVector, 3>&, const tnsr::I<DataVector, 3>&)>(
            &::gr::psi_4_real),
        py::arg("spatial_ricci"), py::arg("extrinsic_curvature"),
        py::arg("cov_deriv_extrinsic_curvature"), py::arg("spatial_metric"),
        py::arg("inverse_spatial_metric"), py::arg("inertial_coords"));
}
}  // namespace GeneralRelativity
