// Distributed under the MIT License.
// See LICENSE.txt for details.

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "DataStructures/Tensor/Tensor.hpp"
#include "NumericalAlgorithms/SphericalHarmonics/Strahlkorper.hpp"
#include "ParallelAlgorithms/ApparentHorizonFinder/FastFlow.hpp"
#include "Utilities/ErrorHandling/SegfaultHandler.hpp"

namespace py = pybind11;

PYBIND11_MODULE(_Pybindings, m) {  // NOLINT
  enable_segfault_handler();
  py::module_::import("spectre.SphericalHarmonics");
  py::enum_<FastFlow::FlowType>(m, "FlowType")
      .value("Jacobi", FastFlow::FlowType::Jacobi)
      .value("Curvature", FastFlow::FlowType::Curvature)
      .value("Fast", FastFlow::FlowType::Fast);
  py::enum_<FastFlow::Status>(m, "Status")
      .value("SuccessfulIteration", FastFlow::Status::SuccessfulIteration)
      .value("AbsTol", FastFlow::Status::AbsTol)
      .value("TruncationTol", FastFlow::Status::TruncationTol)
      .value("MaxIts", FastFlow::Status::MaxIts)
      .value("NegativeRadius", FastFlow::Status::NegativeRadius)
      .value("DivergenceError", FastFlow::Status::DivergenceError)
      .value("InterpolationFailure", FastFlow::Status::InterpolationFailure);
  py::class_<FastFlow::IterInfo>(m, "IterInfo")
      .def_readonly("iteration", &FastFlow::IterInfo::iteration)
      .def_readonly("r_min", &FastFlow::IterInfo::r_min)
      .def_readonly("r_max", &FastFlow::IterInfo::r_max)
      .def_readonly("min_residual", &FastFlow::IterInfo::min_residual)
      .def_readonly("max_residual", &FastFlow::IterInfo::max_residual)
      .def_readonly("residual_ylm", &FastFlow::IterInfo::residual_ylm)
      .def_readonly("residual_mesh", &FastFlow::IterInfo::residual_mesh);
  py::class_<FastFlow>(m, "FastFlow")
      .def(py::init<FastFlow::FlowType, double, double, double, double, double,
                    size_t, size_t>(),
           py::arg("flow_type"), py::arg("alpha"), py::arg("beta"),
           py::arg("abs_tol"), py::arg("truncation_tol"),
           py::arg("divergence_tol"), py::arg("divergence_iter"),
           py::arg("max_its"))
      .def(
          "iterate_horizon_finder",
          [](FastFlow& fast_flow,
             ylm::Strahlkorper<Frame::Inertial>& current_strahlkorper,
             const tnsr::II<DataVector, 3>& upper_spatial_metric,
             const tnsr::ii<DataVector, 3>& extrinsic_curvature,
             const tnsr::Ijj<DataVector, 3>& christoffel_2nd_kind) {
            return fast_flow.iterate_horizon_finder<Frame::Inertial>(
                make_not_null(&current_strahlkorper), upper_spatial_metric,
                extrinsic_curvature, christoffel_2nd_kind);
          },
          py::arg("current_strahlkorper"), py::arg("upper_spatial_metric"),
          py::arg("extrinsic_curvature"), py::arg("christoffel_2nd_kind"))
      .def("current_l_mesh", &FastFlow::current_l_mesh<Frame::Inertial>)
      .def("reset_for_next_find", &FastFlow::reset_for_next_find);
}
