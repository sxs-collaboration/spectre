// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "PointwiseFunctions/Hydro/Python/MassWeightedFluidItems.hpp"

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <string>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "PointwiseFunctions/Hydro/MassWeightedFluidItems.hpp"

namespace py = pybind11;

namespace py_bindings {
void bind_massWeighted(py::module& m) {
  m.def("u_lower_t", &hydro::u_lower_t<DataVector, 1, Frame::Inertial>,
        py::arg("result"), py::arg("lorentz_factor"),
        py::arg("spatial_velocity"), py::arg("spatial_metric"),
        py::arg("lapse"), py::arg("shift"));
  m.def("u_lower_t", &hydro::u_lower_t<DataVector, 2, Frame::Inertial>,
        py::arg("result"), py::arg("lorentz_factor"),
        py::arg("spatial_velocity"), py::arg("spatial_metric"),
        py::arg("lapse"), py::arg("shift"));
  m.def("u_lower_t", &hydro::u_lower_t<DataVector, 3, Frame::Inertial>,
        py::arg("result"), py::arg("lorentz_factor"),
        py::arg("spatial_velocity"), py::arg("spatial_metric"),
        py::arg("lapse"), py::arg("shift"));
  m.def("mass_weighted_internal_energy",
        &hydro::mass_weighted_internal_energy<DataVector>, py::arg("result"),
        py::arg("tilde_d"), py::arg("specific_internal_energy"));
  m.def("mass_weighted_kinetic_energy",
        &hydro::mass_weighted_kinetic_energy<DataVector>, py::arg("result"),
        py::arg("tilde_d"), py::arg("lorentz_factor"));
  m.def("tilde_d_unbound_ut_criterion",
        &hydro::tilde_d_unbound_ut_criterion<DataVector, 1, Frame::Inertial>,
        py::arg("result"), py::arg("tilde_d"), py::arg("lorentz_factor"),
        py::arg("spatial_velocity"), py::arg("spatial_metric"),
        py::arg("lapse"), py::arg("shift"));
  m.def("tilde_d_unbound_ut_criterion",
        &hydro::tilde_d_unbound_ut_criterion<DataVector, 2, Frame::Inertial>,
        py::arg("result"), py::arg("tilde_d"), py::arg("lorentz_factor"),
        py::arg("spatial_velocity"), py::arg("spatial_metric"),
        py::arg("lapse"), py::arg("shift"));
  m.def("tilde_d_unbound_ut_criterion",
        &hydro::tilde_d_unbound_ut_criterion<DataVector, 3, Frame::Inertial>,
        py::arg("result"), py::arg("tilde_d"), py::arg("lorentz_factor"),
        py::arg("spatial_velocity"), py::arg("spatial_metric"),
        py::arg("lapse"), py::arg("shift"));
  m.def("mass_weighted_coords",
        &hydro::mass_weighted_coords<::domain::ObjectLabel::None, DataVector, 1,
                                     Frame::Inertial>,
        py::arg("result"), py::arg("tilde_d"), py::arg("grid_coords"),
        py::arg("compute_coords"));
  m.def("mass_weighted_coords",
        &hydro::mass_weighted_coords<::domain::ObjectLabel::None, DataVector, 2,
                                     Frame::Inertial>,
        py::arg("result"), py::arg("tilde_d"), py::arg("grid_coords"),
        py::arg("compute_coords"));
  m.def("mass_weighted_coords",
        &hydro::mass_weighted_coords<::domain::ObjectLabel::None, DataVector, 3,
                                     Frame::Inertial>,
        py::arg("result"), py::arg("tilde_d"), py::arg("grid_coords"),
        py::arg("compute_coords"));
  m.def("mass_weighted_coords",
        &hydro::mass_weighted_coords<::domain::ObjectLabel::A, DataVector, 1,
                                     Frame::Inertial>,
        py::arg("result"), py::arg("tilde_d"), py::arg("grid_coords"),
        py::arg("compute_coords"));
  m.def("mass_weighted_coords",
        &hydro::mass_weighted_coords<::domain::ObjectLabel::A, DataVector, 2,
                                     Frame::Inertial>,
        py::arg("result"), py::arg("tilde_d"), py::arg("grid_coords"),
        py::arg("compute_coords"));
  m.def("mass_weighted_coords",
        &hydro::mass_weighted_coords<::domain::ObjectLabel::A, DataVector, 3,
                                     Frame::Inertial>,
        py::arg("result"), py::arg("tilde_d"), py::arg("grid_coords"),
        py::arg("compute_coords"));
  m.def("mass_weighted_coords",
        &hydro::mass_weighted_coords<::domain::ObjectLabel::B, DataVector, 1,
                                     Frame::Inertial>,
        py::arg("result"), py::arg("tilde_d"), py::arg("grid_coords"),
        py::arg("compute_coords"));
  m.def("mass_weighted_coords",
        &hydro::mass_weighted_coords<::domain::ObjectLabel::B, DataVector, 2,
                                     Frame::Inertial>,
        py::arg("result"), py::arg("tilde_d"), py::arg("grid_coords"),
        py::arg("compute_coords"));
  m.def("mass_weighted_coords",
        &hydro::mass_weighted_coords<::domain::ObjectLabel::B, DataVector, 3,
                                     Frame::Inertial>,
        py::arg("result"), py::arg("tilde_d"), py::arg("grid_coords"),
        py::arg("compute_coords"));
}
}  // namespace py_bindings
