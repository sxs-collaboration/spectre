// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "PointwiseFunctions/Hydro/Python/MassFlux.hpp"

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <string>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "PointwiseFunctions/Hydro/MassFlux.hpp"

namespace py = pybind11;
namespace {

template <typename DataType, size_t Dim, typename Frame>
void bind_mass_flux_impl(py::module& m) {
  m.def("mass_flux",
        static_cast<tnsr::I<DataType, Dim, Frame> (*)(
            const Scalar<DataType>&, const tnsr::I<DataType, Dim, Frame>&,
            const Scalar<DataType>&, const Scalar<DataType>&,
            const tnsr::I<DataType, Dim, Frame>&, const Scalar<DataType>&)>(
            &hydro::mass_flux<DataType, Dim, Frame>),
        py::arg("rest_mass_density"), py::arg("spatial_velocity"),
        py::arg("lorentz_factor"), py::arg("lapse"), py::arg("shift"),
        py::arg("sqrt_det_spatial_metric"));
}
}
namespace py_bindings {
void bind_mass_flux(py::module& m) {
  bind_mass_flux_impl<double, 1, Frame::Grid>(m);
  bind_mass_flux_impl<double, 2, Frame::Grid>(m);
  bind_mass_flux_impl<double, 3, Frame::Grid>(m);
  bind_mass_flux_impl<DataVector, 1, Frame::Grid>(m);
  bind_mass_flux_impl<DataVector, 2, Frame::Grid>(m);
  bind_mass_flux_impl<DataVector, 3, Frame::Grid>(m);
  bind_mass_flux_impl<double, 1, Frame::Inertial>(m);
  bind_mass_flux_impl<double, 2, Frame::Inertial>(m);
  bind_mass_flux_impl<double, 3, Frame::Inertial>(m);
  bind_mass_flux_impl<DataVector, 1, Frame::Inertial>(m);
  bind_mass_flux_impl<DataVector, 2, Frame::Inertial>(m);
  bind_mass_flux_impl<DataVector, 3, Frame::Inertial>(m);
}
}  // namespace py_bindings
