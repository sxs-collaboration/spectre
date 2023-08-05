// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "PointwiseFunctions/Hydro/Python/LorentzFactor.hpp"

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <string>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "PointwiseFunctions/Hydro/LorentzFactor.hpp"

namespace py = pybind11;

namespace py_bindings {
template <typename DataType>
void bind_lorentz_impl(py::module& m) {
  m.def("lorentz_factor",
        static_cast<Scalar<DataType> (*)(const Scalar<DataType>&)>(
            &hydro::lorentz_factor<DataType>),
        py::arg("spatial_velocity_squared"));
}
template <typename DataType, size_t Dim, typename Frame>
void bind_lorentz_factor_impl(py::module& m) {
  m.def("lorentz_factor",
        static_cast<Scalar<DataType> (*)(const tnsr::I<DataType, Dim, Frame>&,
                                         const tnsr::i<DataType, Dim, Frame>&)>(
            &hydro::lorentz_factor<DataType, Dim, Frame>),
        py::arg("spatial_velocity"), py::arg("spatial_velocity_form"));
}
void bind_lorentz(py::module& m) {
  bind_lorentz_impl<double>(m);
  bind_lorentz_impl<DataVector>(m);
}
void bind_lorentz_factor(py::module& m) {
  bind_lorentz_factor_impl<double, 1, Frame::Grid>(m);
  bind_lorentz_factor_impl<double, 2, Frame::Grid>(m);
  bind_lorentz_factor_impl<double, 3, Frame::Grid>(m);
  bind_lorentz_factor_impl<DataVector, 1, Frame::Grid>(m);
  bind_lorentz_factor_impl<DataVector, 2, Frame::Grid>(m);
  bind_lorentz_factor_impl<DataVector, 3, Frame::Grid>(m);
  bind_lorentz_factor_impl<double, 1, Frame::Inertial>(m);
  bind_lorentz_factor_impl<double, 2, Frame::Inertial>(m);
  bind_lorentz_factor_impl<double, 3, Frame::Inertial>(m);
  bind_lorentz_factor_impl<DataVector, 1, Frame::Inertial>(m);
  bind_lorentz_factor_impl<DataVector, 2, Frame::Inertial>(m);
  bind_lorentz_factor_impl<DataVector, 3, Frame::Inertial>(m);
}
}  // namespace py_bindings
