// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "PointwiseFunctions/Hydro/Python/ComovingMagneticField.hpp"

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <string>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "PointwiseFunctions/Hydro/ComovingMagneticField.hpp"

namespace py = pybind11;

namespace py_bindings {
template <typename DataType>
void bind_comovingMF_impl(py::module& m) {
  // Wrapper for calculating Co-moving magnetic fields
  m.def("comoving_magnetic_field_one_form",
        static_cast<tnsr::a<DataType, 3> (*)(
            const tnsr::i<DataType, 3>&, const tnsr::i<DataType, 3>&,
            const Scalar<DataType>&, const Scalar<DataType>&,
            const tnsr::I<DataType, 3>&, const Scalar<DataType>&)>(
            &hydro::comoving_magnetic_field_one_form<DataType>),
        py::arg("spatial_velocity_one_form"),
        py::arg("magnetic_field_one_form"),
        py::arg("magnetic_field_dot_spatial_velocity"),
        py::arg("lorentz_factor"), py::arg("shift"), py::arg("lapse"));
  m.def("comoving_magnetic_field_squared",
        static_cast<Scalar<DataType> (*)(const Scalar<DataType>&,
                                         const Scalar<DataType>&,
                                         const Scalar<DataType>&)>(
            &hydro::comoving_magnetic_field_squared<DataType>),
        py::arg("magnetic_field_squared"),
        py::arg("magnetic_field_dot_spatial_velocity"),
        py::arg("lorentz_factor"));
}
void bind_comovingMF(py::module& m) {
  bind_comovingMF_impl<double>(m);
  bind_comovingMF_impl<DataVector>(m);
}

}  // namespace py_bindings
