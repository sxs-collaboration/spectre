// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "PointwiseFunctions/Hydro/Python/SoundSpeedSquared.hpp"

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <string>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "PointwiseFunctions/Hydro/EquationsOfState/EquationOfState.hpp"
#include "PointwiseFunctions/Hydro/SoundSpeedSquared.hpp"

namespace py = pybind11;
namespace{

template <typename DataType, size_t ThermodynamicDim>
void bind_sound_speed_impl(py::module& m) {
  m.def("sound_speed_squared",
        static_cast<Scalar<DataType> (*)(
            const Scalar<DataType>&, const Scalar<DataType>&,
            const Scalar<DataType>&,
            const EquationsOfState::EquationOfState<true, ThermodynamicDim>&)>(
            &hydro::sound_speed_squared<DataType, ThermodynamicDim>),
        py::arg("rest_mass_density"), py::arg("specific_internal_energy"),
        py::arg("specific_enthalpy"), py::arg("equation_of_state"));
}
}
namespace py_bindings {
void bind_sound_speed(py::module& m) {
  bind_sound_speed_impl<double, 1>(m);
  bind_sound_speed_impl<double, 2>(m);
  bind_sound_speed_impl<DataVector, 1>(m);
  bind_sound_speed_impl<DataVector, 2>(m);
}

}  // namespace py_bindings
