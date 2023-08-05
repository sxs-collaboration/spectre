// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "PointwiseFunctions/Hydro/Python/SoundSpeedSquared.hpp"

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <string>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "PointwiseFunctions/Hydro/SoundSpeedSquared.hpp"

namespace py = pybind11;

namespace py_bindings {
template <typename DataType, size_t ThermodynamicDim>
void bind_soundSpeed(py::module& m) {
  m.def("sound_speed_squared",
        static_cast<Scalar<DataType> (*)(
            const Scalar<DataType>&, const Scalar<DataType>&,
            const Scalar<DataType>&,
            const EquationsOfState::EquationOfState<true, ThermodynamicDim>&)>(
            &hydro::sound_speed_squared<DataType, ThermodynamicDim>),
        py::arg("rest_mass_density"), py::arg("specific_internal_energy"),
        py::arg("specific_enthalpy"), py::arg("equation_of_state"));
}
}  // namespace py_bindings
