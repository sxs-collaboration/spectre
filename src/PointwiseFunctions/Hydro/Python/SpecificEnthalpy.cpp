// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "PointwiseFunctions/Hydro/Python/SpecificEnthalpy.hpp"

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <string>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "PointwiseFunctions/Hydro/SpecificEnthalpy.hpp"

namespace py = pybind11;
namespace{

template <typename DataType>
void bind_specific_enthalpy_impl(py::module& m) {
  m.def("relativistic_specific_enthalpy",
        static_cast<Scalar<DataType> (*)(const Scalar<DataType>&,
                                         const Scalar<DataType>&,
                                         const Scalar<DataType>&)>(
            &hydro::relativistic_specific_enthalpy<DataType>),
        py::arg("rest_mass_density"), py::arg("specific_internal_energy"),
        py::arg("pressure"));
}
}
namespace py_bindings {
void bind_specific_enthalpy(py::module& m) {
    bind_specific_enthalpy_impl<double>(m);
    bind_specific_enthalpy_impl<DataVector>(m);
}
}  // namespace py_bindings
