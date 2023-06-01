// Distributed under the MIT License.
// See LICENSE.txt for details.

#include <pybind11/pybind11.h>

#include <string>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "PointwiseFunctions/Punctures/AdmIntegrals.hpp"
#include "Utilities/ErrorHandling/SegfaultHandler.hpp"

namespace py = pybind11;

namespace Punctures {

PYBIND11_MODULE(_Pybindings, m) {  // NOLINT
  enable_segfault_handler();
  py::module_::import("spectre.DataStructures");
  py::module_::import("spectre.DataStructures.Tensor");
  m.def("adm_mass_integrand",
        py::overload_cast<const Scalar<DataVector>&, const Scalar<DataVector>&,
                          const Scalar<DataVector>&>(&adm_mass_integrand),
        py::arg("field"), py::arg("alpha"), py::arg("beta"));
}

}  // namespace Punctures
