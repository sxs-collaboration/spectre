// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Domain/Creators/TimeDependentOptions/Python/GridCenters.hpp"

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <string>

#include "Domain/Creators/TimeDependentOptions/BinaryCompactObject.hpp"
#include "Domain/Creators/TimeDependentOptions/GridCenters.hpp"

namespace py = pybind11;

namespace domain::creators::time_dependent_options::py_bindings {
void bind_grid_centers(py::module& m) {
  py::class_<time_dependent_options::GridCentersOptions>(m,
                                                         "GridCentersOptions")
      .def(py::init<std::string, std::optional<double>>(),
           py::arg("spec_evolution_parameters_perl_file"),
           py::arg("in_scale_inspiral_rate_by"));
}
}  // namespace domain::creators::time_dependent_options::py_bindings
