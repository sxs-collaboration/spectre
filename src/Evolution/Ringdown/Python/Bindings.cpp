// Distributed under the MIT License.
// See LICENSE.txt for details.

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "Domain/Creators/RegisterDerivedWithCharm.hpp"
#include "Domain/Creators/Sphere.hpp"
#include "Domain/Creators/TimeDependence/RegisterDerivedWithCharm.hpp"
#include "Domain/FunctionsOfTime/RegisterDerivedWithCharm.hpp"
#include "Evolution/Ringdown/StrahlkorperCoefsInRingdownDistortedFrame.hpp"
#include "Utilities/ErrorHandling/SegfaultHandler.hpp"
#include "Utilities/Serialization/RegisterDerivedClassesWithCharm.hpp"
#include "Utilities/TMPL.hpp"

namespace py = pybind11;

namespace evolution::Ringdown::py_bindings {  // NOLINT
// Silence warning about no previous declaration
void bind_strahlkorper_coefs_in_ringdown_distorted_frame(py::module& m);

void bind_strahlkorper_coefs_in_ringdown_distorted_frame(py::module& m) {
  domain::creators::register_derived_with_charm();
  domain::creators::time_dependence::register_derived_with_charm();
  domain::FunctionsOfTime::register_derived_with_charm();

  m.def("strahlkorper_coefs_in_ringdown_distorted_frame",
        &evolution::Ringdown::strahlkorper_coefs_in_ringdown_distorted_frame,
        py::arg("path_to_horizons_h5"), py::arg("surface_subfile_name"),
        py::arg("requested_number_of_times_from_end"), py::arg("match_time"),
        py::arg("settling_timescale"), py::arg("exp_func_and_2_derivs"),
        py::arg("exp_outer_bdry_func_and_2_derivs"),
        py::arg("rot_func_and_2_derivs"));
}
}  // namespace evolution::Ringdown::py_bindings

PYBIND11_MODULE(_Pybindings, m) {  // NOLINT
  enable_segfault_handler();
  // So return types are converted to DataVectors
  py::module_::import("spectre.DataStructures");
  evolution::Ringdown::py_bindings::
      bind_strahlkorper_coefs_in_ringdown_distorted_frame(m);
}
