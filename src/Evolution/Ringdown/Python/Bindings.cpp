// Distributed under the MIT License.
// See LICENSE.txt for details.

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "Domain/Creators/RegisterDerivedWithCharm.hpp"
#include "Domain/Creators/TimeDependence/RegisterDerivedWithCharm.hpp"
#include "Domain/FunctionsOfTime/RegisterDerivedWithCharm.hpp"
#include "Evolution/Ringdown/MinimumAhCExcisionRadius.hpp"
#include "Evolution/Ringdown/StrahlkorperCoefsAndCenters.hpp"
#include "Utilities/ErrorHandling/SegfaultHandler.hpp"
#include "Utilities/Serialization/RegisterDerivedClassesWithCharm.hpp"
#include "Utilities/TMPL.hpp"

namespace py = pybind11;

namespace evolution::Ringdown::py_bindings {  // NOLINT
// Silence warning about no previous declaration
void bind_strahlkorper_coefs_and_centers(py::module& m);

void bind_strahlkorper_coefs_and_centers(py::module& m) {
  domain::creators::register_derived_with_charm();
  domain::creators::time_dependence::register_derived_with_charm();
  domain::FunctionsOfTime::register_derived_with_charm();

  m.def("strahlkorper_coefs_and_centers",
        &evolution::Ringdown::strahlkorper_coefs_and_centers,
        py::arg("path_to_volume_data"), py::arg("volume_subfile_name"),
        py::arg("path_to_horizons_h5"), py::arg("surface_subfile_name"),
        py::arg("requested_number_of_times_from_end"), py::arg("match_time"),
        py::arg("settling_timescale"), py::arg("exp_func_and_2_derivs"),
        py::arg("exp_outer_bdry_func_and_2_derivs"),
        py::arg("rot_func_and_2_derivs"), py::arg("trans_func_and_2_derivs"));
}

void bind_minimum_ahc_excision_radius(py::module& m);

void bind_minimum_ahc_excision_radius(py::module& m) {
  domain::creators::register_derived_with_charm();
  domain::creators::time_dependence::register_derived_with_charm();
  domain::FunctionsOfTime::register_derived_with_charm();

  m.def("minimum_ahc_excision_radius",
        &evolution::Ringdown::minimum_ahc_excision_radius,
        py::arg("path_to_volume_data"), py::arg("volume_subfile_name"),
        py::arg("path_to_horizons_h5"), py::arg("surface_subfile_name"),
        py::arg("path_to_ahc_distorted_h5"),
        py::arg("ahc_distorted_subfile_names"), py::arg("match_time"),
        py::arg("settling_timescale"), py::arg("excision_a_radius"),
        py::arg("excision_b_radius"), py::arg("excision_a_center"),
        py::arg("excision_b_center"), py::arg("exp_func_and_2_derivs"),
        py::arg("exp_outer_bdry_func_and_2_derivs"),
        py::arg("rot_func_and_2_derivs"), py::arg("trans_func_and_2_derivs"),
        py::arg("match_time_tol"));
}

}  // namespace evolution::Ringdown::py_bindings

PYBIND11_MODULE(_Pybindings, m) {  // NOLINT
  enable_segfault_handler();
  // So return types are converted to DataVectors
  py::module_::import("spectre.DataStructures");
  evolution::Ringdown::py_bindings::bind_strahlkorper_coefs_and_centers(m);
  evolution::Ringdown::py_bindings::bind_minimum_ahc_excision_radius(m);
}
