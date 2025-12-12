// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Domain/Creators/TimeDependentOptions/Python/ShapeMap.hpp"

#include <array>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <string>
#include <vector>

#include "Domain/Creators/TimeDependentOptions/BinaryCompactObject.hpp"
#include "Domain/Creators/TimeDependentOptions/ShapeMap.hpp"
#include "Domain/Structure/ObjectLabel.hpp"
#include "Utilities/GetOutput.hpp"

namespace py = pybind11;

namespace domain::creators::time_dependent_options::py_bindings {
namespace {
void bind_kerrschild_from_boyer_lindquist_impl(py::module& m) {
  py::class_<time_dependent_options::KerrSchildFromBoyerLindquist>(
      m, "KerrSchildFromBoyerLindquist")
      .def(py::init<double, std::array<double, 3>>(), py::arg("mass"),
           py::arg("spin"));
}

void bind_ylms_from_file_impl(py::module& m) {
  py::class_<time_dependent_options::YlmsFromFile>(m, "YlmsFromFile")
      .def(py::init<std::string, std::vector<std::string>, double,
                    std::optional<double>, bool, bool>(),
           py::arg("h5_filename"), py::arg("subfile_names"),
           py::arg("match_time"), py::arg("match_time_epsilon") = std::nullopt,
           py::arg("set_l1_coefs_to_zero"), py::arg("check_frame"));
}

void bind_ylms_from_spec_impl(py::module& m) {
  py::class_<time_dependent_options::YlmsFromSpEC>(m, "YlmsFromSpEC")
      .def(py::init<std::string, double, std::optional<double>, bool>(),
           py::arg("dat_filename"), py::arg("match_time"),
           py::arg("match_time_epsilon"), py::arg("set_l1_coefs_to_zero"));
}

template <domain::ObjectLabel Object>
void bind_shape_map_impl(py::module& m) {
  py::class_<time_dependent_options::ShapeMapOptions<true, Object>>(
      m, ("ShapeMapOptions" + get_output(name(Object))).c_str())
      .def(
          py::init<size_t,
                   std::optional<std::variant<
                       domain::creators::time_dependent_options::
                           KerrSchildFromBoyerLindquist,
                       domain::creators::time_dependent_options::YlmsFromFile,
                       domain::creators::time_dependent_options::YlmsFromSpEC>>,
                   std::optional<std::array<double, 3>>, double, bool>(),
          py::arg("l_max"), py::arg("initial_values"),
          py::arg("initial_size_values") = std::nullopt,
          py::arg("coefficient_truncation_limit") = 0.0,
          py::arg("transition_ends_at_cube") = true);
}
}  // namespace

void bind_shape_map(py::module& m) {
  py_bindings::bind_kerrschild_from_boyer_lindquist_impl(m);
  py_bindings::bind_shape_map_impl<ObjectLabel::A>(m);
  py_bindings::bind_shape_map_impl<ObjectLabel::B>(m);
  py_bindings::bind_shape_map_impl<ObjectLabel::C>(m);
  py_bindings::bind_shape_map_impl<ObjectLabel::None>(m);
  py_bindings::bind_ylms_from_file_impl(m);
  py_bindings::bind_ylms_from_spec_impl(m);
}
}  // namespace domain::creators::time_dependent_options::py_bindings
