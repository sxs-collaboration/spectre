// Distributed under the MIT License.
// See LICENSE.txt for details.

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <string>

#include "DataStructures/DataVector.hpp"
#include "IO/H5/VolumeData.hpp"

namespace py = pybind11;

namespace py_bindings {
void bind_h5vol(py::module& m) {  // NOLINT
  // Wrapper for basic H5VolumeData operations
  py::class_<h5::VolumeData>(m, "H5Vol")
      .def_static("extension", &h5::VolumeData::extension)
      .def("get_header", &h5::VolumeData::get_header)
      .def("get_version", &h5::VolumeData::get_version)
      .def("list_observation_ids", &h5::VolumeData::list_observation_ids)
      .def("get_observation_value", &h5::VolumeData::get_observation_value,
           py::arg("observation_id"))
      .def("get_grid_names", &h5::VolumeData::get_grid_names,
           py::arg("observation_id"))
      .def("list_tensor_components", &h5::VolumeData::list_tensor_components,
           py::arg("observation_id"))
      .def("get_tensor_component", &h5::VolumeData::get_tensor_component,
           py::arg("observation_id"), py::arg("tensor_component"))
      .def("get_extents", &h5::VolumeData::get_extents,
           py::arg("observation_id"));
  m.def("offset_and_length_for_grid", &h5::offset_and_length_for_grid,
        py::arg("grid_name"), py::arg("all_grid_names"),
        py::arg("all_extents"));
}
}  // namespace py_bindings
