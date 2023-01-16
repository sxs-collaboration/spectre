// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "IO/H5/Python/VolumeData.hpp"

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <string>

#include "DataStructures/DataVector.hpp"
#include "IO/H5/TensorData.hpp"
#include "IO/H5/VolumeData.hpp"

namespace py = pybind11;

namespace py_bindings {
void bind_h5vol(py::module& m) {
  // Wrapper for basic H5VolumeData operations
  py::class_<h5::VolumeData>(m, "H5Vol")
      .def_static("extension", &h5::VolumeData::extension)
      .def("get_header", &h5::VolumeData::get_header)
      .def("get_version", &h5::VolumeData::get_version)
      .def("get_dimension", &h5::VolumeData::get_dimension)
      .def("write_volume_data", &h5::VolumeData::write_volume_data,
           py::arg("observation_id"), py::arg("observation_value"),
           py::arg("elements"), py::arg("serialized_domain") = std::nullopt,
           py::arg("serialized_functions_of_time") = std::nullopt)
      .def("write_tensor_component",
           py::overload_cast<size_t, const std::string&, const DataVector&>(
               &h5::VolumeData::write_tensor_component),
           py::arg("observation_id"), py::arg("component_name"),
           py::arg("contiguous_tensor_data"))
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
           py::arg("observation_id"))
      .def("get_quadratures", &h5::VolumeData::get_quadratures,
           py::arg("observation_id"))
      .def("get_bases", &h5::VolumeData::get_bases, py::arg("observation_id"))
      .def("get_domain", &h5::VolumeData::get_domain, py::arg("observation_id"))
      .def("get_functions_of_time", &h5::VolumeData::get_functions_of_time,
           py::arg("observation_id"))
      .def("get_data_by_element", &h5::VolumeData::get_data_by_element,
           py::arg("start_observation_value"), py::arg("end_observation_value"),
           py::arg("components_to_retrieve") = std::nullopt);
  m.def("offset_and_length_for_grid", &h5::offset_and_length_for_grid,
        py::arg("grid_name"), py::arg("all_grid_names"),
        py::arg("all_extents"));
}
}  // namespace py_bindings
