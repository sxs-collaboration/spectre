// Distributed under the MIT License.
// See LICENSE.txt for details.

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <string>
#include <vector>

#include "IO/H5/Dat.hpp"
#include "IO/H5/File.hpp"
#include "IO/H5/VolumeData.hpp"

namespace py = pybind11;

namespace py_bindings {
void bind_h5file(py::module& m) {  // NOLINT
  // Wrapper for basic H5File operations
  using h5_file_rw = h5::H5File<h5::AccessType::ReadWrite>;
  py::class_<h5_file_rw>(m, "H5File")
      .def(py::init<std::string, bool>(), py::arg("file_name"),
           py::arg("append_to_file") = false)
      .def("name", &h5_file_rw::name)
      .def(
          "get_dat",
          +[](const h5_file_rw& f, const std::string& path) -> const h5::Dat& {
            return f.get<h5::Dat>(path);
          },
          py::return_value_policy::reference, py::arg("path"))
      .def(
          "insert_dat",
          +[](h5_file_rw& f, const std::string& path,
              const std::vector<std::string>& legend, const uint32_t version) {
            f.insert<h5::Dat>(path, legend, version);
          },
          py::arg("path"), py::arg("legend"), py::arg("version"))
      .def("close", &h5_file_rw::close_current_object)
      .def("groups", &h5_file_rw::groups)
      .def(
          "get_vol",
          +[](const h5_file_rw& f, const std::string& path)
              -> const h5::VolumeData& { return f.get<h5::VolumeData>(path); },
          py::return_value_policy::reference, py::arg("path"))
      .def(
          "insert_vol",
          +[](h5_file_rw& f, const std::string& path, const uint32_t version) {
            f.insert<h5::VolumeData>(path, version);
          },
          py::arg("path"), py::arg("version"));
}
}  // namespace py_bindings
