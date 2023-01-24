// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "IO/H5/Python/File.hpp"

#include <boost/algorithm/string/join.hpp>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <string>
#include <vector>

#include "IO/H5/Dat.hpp"
#include "IO/H5/File.hpp"
#include "IO/H5/VolumeData.hpp"
#include "Utilities/MakeString.hpp"

namespace py = pybind11;

namespace py_bindings {
namespace {
template <h5::AccessType Access_t>
void bind_h5file_impl(py::module& m) {  // NOLINT
  // Wrapper for basic H5File operations
  // Manually check for existence as we currently can not propagate exceptions
  // that can be caught by pybind, see issue #2312
  using H5File = h5::H5File<Access_t>;
  auto bind_h5_file =
      py::class_<H5File>(
          m,
          (std::string("_H5File") +
           (Access_t == h5::AccessType::ReadWrite ? "ReadWrite" : "ReadOnly"))
              .c_str())
          .def("name", &H5File::name)
          .def(
              "get_dat",
              [](const H5File& f, const std::string& path) -> const h5::Dat& {
                return f.template get<h5::Dat>(path);
              },
              py::return_value_policy::reference, py::arg("path"))
          .def("close_current_object", &H5File::close_current_object)
          .def("close", &H5File::close)
          .def("all_files",
               [](const H5File& f) -> const std::vector<std::string> {
                 return f.template all_files<void>("/");
               })
          .def("all_dat_files",
               [](const H5File& f) -> const std::vector<std::string> {
                 return f.template all_files<h5::Dat>("/");
               })
          .def("all_vol_files",
               [](const H5File& f) -> const std::vector<std::string> {
                 return f.template all_files<h5::VolumeData>("/");
               })
          .def("groups", &H5File::groups)
          .def(
              "get_vol",
              [](const H5File& f,
                 const std::string& path) -> const h5::VolumeData& {
                return f.template get<h5::VolumeData>(path);
              },
              py::return_value_policy::reference, py::arg("path"))
          .def("input_source", &H5File::input_source)
          .def("__enter__", [](H5File& file) -> H5File& { return file; })
          .def("__exit__", [](H5File& f, const py::object& /* exception_type */,
                              const py::object& /* val */,
                              const py::object& /* traceback */) {
            f.close();
          });

  if constexpr (Access_t == h5::AccessType::ReadWrite) {
    bind_h5_file
        .def(
            "insert_dat",
            [](H5File& f, const std::string& path,
               const std::vector<std::string>& legend,
               const uint32_t version) -> h5::Dat& {
              return f.template insert<h5::Dat>(path, legend, version);
            },
            py::return_value_policy::reference, py::arg("path"),
            py::arg("legend"), py::arg("version"))
        .def(
            "insert_vol",
            [](H5File& f, const std::string& path,
               const uint32_t version) -> h5::VolumeData& {
              return f.template insert<h5::VolumeData>(path, version);
            },
            py::return_value_policy::reference, py::arg("path"),
            py::arg("version"));
  }
}
}  // namespace

void bind_h5file(py::module& m) {
  bind_h5file_impl<h5::AccessType::ReadOnly>(m);
  bind_h5file_impl<h5::AccessType::ReadWrite>(m);

  m.def(
      "H5File",
      [](const std::string& file_name, const std::string& mode) {
        if (mode == "r") {
          return py::cast(
              h5::H5File<h5::AccessType::ReadOnly>{file_name, false});
        }
        return py::cast(h5::H5File<h5::AccessType::ReadWrite>{
            file_name, (mode == "a" or mode == "r+")});
      },
      py::arg("file_name"), py::arg("mode") = "r",
      "Open an H5File object\n\nfile_name: the name of the H5File to "
      "open\nmode: mode to open the file. Available modes are 'r', 'r+', 'w-', "
      "'x', and 'a'. For details see "
      "https://docs.h5py.org/en/stable/high/file.html");
}
}  // namespace py_bindings
