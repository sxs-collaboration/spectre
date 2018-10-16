// Distributed under the MIT License.
// See LICENSE.txt for details.

#include <boost/python.hpp>
#include <boost/python/reference_existing_object.hpp>
#include <boost/python/return_value_policy.hpp>
#include <boost/python/suite/indexing/vector_indexing_suite.hpp>
#include <hdf5.h>
#include <sstream>
#include <string>
#include <utility>

#include "IO/H5/Dat.hpp"
#include "IO/H5/File.hpp"
#include "IO/H5/Header.hpp"
#include "IO/H5/OpenGroup.hpp"
#include "IO/H5/Type.hpp"
#include "IO/H5/Version.hpp"
#include "PythonBindings/VectorPyList.hpp"

namespace bp = boost::python;

namespace py_bindings {
void bind_h5file() {
  // Wrapper for basic H5File operations
  bp::class_<h5::H5File<h5::AccessType::ReadWrite>, boost::noncopyable>(
      "H5File", bp::init<std::string, bool>())
      .def("name",
           +[](const h5::H5File<h5::AccessType::ReadWrite>& f) {
             return f.name();
           })
      .def("get_dat",
           +[](const h5::H5File<h5::AccessType::ReadWrite>& f,
               const bp::str& path) -> const h5::Dat& {
             const auto& dat_file =
                 f.get<h5::Dat>(bp::extract<std::string>(path));
             return dat_file;
           },
           bp::return_value_policy<bp::reference_existing_object>())
      .def("insert_dat",
           +[](h5::H5File<h5::AccessType::ReadWrite>& f, const bp::str& path,
               const bp::list& legend, uint32_t version) {
             f.insert<h5::Dat>(bp::extract<std::string>(path),
                               py_list_to_std_vector<std::string>(legend),
                               version);
           })
      .def("close", +[](const h5::H5File<h5::AccessType::ReadWrite>& f) {
        f.close_current_object();
      });
}
}  // namespace py_bindings
