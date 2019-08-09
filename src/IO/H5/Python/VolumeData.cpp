// Distributed under the MIT License.
// See LICENSE.txt for details.

#include <Python.h>
#include <boost/python.hpp>
#include <boost/python/stl_iterator.hpp>
#include <boost/python/suite/indexing/vector_indexing_suite.hpp>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

#include "DataStructures/DataVector.hpp"
#include "IO/H5/Header.hpp"
#include "IO/H5/Object.hpp"
#include "IO/H5/OpenGroup.hpp"
#include "IO/H5/Python/Numpy.hpp"
#include "IO/H5/Python/ToNumpy.hpp"
#include "IO/H5/Version.hpp"
#include "IO/H5/VolumeData.hpp"
#include "PythonBindings/VectorPyList.hpp"

namespace bp = boost::python;

namespace py_bindings {
void bind_h5vol() {
  // Wrapper for basic H5VolumeData operations
  bp::class_<h5::VolumeData, boost::noncopyable>("H5Vol", bp::no_init)
      .def("extension", &h5::VolumeData::extension)
      // The method extension() is static
      .staticmethod("extension")

      .def("get_header",
           +[](const h5::VolumeData& volume_file) {
             return volume_file.get_header();
           })

      .def("get_version",
           +[](const h5::VolumeData& volume_file) {
             return volume_file.get_version();
           })

      .def("list_observation_ids",
           +[](const h5::VolumeData& volume_file) {
             return std_vector_to_py_list<size_t>(
                 volume_file.list_observation_ids());
           })
      .def("get_observation_value",
           +[](const h5::VolumeData& volume_file, const size_t observation_id) {
             return volume_file.get_observation_value(observation_id);
           })
      .def("get_grid_names",
           +[](const h5::VolumeData& volume_file, const size_t observation_id) {
             return std_vector_to_py_list<std::string>(
                 volume_file.get_grid_names(observation_id));
           })
      .def("list_tensor_components",
           +[](const h5::VolumeData& volume_file, const size_t observation_id) {
             return std_vector_to_py_list<std::string>(
                 volume_file.list_tensor_components(observation_id));
           })
      .def("get_tensor_component",
           +[](const h5::VolumeData& volume_file, const size_t observation_id,
               const std::string& tensor_component) {
             return volume_file.get_tensor_component(observation_id,
                                                     tensor_component);
           })
      .def("get_extents", +[](const h5::VolumeData& volume_file,
                              const size_t observation_id) {
        bp::list total_extents;
        for (const auto& extents : volume_file.get_extents(observation_id)) {
          total_extents.append(std_vector_to_py_list<size_t>(extents));
        }
        return total_extents;
      });
}
}  // namespace py_bindings
