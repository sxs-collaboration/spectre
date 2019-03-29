// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <vector>

#include <boost/python.hpp>
#include <boost/python/stl_iterator.hpp>
#include <boost/python/suite/indexing/vector_indexing_suite.hpp>

namespace bp = boost::python;

/*!
 * \ingroup PythonBindingsGroup
 * \brief A namespace containing functions for binding Python wrappers
 * to their respective c++ classes and functions, as well as helper functions
 * for these wrappers.
 */
namespace py_bindings {

/*!
 * \ingroup PythonBindingsGroup
 * \brief Convert a bp::list of bp::object& to a std::vector
 */
template <typename T>
std::vector<T> py_list_to_std_vector(const bp::object& iterable) {
  return std::vector<T>(bp::stl_input_iterator<T>(iterable),
                        bp::stl_input_iterator<T>());
}

/*!
 * \ingroup PythonBindingsGroup
 * \brief Convert a std::vector into a bp::list of bp::objects
 */
template <class T>
bp::list std_vector_to_py_list(const std::vector<T>& vector) {
  bp::list list;
  for (auto iter = vector.begin(); iter != vector.end(); ++iter) {
    list.append(*iter);
  }
  return list;
}

}  // namespace py_bindings
