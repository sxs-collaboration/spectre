// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <vector>

#include <boost/python.hpp>
#include <boost/python/stl_iterator.hpp>
#include <boost/python/suite/indexing/vector_indexing_suite.hpp>

namespace bp = boost::python;

namespace py_bindings {
template <typename T>
std::vector<T> py_list_to_std_vector(const bp::object& iterable) {
  return std::vector<T>(bp::stl_input_iterator<T>(iterable),
                        bp::stl_input_iterator<T>());
}

template <class T>
bp::list std_vector_to_py_list(const std::vector<T>& vector) {
  bp::list list;
  for (auto iter = vector.begin(); iter != vector.end(); ++iter) {
    list.append(*iter);
  }
  return list;
}

}  // namespace py_bindings
