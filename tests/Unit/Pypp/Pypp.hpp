// Distributed under the MIT License.
// See LICENSE.txt for details.

/// \file
/// Defines function pypp::call<R,Args...>

#pragma once

#include <Python.h>
#include <stdexcept>
#include <string>

#include "tests/Unit/Pypp/PyppFundamentals.hpp"

/// \ingroup TestingFrameworkGroup
/// Contains all functions for calling python from C++
namespace pypp {

/// Calls a Python function from a module/file with given parameters
///
/// \param module_name name of module the function is in
/// \param function_name name of Python function in module
/// \param t the arguments to be passed to the Python function
/// \return the object returned by the Python function converted to a C++ type
///
/// Custom classes can be converted between Python and C++ by overloading the
/// `pypp::ToPythonObject<T>` and `pypp::FromPythonObject<T>` structs for your
/// own types. This tells C++ how to deconstruct the Python object into
/// fundamental types and reconstruct the C++ object and vice-versa.
///
/// \note In order to setup the python interpreter and add the local directories
/// to the path, a SetupLocalPythonEnvironment object needs to be constructed
/// in the local scope.
///
/// \example
/// The following example calls the function `test_numeric` from the module
/// `pypp_py_tests` which multiplies two integers.
/// \snippet Test_Pypp.cpp pypp_int_test
/// Alternatively, this examples calls `test_vector` from `pypp_py_tests` which
/// converts two vectors to python lists and multiplies them pointwise.
/// \snippet Test_Pypp.cpp pypp_vector_test
template <typename R, typename... Args>
R call(const std::string& module_name, const std::string& function_name,
       const Args&... t) {
  PyObject* module = PyImport_ImportModule(module_name.c_str());
  if (module == nullptr) {
    PyErr_Print();
    throw std::runtime_error{std::string("Could not find python module.\n") +
                             module_name};
  }
  PyObject* func = PyObject_GetAttrString(module, function_name.c_str());
  if (func == nullptr or not PyCallable_Check(func)) {
    if (PyErr_Occurred()) {
      PyErr_Print();
    }
    throw std::runtime_error{"Could not find python function in module.\n"};
  }
  PyObject* args = make_py_tuple(t...);
  PyObject* value = PyObject_CallObject(func, args);
  Py_DECREF(args);  // NOLINT
  if (value == nullptr) {
    Py_DECREF(func);    // NOLINT
    Py_DECREF(module);  // NOLINT
    PyErr_Print();
    throw std::runtime_error{"Function returned null"};
  }
  auto ret = from_py_object<R>(value);
  Py_DECREF(value);   // NOLINT
  Py_DECREF(func);    // NOLINT
  Py_DECREF(module);  // NOLINT
  return ret;
}
}  // namespace pypp
