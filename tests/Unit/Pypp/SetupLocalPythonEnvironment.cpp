// Distributed under the MIT License.
// See LICENSE.txt for details.

#include <codecvt>
#include <string>

#ifndef PY_ARRAY_UNIQUE_SYMBOL
#define PY_ARRAY_UNIQUE_SYMBOL SPECTRE_PY_API
#endif

#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <Python.h>
#include <numpy/arrayobject.h>

#include "ErrorHandling/FloatingPointExceptions.hpp"
#include "Informer/InfoFromBuild.hpp"
#include "tests/Unit/Pypp/PyppFundamentals.hpp"
#include "tests/Unit/Pypp/SetupLocalPythonEnvironment.hpp"

namespace pypp {
SetupLocalPythonEnvironment::SetupLocalPythonEnvironment(
    const std::string &cur_dir_relative_to_unit_test_path) {
  Py_Initialize();
  disable_floating_point_exceptions();
  init_numpy();
  enable_floating_point_exceptions();
  // clang-tidy: Do not use const-cast
  PyObject *pyob_old_paths =
      PySys_GetObject(const_cast<char *>("path"));  // NOLINT
  const auto old_paths =
      pypp::from_py_object<std::vector<std::string>>(pyob_old_paths);
  std::string new_path = unit_test_path() + cur_dir_relative_to_unit_test_path;
  for (const auto &p : old_paths) {
    new_path += ":";
    new_path += p;
  }


#if PY_MAJOR_VERSION == 3
  PySys_SetPath(std::wstring_convert<std::codecvt_utf8<wchar_t>>()
                    .from_bytes(new_path)
                    .c_str());
#else
  // clang-tidy: Do not use const-cast
  PySys_SetPath(const_cast<char*>(new_path.c_str()));  // NOLINT
#endif
}

SetupLocalPythonEnvironment::~SetupLocalPythonEnvironment() { Py_Finalize(); }

#if PY_MAJOR_VERSION == 3
std::nullptr_t SetupLocalPythonEnvironment::init_numpy() {
  import_array();
  return nullptr;
}
#else
void SetupLocalPythonEnvironment::init_numpy() { import_array(); }
#endif
}  // namespace pypp
