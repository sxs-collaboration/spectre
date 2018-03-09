// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <codecvt>
#include <string>

#include "Informer/InfoFromBuild.hpp"
#include "tests/Unit/Pypp/PyppFundamentals.hpp"

/// Contains all functions for pypp
namespace pypp {

/// Enable calling of python in the local scope, and add directory(ies) to the
/// front of the search path for modules. The directory which is appended to the
/// path is relative to the `tests/Unit` directory.
struct SetupLocalPythonEnvironment {
  explicit SetupLocalPythonEnvironment(
      const std::string& cur_dir_relative_to_unit_test_path) {
    Py_Initialize();
    // clang-tidy: Do not use const-cast
    PyObject* pyob_old_paths =
        PySys_GetObject(const_cast<char*>("path"));  // NOLINT
    const auto old_paths =
        pypp::from_py_object<std::vector<std::string>>(pyob_old_paths);
    std::string new_path =
        unit_test_path() + cur_dir_relative_to_unit_test_path;
    for (const auto& p : old_paths) {
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
  ~SetupLocalPythonEnvironment() { Py_Finalize(); }

  SetupLocalPythonEnvironment(const SetupLocalPythonEnvironment&) = delete;
  SetupLocalPythonEnvironment& operator=(const SetupLocalPythonEnvironment&) =
      delete;
  SetupLocalPythonEnvironment(const SetupLocalPythonEnvironment&&) = delete;
  SetupLocalPythonEnvironment& operator=(const SetupLocalPythonEnvironment&&) =
      delete;
};
}  // namespace pypp
