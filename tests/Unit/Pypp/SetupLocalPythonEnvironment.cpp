// Distributed under the MIT License.
// See LICENSE.txt for details.

#include <codecvt>  // IWYU pragma: keep
#include <locale>   // IWYU pragma: keep
#include <string>
#include <vector>

#ifndef PY_ARRAY_UNIQUE_SYMBOL
#define PY_ARRAY_UNIQUE_SYMBOL SPECTRE_PY_API
#endif

#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <Python.h>
#include <numpy/arrayobject.h>  // IWYU pragma: keep
#include <pydebug.h>

#include "ErrorHandling/FloatingPointExceptions.hpp"
#include "Informer/InfoFromBuild.hpp"
#include "tests/Unit/Pypp/PyppFundamentals.hpp"
#include "tests/Unit/Pypp/SetupLocalPythonEnvironment.hpp"

namespace pypp {
SetupLocalPythonEnvironment::SetupLocalPythonEnvironment(
    const std::string &cur_dir_relative_to_unit_test_path) {
  // In the case where we run all the non-failure tests at once we must ensure
  // that we only initialize and finalize the python env once. Initialization is
  // done in the constructor of SetupLocalPythonEnvironment, while finalization
  // is done in the constructor of RunTests.
  if (not initialized) {
    // Don't produce the __pycache__ dir (python 3.2 and newer) or the .pyc
    // files (python 2.7) in the tests directory to avoid cluttering the source
    // tree. The overhead of not having the compile files is <= 0.01s
    Py_DontWriteBytecodeFlag = 1;
    Py_Initialize();
    // On some python versions init_numpy() can throw an FPE, this occurred at
    // least with python 3.6, numpy 1.14.2.
    disable_floating_point_exceptions();
    init_numpy();
    enable_floating_point_exceptions();
  }
  initialized = true;
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

#if PY_MAJOR_VERSION == 3
std::nullptr_t SetupLocalPythonEnvironment::init_numpy() {
  import_array();
  return nullptr;
}
#else
void SetupLocalPythonEnvironment::init_numpy() { import_array(); }
#endif

void SetupLocalPythonEnvironment::finalize_env() {
  if (not finalized and initialized) {
    Py_Finalize();
  }
  finalized = true;
}

bool SetupLocalPythonEnvironment::initialized = false;
bool SetupLocalPythonEnvironment::finalized = false;
}  // namespace pypp
