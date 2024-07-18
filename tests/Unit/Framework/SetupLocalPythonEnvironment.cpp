// Distributed under the MIT License.
// See LICENSE.txt for details.

#include <boost/preprocessor.hpp>
#include <string>
#include <vector>

#ifndef PY_ARRAY_UNIQUE_SYMBOL
#define PY_ARRAY_UNIQUE_SYMBOL SPECTRE_PY_API
#endif

#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <Python.h>
#include <numpy/arrayobject.h>

#include "Framework/PyppFundamentals.hpp"
#include "Framework/SetupLocalPythonEnvironment.hpp"
#include "Informer/InfoFromBuild.hpp"
#include "Utilities/ErrorHandling/Error.hpp"
#include "Utilities/ErrorHandling/FloatingPointExceptions.hpp"
#include "Utilities/FileSystem.hpp"

namespace {
void prepend_python_path(const std::string& new_path) {
  if (not file_system::check_if_dir_exists(new_path)) {
    ERROR_NO_TRACE("Trying to add path '"
                   << new_path
                   << "' to the python environment during setup "
                      "but this directory does not exist. Either "
                      "you have a typo in your path, or you need "
                      "to create the appropriate site-packages "
                      "directory after upgrading python.");
  }
  PyObject* sys_path = PySys_GetObject("path");
  PyList_Insert(sys_path, 0, PyUnicode_FromString(new_path.c_str()));
}
}  // namespace

namespace pypp {
SetupLocalPythonEnvironment::SetupLocalPythonEnvironment(
    const std::string& cur_dir_relative_to_unit_test_path) {
  // We have to clean up the Python environment only after all tests have
  // finished running, since there could be multiple tests run in a single
  // executable launch. This is done in TestMain(Charm).cpp.
  if (not initialized) {
    PyStatus status;
    PyConfig config;
    PyConfig_InitPythonConfig(&config);
    // Populate the Python config with the standard values
    status = PyConfig_Read(&config);
    // Don't produce the __pycache__ dir (python 3.2 and newer) or the .pyc
    // files (python 2.7) in the tests directory to avoid cluttering the source
    // tree. The overhead of not having the compile files is <= 0.01s
    config.write_bytecode = 0;
    status = Py_InitializeFromConfig(&config);
    PyConfig_Clear(&config);

    // Add directory for installed packages (see CMakeLists.txt for details)
    prepend_python_path(PYTHON_SITELIB);

    // On some python versions init_numpy() can throw an FPE, this occurred at
    // least with python 3.6, numpy 1.14.2.
    ScopedFpeState disable_fpes(false);
    init_numpy();
    disable_fpes.restore_exceptions();
  }
  initialized = true;

  // Add test directory to the Python path
  prepend_python_path(unit_test_src_path() +
                      cur_dir_relative_to_unit_test_path);
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
