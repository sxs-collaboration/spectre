# Distributed under the MIT License.
# See LICENSE.txt for details.

option(BOOTSTRAP_PY_DEPS
  "Install missing Python dependencies in the build directory"
  ${SPECTRE_FETCH_MISSING_DEPS})
option(BOOTSTRAP_PY_DEV_DEPS
  "Install missing Python dev dependencies in the build directory"
  ${SPECTRE_FETCH_MISSING_DEPS})

if (NOT (BOOTSTRAP_PY_DEPS OR BOOTSTRAP_PY_DEV_DEPS))
  return()
endif()

message(STATUS "Bootstrapping missing Python dependencies to: \
${SPECTRE_PYTHON_SITE_PACKAGES}")

# Install the packages with pip
set(_BOOTSTRAP_PY_DEPS_FLAG "")
if (BOOTSTRAP_PY_DEPS)
  set(_BOOTSTRAP_PY_DEPS_FLAG
    -r ${CMAKE_SOURCE_DIR}/support/Python/requirements.txt)
endif()
set(_BOOTSTRAP_PY_DEV_DEPS_FLAG "")
if (BOOTSTRAP_PY_DEV_DEPS)
  set(_BOOTSTRAP_PY_DEV_DEPS_FLAG
    -r ${CMAKE_SOURCE_DIR}/support/Python/dev_requirements.txt)
endif()
set(_BOOTSTRAP_PY_DEPS_LOG_FILE "${CMAKE_BINARY_DIR}/BootstrapPyDeps.log")
execute_process(COMMAND
  ${CMAKE_COMMAND} -E env
  PYTHONPATH=${PYTHONPATH}
  ${Python_EXECUTABLE} -m pip install --disable-pip-version-check
  --prefix ${CMAKE_BINARY_DIR} --no-warn-script-location
  ${_BOOTSTRAP_PY_DEPS_FLAG} ${_BOOTSTRAP_PY_DEV_DEPS_FLAG}
  RESULT_VARIABLE _BOOTSTRAP_PY_DEPS_RESULT
  OUTPUT_FILE ${_BOOTSTRAP_PY_DEPS_LOG_FILE}
  ERROR_FILE ${_BOOTSTRAP_PY_DEPS_LOG_FILE}
  )
if (NOT _BOOTSTRAP_PY_DEPS_RESULT EQUAL 0)
  message(WARNING "Bootstrapping of Python dependencies failed."
    "See log file: ${_BOOTSTRAP_PY_DEPS_LOG_FILE}")
endif()
