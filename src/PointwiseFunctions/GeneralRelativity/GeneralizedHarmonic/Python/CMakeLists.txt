# Distributed under the MIT License.
# See LICENSE.txt for details.

set(LIBRARY "PyGeneralizedHarmonic")

spectre_python_add_module(
    GeneralizedHarmonic
    LIBRARY_NAME ${LIBRARY}
    MODULE_PATH "PointwiseFunctions/GeneralRelativity"
    SOURCES
    Bindings.cpp
    PYTHON_FILES
    __init__.py
)

spectre_python_headers(
    ${LIBRARY}
    INCLUDE_DIRECTORY ${CMAKE_SOURCE_DIR}/src
    HEADERS
)

spectre_python_link_libraries(
    ${LIBRARY}
    PRIVATE
    DataStructures
    GeneralizedHarmonic
    pybind11::module
    Utilities
)

spectre_python_add_dependencies(
  ${LIBRARY}
  PyDataStructures
  PyTensor
  )
