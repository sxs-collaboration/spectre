# Distributed under the MIT License.
# See LICENSE.txt for details.

set(LIBRARY "PyRingdown")

spectre_python_add_module(
    Ringdown
    LIBRARY_NAME ${LIBRARY}
    MODULE_PATH "Evolution"
    SOURCES
    Bindings.cpp
    PYTHON_FILES
    __init__.py
    ComputeAhCCoefsInRingdownDistortedFrame.py
    FunctionsOfTimeFromVolume.py
)

spectre_python_headers(
    ${LIBRARY}
    INCLUDE_DIRECTORY ${CMAKE_SOURCE_DIR}/src
    HEADERS
)

spectre_python_link_libraries(
    ${LIBRARY}
    PRIVATE
    Boost::boost
    CoordinateMaps
    DataStructures
    Domain
    DomainCreators
    DomainStructure
    ErrorHandling
    Options
    ParallelInterpolation
    Ringdown
    Serialization
    Spectral
    SphericalHarmonics
    SphericalHarmonicsIO
    Utilities
    pybind11::module
)

spectre_python_add_dependencies(
  ${LIBRARY}
  PyDataStructures
  PyH5
  PySpectral
  PyTensor
  )
