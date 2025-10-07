# Distributed under the MIT License.
# See LICENSE.txt for details.

set(LIBRARY "PyDomainTimeDependentOptions")

spectre_python_add_module(
    TimeDependentOptions
    LIBRARY_NAME ${LIBRARY}
    MODULE_PATH "Domain/Creators"
    SOURCES
    BinaryCompactObject.cpp
    Bindings.cpp
    ExpansionMap.cpp
    GridCenters.cpp
    RotationMap.cpp
    ShapeMap.cpp
    SkewMap.cpp
    TranslationMap.cpp
    PYTHON_FILES
    __init__.py
)

spectre_python_headers(
    ${LIBRARY}
    INCLUDE_DIRECTORY ${CMAKE_SOURCE_DIR}/src
    HEADERS
    BinaryCompactObject.hpp
    ExpansionMap.hpp
    GridCenters.hpp
    RotationMap.hpp
    ShapeMap.hpp
    SkewMap.hpp
    TranslationMap.hpp
)

spectre_python_link_libraries(
  ${LIBRARY}
  PRIVATE
  DomainCreators
  DomainTimeDependence
  pybind11::module
  )

spectre_python_add_dependencies(
  ${LIBRARY}
  PyDomainCreators
  )
