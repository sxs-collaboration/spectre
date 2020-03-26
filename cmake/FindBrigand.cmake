# Distributed under the MIT License.
# See LICENSE.txt for details.

if(NOT BRIGAND_ROOT)
  # Need to set to empty to avoid warnings with --warn-uninitialized
  set(BRIGAND_ROOT "")
  set(BRIGAND_ROOT $ENV{BRIGAND_ROOT})
endif()

find_path(
    BRIGAND_INCLUDE_DIR
    PATH_SUFFIXES include
    NAMES brigand/brigand.hpp
    HINTS ${BRIGAND_ROOT}
    DOC "Brigand include directory. Used BRIGAND_ROOT to set a search dir."
)

set(BRIGAND_INCLUDE_DIRS ${BRIGAND_INCLUDE_DIR})

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(
    Brigand
    DEFAULT_MSG BRIGAND_INCLUDE_DIR BRIGAND_INCLUDE_DIRS
)
mark_as_advanced(BRIGAND_INCLUDE_DIR BRIGAND_INCLUDE_DIRS)
