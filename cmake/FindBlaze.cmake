# Distributed under the MIT License.
# See LICENSE.txt for details.

find_path(
    BLAZE_INCLUDE_DIR
    PATH_SUFFIXES include
    NAMES blaze/Blaze.h
    HINTS ${BLAZE_ROOT}
    DOC "Blaze include directory. Used BLAZE_ROOT to set a search dir."
)

set(BLAZE_INCLUDE_DIRS ${BLAZE_INCLUDE_DIR})

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(
    Blaze
    DEFAULT_MSG BLAZE_INCLUDE_DIR BLAZE_INCLUDE_DIRS
)
mark_as_advanced(BLAZE_INCLUDE_DIR BLAZE_INCLUDE_DIRS)
