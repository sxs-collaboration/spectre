# Distributed under the MIT License.
# See LICENSE.txt for details.

set(LIBRARY SPHEREPACK)

set(LIBRARY_SOURCES
    alf.f
    divgs.f
    gaqd.f
    gradgs.f
    hrfft.f
    shags.f
    shsgs.f
    slapgs.f
    sphcom.f
    vhags.f
    vhsgs.f
    vrtgs.f
    vtsgs.f
    )

# Starting in gcc8, gfortran warns about instances of Fortran77 coding style in
# certain SPHEREPACK files. The warning is:
#   Warning: Array reference at ... out of bounds ... in loop beginning ...
#
# Attempts to silence the warning by use of individual flags were unsuccessful,
# so instead we silence all warnings for the problematic files.
file(GLOB FILES_WITH_WARNINGS
    alf.f
    sphcom.f
    vhags.f
    vhsgs.f
    vtsgs.f
    )
set_source_files_properties(${FILES_WITH_WARNINGS} PROPERTIES COMPILE_FLAGS -w)

add_spectre_library(${LIBRARY} ${LIBRARY_SOURCES})
