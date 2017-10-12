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

add_library(${LIBRARY} ${LIBRARY_SOURCES})
