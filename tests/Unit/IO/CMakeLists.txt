# Distributed under the MIT License.
# See LICENSE.txt for details.

set(SPECTRE_UNIT_IO_TESTS)

set(SPECTRE_UNIT_IO_TESTS
    IO/Test_H5.cpp
    )

set(SPECTRE_TESTS "${SPECTRE_TESTS};${SPECTRE_UNIT_IO_TESTS}"
    PARENT_SCOPE)
