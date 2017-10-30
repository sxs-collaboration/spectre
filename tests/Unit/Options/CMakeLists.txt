# Distributed under the MIT License.
# See LICENSE.txt for details.

set(OPTIONS_TESTS
    Options/Test_CustomTypeConstruction.cpp
    Options/Test_Factory.cpp
    Options/Test_Options.cpp
    )

set(SPECTRE_TESTS "${SPECTRE_TESTS};${OPTIONS_TESTS}" PARENT_SCOPE)
