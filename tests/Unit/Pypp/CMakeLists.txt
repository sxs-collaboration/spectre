# Distributed under the MIT License.
# See LICENSE.txt for details.

set(PYPP_TESTS
    Pypp/Test_Pypp.cpp
  )

set(SPECTRE_TESTS "${SPECTRE_TESTS};${PYPP_TESTS}" PARENT_SCOPE)
