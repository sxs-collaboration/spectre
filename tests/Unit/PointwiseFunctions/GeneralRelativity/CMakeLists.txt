# Distributed under the MIT License.
# See LICENSE.txt for details.

set(SPECTRE_UNIT_GENERAL_RELATIVITY
    PointwiseFunctions/GeneralRelativity/GrTestHelpers.cpp
    PointwiseFunctions/GeneralRelativity/Test_Christoffel.cpp
    PointwiseFunctions/GeneralRelativity/Test_ComputeGhQuantities.cpp
    PointwiseFunctions/GeneralRelativity/Test_ComputeSpacetimeQuantities.cpp
    PointwiseFunctions/GeneralRelativity/Test_IndexManipulation.cpp
    )

set(SPECTRE_UNIT_POINTWISE_FUNCTIONS
  "${SPECTRE_UNIT_POINTWISE_FUNCTIONS};${SPECTRE_UNIT_GENERAL_RELATIVITY}"
  PARENT_SCOPE)
