# Distributed under the MIT License.
# See LICENSE.txt for details.

set(SPECTRE_UNIT_GENERAL_RELATIVITY
    PointwiseFunctions/GeneralRelativity/GrTestHelpers.cpp
    PointwiseFunctions/GeneralRelativity/Test_Christoffel.cpp
    PointwiseFunctions/GeneralRelativity/Test_ComputeGhQuantities.cpp
    PointwiseFunctions/GeneralRelativity/Test_ComputeSpacetimeQuantities.cpp
    PointwiseFunctions/GeneralRelativity/Test_IndexManipulation.cpp
    PointwiseFunctions/GeneralRelativity/Test_Ricci.cpp
    )

set(SPECTRE_UNIT_POINTWISE_FUNCTIONS
  "${SPECTRE_UNIT_POINTWISE_FUNCTIONS};${SPECTRE_UNIT_GENERAL_RELATIVITY}"
  PARENT_SCOPE)
set(LIBRARIES_TO_LINK "${LIBRARIES_TO_LINK};GeneralRelativity" PARENT_SCOPE)
