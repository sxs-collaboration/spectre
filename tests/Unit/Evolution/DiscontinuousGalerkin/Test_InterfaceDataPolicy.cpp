// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include "Evolution/DiscontinuousGalerkin/InterfaceDataPolicy.hpp"
#include "Utilities/GetOutput.hpp"

SPECTRE_TEST_CASE("Unit.Evolution.DG.InterfaceDataPolicy",
                  "[Unit][Evolution]") {
  CHECK(get_output(evolution::dg::InterfaceDataPolicy::Uninitialized) ==
        "Uninitialized");
  CHECK(get_output(evolution::dg::InterfaceDataPolicy::CopyProject) ==
        "CopyProject");
  CHECK(get_output(evolution::dg::InterfaceDataPolicy::OrientCopyProject) ==
        "OrientCopyProject");
  CHECK(get_output(
            evolution::dg::InterfaceDataPolicy::NonconformingBothInterpolate) ==
        "NonconformingBothInterpolate");
  CHECK(
      get_output(
          evolution::dg::InterfaceDataPolicy::NonconformingSelfInterpolates) ==
      "NonconformingSelfInterpolates");
  CHECK(get_output(evolution::dg::InterfaceDataPolicy::
                       NonconformingNeighborInterpolates) ==
        "NonconformingNeighborInterpolates");
}
