// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include "Evolution/Systems/NewtonianEuler/FiniteDifference/MonotisedCentral.hpp"
#include "Helpers/Evolution/Systems/NewtonianEuler/FiniteDifference/PrimReconstructor.hpp"

SPECTRE_TEST_CASE(
    "Unit.Evolution.Systems.NewtonianEuler.Fd.MonotisedCentralPrim",
    "[Unit][Evolution]") {
  namespace helpers = TestHelpers::NewtonianEuler::fd;
  helpers::test_prim_reconstructor<1>(
      5, NewtonianEuler::fd::MonotisedCentralPrim<1>{});
  helpers::test_prim_reconstructor<2>(
      5, NewtonianEuler::fd::MonotisedCentralPrim<2>{});
  helpers::test_prim_reconstructor<3>(
      5, NewtonianEuler::fd::MonotisedCentralPrim<3>{});
}
