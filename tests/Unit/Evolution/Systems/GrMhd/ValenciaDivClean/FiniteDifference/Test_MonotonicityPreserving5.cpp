// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include "Evolution/Systems/GrMhd/ValenciaDivClean/FiniteDifference/Factory.hpp"
#include "Evolution/Systems/GrMhd/ValenciaDivClean/FiniteDifference/MonotonicityPreserving5.hpp"
#include "Evolution/Systems/GrMhd/ValenciaDivClean/FiniteDifference/Tag.hpp"
#include "Framework/TestCreation.hpp"
#include "Helpers/Evolution/Systems/GrMhd/ValenciaDivClean/FiniteDifference/PrimReconstructor.hpp"

SPECTRE_TEST_CASE("Unit.Evolution.Systems.GrMhd.ValenciaDivClean.Fd.Mp5Prim",
                  "[Unit][Evolution]") {
  namespace helpers = TestHelpers::grmhd::ValenciaDivClean::fd;
  // Test reconstructing T and rho*T. For low-order reconstructors this can
  // fail (like MC) because the nonlinear terms (quadratics) aren't captured
  // accurately enough.
  const grmhd::ValenciaDivClean::fd::MonotonicityPreserving5Prim mp5_recons{
      4.0, 1e-10, false};
  helpers::test_prim_reconstructor(4, mp5_recons);
  helpers::test_prim_reconstructor(
      4, grmhd::ValenciaDivClean::fd::MonotonicityPreserving5Prim{4.0, 1e-10,
                                                                  true});

  const auto wcns5z_from_options_base = TestHelpers::test_factory_creation<
      grmhd::ValenciaDivClean::fd::Reconstructor,
      grmhd::ValenciaDivClean::fd::OptionTags::Reconstructor>(
      "MonotonicityPreserving5Prim:\n"
      "  Alpha: 4.0\n"
      "  Epsilon: 1.0e-10\n"
      "  ReconstructRhoTimesTemperature: false\n");
  auto* const mp5_from_options = dynamic_cast<
      const grmhd::ValenciaDivClean::fd::MonotonicityPreserving5Prim*>(
      wcns5z_from_options_base.get());
  REQUIRE(mp5_from_options != nullptr);
  CHECK(*mp5_from_options == mp5_recons);

  CHECK(mp5_recons != grmhd::ValenciaDivClean::fd::MonotonicityPreserving5Prim(
                          3.0, 1e-10, false));
  CHECK(mp5_recons != grmhd::ValenciaDivClean::fd::MonotonicityPreserving5Prim(
                          4.0, 2e-10, false));
  CHECK(mp5_recons != grmhd::ValenciaDivClean::fd::MonotonicityPreserving5Prim(
                          4.0, 1e-10, true));
}
