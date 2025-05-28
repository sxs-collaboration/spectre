// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include "Evolution/Systems/GrMhd/ValenciaDivClean/FiniteDifference/Factory.hpp"
#include "Evolution/Systems/GrMhd/ValenciaDivClean/FiniteDifference/MonotonisedCentral.hpp"
#include "Evolution/Systems/GrMhd/ValenciaDivClean/FiniteDifference/Tag.hpp"
#include "Framework/TestCreation.hpp"
#include "Helpers/Evolution/Systems/GrMhd/ValenciaDivClean/FiniteDifference/PrimReconstructor.hpp"

SPECTRE_TEST_CASE(
    "Unit.Evolution.Systems.GrMhd.ValenciaDivClean.Fd.MonotonisedCentralPrim",
    "[Unit][Evolution]") {
  namespace helpers = TestHelpers::grmhd::ValenciaDivClean::fd;
  // We only test reconstructing T because for low-order reconstructors this
  // fails because the nonlinear terms (quadratics) aren't captured
  // accurately enough.
  const grmhd::ValenciaDivClean::fd::MonotonisedCentralPrim mc_recons{false};
  helpers::test_prim_reconstructor(5, mc_recons);
  const auto mc_from_options_base = TestHelpers::test_factory_creation<
      grmhd::ValenciaDivClean::fd::Reconstructor,
      grmhd::ValenciaDivClean::fd::OptionTags::Reconstructor>(
      "MonotonisedCentralPrim:\n"
      "  ReconstructRhoTimesTemperature: false\n");
  auto* const mc_from_options =
      dynamic_cast<const grmhd::ValenciaDivClean::fd::MonotonisedCentralPrim*>(
          mc_from_options_base.get());
  REQUIRE(mc_from_options != nullptr);
  CHECK(*mc_from_options == mc_recons);
  const grmhd::ValenciaDivClean::fd::MonotonisedCentralPrim mc_recons_recon_T{
      true};
  CHECK_FALSE(mc_recons_recon_T == mc_recons);
}
