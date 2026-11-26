// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include "Evolution/Systems/GrMhd/ValenciaDivClean/FiniteDifference/Factory.hpp"
#include "Evolution/Systems/GrMhd/ValenciaDivClean/FiniteDifference/Ppm.hpp"
#include "Evolution/Systems/GrMhd/ValenciaDivClean/FiniteDifference/Tag.hpp"
#include "Framework/TestCreation.hpp"
#include "Helpers/Evolution/Systems/GrMhd/ValenciaDivClean/FiniteDifference/PrimReconstructor.hpp"

SPECTRE_TEST_CASE("Unit.Evolution.Systems.GrMhd.ValenciaDivClean.Fd.PpmPrim",
                  "[Unit][Evolution]") {
  namespace helpers = TestHelpers::grmhd::ValenciaDivClean::fd;
  // We only test reconstructing T because for low-order reconstructors this
  // fails because the nonlinear terms (quadratics) aren't captured
  // accurately enough.
  const grmhd::ValenciaDivClean::fd::PpmPrim ppm_recons{false};
  helpers::test_prim_reconstructor(5, ppm_recons);
  const auto ppm_from_options_base = TestHelpers::test_factory_creation<
      grmhd::ValenciaDivClean::fd::Reconstructor,
      grmhd::ValenciaDivClean::fd::OptionTags::Reconstructor>(
      "PpmPrim:\n"
      "  ReconstructRhoTimesTemperature: false\n");
  auto* const ppm_from_options =
      dynamic_cast<const grmhd::ValenciaDivClean::fd::PpmPrim*>(
          ppm_from_options_base.get());
  REQUIRE(ppm_from_options != nullptr);
  CHECK(*ppm_from_options == ppm_recons);
  const grmhd::ValenciaDivClean::fd::PpmPrim ppm_recons_recon_T{true};
  CHECK_FALSE(ppm_recons_recon_T == ppm_recons);
}
