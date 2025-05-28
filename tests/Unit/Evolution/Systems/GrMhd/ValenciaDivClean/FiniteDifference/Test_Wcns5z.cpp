// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include "Evolution/Systems/GrMhd/ValenciaDivClean/FiniteDifference/Factory.hpp"
#include "Evolution/Systems/GrMhd/ValenciaDivClean/FiniteDifference/Tag.hpp"
#include "Evolution/Systems/GrMhd/ValenciaDivClean/FiniteDifference/Wcns5z.hpp"
#include "Framework/TestCreation.hpp"
#include "Helpers/Evolution/Systems/GrMhd/ValenciaDivClean/FiniteDifference/PrimReconstructor.hpp"
#include "NumericalAlgorithms/FiniteDifference/FallbackReconstructorType.hpp"

SPECTRE_TEST_CASE("Unit.Evolution.Systems.GrMhd.ValenciaDivClean.Fd.Wcns5zPrim",
                  "[Unit][Evolution]") {
  namespace helpers = TestHelpers::grmhd::ValenciaDivClean::fd;
  auto mc = fd::reconstruction::FallbackReconstructorType::MonotonisedCentral;

  // Test reconstructing T and rho*T. For low-order reconstructors this can
  // fail (like MC) because the nonlinear terms (quadratics) aren't captured
  // accurately enough.
  const grmhd::ValenciaDivClean::fd::Wcns5zPrim wcns5z_recons{2, 2.0e-16, mc, 1,
                                                              false};
  helpers::test_prim_reconstructor(4, wcns5z_recons);
  helpers::test_prim_reconstructor(
      4, grmhd::ValenciaDivClean::fd::Wcns5zPrim{2, 2.0e-16, mc, 1, true});

  const auto wcns5z_from_options_base = TestHelpers::test_factory_creation<
      grmhd::ValenciaDivClean::fd::Reconstructor,
      grmhd::ValenciaDivClean::fd::OptionTags::Reconstructor>(
      "Wcns5zPrim:\n"
      "  NonlinearWeightExponent: 2\n"
      "  Epsilon: 2.0e-16\n"
      "  FallbackReconstructor: MonotonisedCentral\n"
      "  MaxNumberOfExtrema: 1\n"
      "  ReconstructRhoTimesTemperature: false\n");
  auto* const wcns5z_from_options =
      dynamic_cast<const grmhd::ValenciaDivClean::fd::Wcns5zPrim*>(
          wcns5z_from_options_base.get());
  REQUIRE(wcns5z_from_options != nullptr);
  CHECK(*wcns5z_from_options == wcns5z_recons);

  CHECK(wcns5z_recons !=
        grmhd::ValenciaDivClean::fd::Wcns5zPrim(1, 2.0e-16, mc, 1, false));
  CHECK(wcns5z_recons !=
        grmhd::ValenciaDivClean::fd::Wcns5zPrim(2, 1.0e-16, mc, 1, false));
  CHECK(wcns5z_recons !=
        grmhd::ValenciaDivClean::fd::Wcns5zPrim(
            2, 2.0e-16, fd::reconstruction::FallbackReconstructorType::None, 1,
            false));
  CHECK(wcns5z_recons !=
        grmhd::ValenciaDivClean::fd::Wcns5zPrim(2, 2.0e-16, mc, 2, false));
  CHECK(wcns5z_recons !=
        grmhd::ValenciaDivClean::fd::Wcns5zPrim(2, 2.0e-16, mc, 1, true));
}
