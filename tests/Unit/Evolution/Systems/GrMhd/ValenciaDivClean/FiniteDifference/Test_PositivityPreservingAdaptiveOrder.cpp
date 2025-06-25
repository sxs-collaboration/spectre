// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include "Evolution/Systems/GrMhd/ValenciaDivClean/FiniteDifference/Factory.hpp"
#include "Evolution/Systems/GrMhd/ValenciaDivClean/FiniteDifference/PositivityPreservingAdaptiveOrder.hpp"
#include "Evolution/Systems/GrMhd/ValenciaDivClean/FiniteDifference/Tag.hpp"
#include "Framework/TestCreation.hpp"
#include "Helpers/Evolution/Systems/GrMhd/ValenciaDivClean/FiniteDifference/PrimReconstructor.hpp"
#include "NumericalAlgorithms/FiniteDifference/FallbackReconstructorType.hpp"

// [[TimeOut, 10]]
SPECTRE_TEST_CASE("Unit.Evolution.Systems.GrMhd.ValenciaDivClean.Fd.PpaoPrim",
                  "[Unit][Evolution]") {
  namespace helpers = TestHelpers::grmhd::ValenciaDivClean::fd;
  auto mc = fd::reconstruction::FallbackReconstructorType::MonotonisedCentral;

  // Test reconstructing T and rho*T. For low-order reconstructors this can
  // fail (like MC) because the nonlinear terms (quadratics) aren't captured
  // accurately enough.
  for (const bool reconstruct_rho_times_T : {false, true}) {
    CAPTURE(reconstruct_rho_times_T);
    helpers::test_prim_reconstructor(
        6, grmhd::ValenciaDivClean::fd::PositivityPreservingAdaptiveOrderPrim{
               4.0, 4.0, std::nullopt, mc, reconstruct_rho_times_T});
    helpers::test_prim_reconstructor(
        8, grmhd::ValenciaDivClean::fd::PositivityPreservingAdaptiveOrderPrim{
               4.0, std::nullopt, 4.0, mc, reconstruct_rho_times_T});
    helpers::test_prim_reconstructor(
        8, grmhd::ValenciaDivClean::fd::PositivityPreservingAdaptiveOrderPrim{
               4.0, 4.0, 4.0, mc, reconstruct_rho_times_T});
  }

  const grmhd::ValenciaDivClean::fd::PositivityPreservingAdaptiveOrderPrim
      ppao_recons{4.0, std::nullopt, std::nullopt, mc, false};
  helpers::test_prim_reconstructor(4, ppao_recons);

  const auto ppao_from_options_base = TestHelpers::test_factory_creation<
      grmhd::ValenciaDivClean::fd::Reconstructor,
      grmhd::ValenciaDivClean::fd::OptionTags::Reconstructor>(
      "PositivityPreservingAdaptiveOrderPrim:\n"
      "  Alpha5: 4.0\n"
      "  Alpha7: None\n"
      "  Alpha9: None\n"
      "  LowOrderReconstructor: MonotonisedCentral\n"
      "  ReconstructRhoTimesTemperature: false\n");
  auto* const ppao_from_options =
      dynamic_cast<const grmhd::ValenciaDivClean::fd::
                       PositivityPreservingAdaptiveOrderPrim*>(
          ppao_from_options_base.get());
  REQUIRE(ppao_from_options != nullptr);
  CHECK(*ppao_from_options == ppao_recons);

  CHECK(ppao_recons !=
        grmhd::ValenciaDivClean::fd::PositivityPreservingAdaptiveOrderPrim(
            4.5, std::nullopt, std::nullopt, mc, false));
  CHECK(ppao_recons !=
        grmhd::ValenciaDivClean::fd::PositivityPreservingAdaptiveOrderPrim(
            4.0, 4.0, std::nullopt, mc, false));
  CHECK(grmhd::ValenciaDivClean::fd::PositivityPreservingAdaptiveOrderPrim(
            4.0, 4.0, std::nullopt, mc, false) !=
        grmhd::ValenciaDivClean::fd::PositivityPreservingAdaptiveOrderPrim(
            4.0, 4.1, std::nullopt, mc, false));
  CHECK(ppao_recons !=
        grmhd::ValenciaDivClean::fd::PositivityPreservingAdaptiveOrderPrim(
            4.0, std::nullopt, 4.0, mc, false));
  CHECK(grmhd::ValenciaDivClean::fd::PositivityPreservingAdaptiveOrderPrim(
            4.0, std::nullopt, 4.0, mc, false) !=
        grmhd::ValenciaDivClean::fd::PositivityPreservingAdaptiveOrderPrim(
            4.0, std::nullopt, 4.1, mc, false));
  CHECK(grmhd::ValenciaDivClean::fd::PositivityPreservingAdaptiveOrderPrim(
            5.0, std::nullopt, 4.0, mc, false) ==
        grmhd::ValenciaDivClean::fd::PositivityPreservingAdaptiveOrderPrim(
            5.0, std::nullopt, 4.0, mc, false));
  CHECK(grmhd::ValenciaDivClean::fd::PositivityPreservingAdaptiveOrderPrim(
            5.0, 6.0, 4.0, mc, false) ==
        grmhd::ValenciaDivClean::fd::PositivityPreservingAdaptiveOrderPrim(
            5.0, 6.0, 4.0, mc, false));
  CHECK(grmhd::ValenciaDivClean::fd::PositivityPreservingAdaptiveOrderPrim(
            5.0, 6.0, 4.0, mc, false) !=
        grmhd::ValenciaDivClean::fd::PositivityPreservingAdaptiveOrderPrim(
            5.1, 6.0, 4.0, mc, false));
  CHECK(grmhd::ValenciaDivClean::fd::PositivityPreservingAdaptiveOrderPrim(
            5.0, 6.0, 4.0, mc, false) !=
        grmhd::ValenciaDivClean::fd::PositivityPreservingAdaptiveOrderPrim(
            5.0, 6.1, 4.0, mc, false));
  CHECK(grmhd::ValenciaDivClean::fd::PositivityPreservingAdaptiveOrderPrim(
            5.0, 6.0, 4.0, mc, false) !=
        grmhd::ValenciaDivClean::fd::PositivityPreservingAdaptiveOrderPrim(
            5.0, 6.0, 4.1, mc, false));
  CHECK(grmhd::ValenciaDivClean::fd::PositivityPreservingAdaptiveOrderPrim(
            5.0, 6.0, 4.0, mc, false) !=
        grmhd::ValenciaDivClean::fd::PositivityPreservingAdaptiveOrderPrim(
            5.0, 6.0, 4.0,
            fd::reconstruction::FallbackReconstructorType::Minmod, false));
  CHECK(ppao_recons !=
        grmhd::ValenciaDivClean::fd::PositivityPreservingAdaptiveOrderPrim(
            4.0, std::nullopt, std::nullopt,
            fd::reconstruction::FallbackReconstructorType::Minmod, false));
  CHECK(grmhd::ValenciaDivClean::fd::PositivityPreservingAdaptiveOrderPrim(
            5.0, 6.0, 4.0, mc, false) !=
        grmhd::ValenciaDivClean::fd::PositivityPreservingAdaptiveOrderPrim(
            5.0, 6.0, 4.0, mc, true));

  CHECK_THROWS_WITH(
      grmhd::ValenciaDivClean::fd::PositivityPreservingAdaptiveOrderPrim(
          4.5, std::nullopt, std::nullopt,
          fd::reconstruction::FallbackReconstructorType::None, false),
      Catch::Matchers::ContainsSubstring(
          "None is not an allowed low-order reconstructor."));
}
