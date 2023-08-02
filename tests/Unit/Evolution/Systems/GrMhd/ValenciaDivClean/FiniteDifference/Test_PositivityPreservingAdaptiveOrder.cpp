// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include "Evolution/Systems/GrMhd/ValenciaDivClean/FiniteDifference/Factory.hpp"
#include "Evolution/Systems/GrMhd/ValenciaDivClean/FiniteDifference/PositivityPreservingAdaptiveOrder.hpp"
#include "Evolution/Systems/GrMhd/ValenciaDivClean/FiniteDifference/Tag.hpp"
#include "Framework/TestCreation.hpp"
#include "Helpers/Evolution/Systems/GrMhd/ValenciaDivClean/FiniteDifference/PrimReconstructor.hpp"
#include "NumericalAlgorithms/FiniteDifference/FallbackReconstructorType.hpp"

SPECTRE_TEST_CASE("Unit.Evolution.Systems.GrMhd.ValenciaDivClean.Fd.PpaoPrim",
                  "[Unit][Evolution]") {
  namespace helpers = TestHelpers::grmhd::ValenciaDivClean::fd;
  auto mc = fd::reconstruction::FallbackReconstructorType::MonotonisedCentral;

  helpers::test_prim_reconstructor(
      6, grmhd::ValenciaDivClean::fd::PositivityPreservingAdaptiveOrderPrim{
             4.0, 4.0, std::nullopt, mc});
  helpers::test_prim_reconstructor(
      8, grmhd::ValenciaDivClean::fd::PositivityPreservingAdaptiveOrderPrim{
             4.0, std::nullopt, 4.0, mc});
  helpers::test_prim_reconstructor(
      8, grmhd::ValenciaDivClean::fd::PositivityPreservingAdaptiveOrderPrim{
             4.0, 4.0, 4.0, mc});

  const grmhd::ValenciaDivClean::fd::PositivityPreservingAdaptiveOrderPrim
      ppao_recons{4.0, std::nullopt, std::nullopt, mc};
  helpers::test_prim_reconstructor(4, ppao_recons);

  const auto ppao_from_options_base = TestHelpers::test_factory_creation<
      grmhd::ValenciaDivClean::fd::Reconstructor,
      grmhd::ValenciaDivClean::fd::OptionTags::Reconstructor>(
      "PositivityPreservingAdaptiveOrderPrim:\n"
      "  Alpha5: 4.0\n"
      "  Alpha7: None\n"
      "  Alpha9: None\n"
      "  LowOrderReconstructor: MonotonisedCentral\n");
  auto* const ppao_from_options =
      dynamic_cast<const grmhd::ValenciaDivClean::fd::
                       PositivityPreservingAdaptiveOrderPrim*>(
          ppao_from_options_base.get());
  REQUIRE(ppao_from_options != nullptr);
  CHECK(*ppao_from_options == ppao_recons);

  CHECK(ppao_recons !=
        grmhd::ValenciaDivClean::fd::PositivityPreservingAdaptiveOrderPrim(
            4.5, std::nullopt, std::nullopt, mc));
  CHECK(ppao_recons !=
        grmhd::ValenciaDivClean::fd::PositivityPreservingAdaptiveOrderPrim(
            4.0, 4.0, std::nullopt, mc));
  CHECK(grmhd::ValenciaDivClean::fd::PositivityPreservingAdaptiveOrderPrim(
            4.0, 4.0, std::nullopt, mc) !=
        grmhd::ValenciaDivClean::fd::PositivityPreservingAdaptiveOrderPrim(
            4.0, 4.1, std::nullopt, mc));
  CHECK(ppao_recons !=
        grmhd::ValenciaDivClean::fd::PositivityPreservingAdaptiveOrderPrim(
            4.0, std::nullopt, 4.0, mc));
  CHECK(grmhd::ValenciaDivClean::fd::PositivityPreservingAdaptiveOrderPrim(
            4.0, std::nullopt, 4.0, mc) !=
        grmhd::ValenciaDivClean::fd::PositivityPreservingAdaptiveOrderPrim(
            4.0, std::nullopt, 4.1, mc));
  CHECK(grmhd::ValenciaDivClean::fd::PositivityPreservingAdaptiveOrderPrim(
            5.0, std::nullopt, 4.0, mc) ==
        grmhd::ValenciaDivClean::fd::PositivityPreservingAdaptiveOrderPrim(
            5.0, std::nullopt, 4.0, mc));
  CHECK(grmhd::ValenciaDivClean::fd::PositivityPreservingAdaptiveOrderPrim(
            5.0, 6.0, 4.0, mc) ==
        grmhd::ValenciaDivClean::fd::PositivityPreservingAdaptiveOrderPrim(
            5.0, 6.0, 4.0, mc));
  CHECK(grmhd::ValenciaDivClean::fd::PositivityPreservingAdaptiveOrderPrim(
            5.0, 6.0, 4.0, mc) !=
        grmhd::ValenciaDivClean::fd::PositivityPreservingAdaptiveOrderPrim(
            5.1, 6.0, 4.0, mc));
  CHECK(grmhd::ValenciaDivClean::fd::PositivityPreservingAdaptiveOrderPrim(
            5.0, 6.0, 4.0, mc) !=
        grmhd::ValenciaDivClean::fd::PositivityPreservingAdaptiveOrderPrim(
            5.0, 6.1, 4.0, mc));
  CHECK(grmhd::ValenciaDivClean::fd::PositivityPreservingAdaptiveOrderPrim(
            5.0, 6.0, 4.0, mc) !=
        grmhd::ValenciaDivClean::fd::PositivityPreservingAdaptiveOrderPrim(
            5.0, 6.0, 4.1, mc));
  CHECK(grmhd::ValenciaDivClean::fd::PositivityPreservingAdaptiveOrderPrim(
            5.0, 6.0, 4.0, mc) !=
        grmhd::ValenciaDivClean::fd::PositivityPreservingAdaptiveOrderPrim(
            5.0, 6.0, 4.0,
            fd::reconstruction::FallbackReconstructorType::Minmod));
  CHECK(ppao_recons !=
        grmhd::ValenciaDivClean::fd::PositivityPreservingAdaptiveOrderPrim(
            4.0, std::nullopt, std::nullopt,
            fd::reconstruction::FallbackReconstructorType::Minmod));

  CHECK_THROWS_WITH(
      grmhd::ValenciaDivClean::fd::PositivityPreservingAdaptiveOrderPrim(
          4.5, std::nullopt, std::nullopt,
          fd::reconstruction::FallbackReconstructorType::None),
      Catch::Matchers::ContainsSubstring(
          "None is not an allowed low-order reconstructor."));
}
