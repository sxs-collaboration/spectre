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

  const grmhd::ValenciaDivClean::fd::PositivityPreservingAdaptiveOrderPrim
      PositivityPreservingAdaptiveOrder_recons{4.0, mc};
  helpers::test_prim_reconstructor(4, PositivityPreservingAdaptiveOrder_recons);

  const auto PositivityPreservingAdaptiveOrder_from_options_base =
      TestHelpers::test_factory_creation<
          grmhd::ValenciaDivClean::fd::Reconstructor,
          grmhd::ValenciaDivClean::fd::OptionTags::Reconstructor>(
          "PositivityPreservingAdaptiveOrderPrim:\n"
          "  Alpha5: 4.0\n"
          "  LowOrderReconstructor: MonotonisedCentral\n");
  auto* const PositivityPreservingAdaptiveOrder_from_options =
      dynamic_cast<const grmhd::ValenciaDivClean::fd::
                       PositivityPreservingAdaptiveOrderPrim*>(
          PositivityPreservingAdaptiveOrder_from_options_base.get());
  REQUIRE(PositivityPreservingAdaptiveOrder_from_options != nullptr);
  CHECK(*PositivityPreservingAdaptiveOrder_from_options ==
        PositivityPreservingAdaptiveOrder_recons);

  CHECK(PositivityPreservingAdaptiveOrder_recons !=
        grmhd::ValenciaDivClean::fd::PositivityPreservingAdaptiveOrderPrim(4.5,
                                                                           mc));
  CHECK(PositivityPreservingAdaptiveOrder_recons !=
        grmhd::ValenciaDivClean::fd::PositivityPreservingAdaptiveOrderPrim(
            4.0, fd::reconstruction::FallbackReconstructorType::Minmod));

  CHECK_THROWS_WITH(
      grmhd::ValenciaDivClean::fd::PositivityPreservingAdaptiveOrderPrim(
          4.5, fd::reconstruction::FallbackReconstructorType::None),
      Catch::Contains("None is not an allowed low-order reconstructor."));
}
