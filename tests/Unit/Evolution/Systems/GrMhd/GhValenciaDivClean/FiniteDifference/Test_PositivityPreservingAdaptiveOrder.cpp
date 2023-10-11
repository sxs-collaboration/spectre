// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include "Evolution/Systems/GrMhd/GhValenciaDivClean/FiniteDifference/Factory.hpp"
#include "Evolution/Systems/GrMhd/GhValenciaDivClean/FiniteDifference/PositivityPreservingAdaptiveOrder.hpp"
#include "Evolution/Systems/GrMhd/GhValenciaDivClean/FiniteDifference/Reconstructor.hpp"
#include "Evolution/Systems/GrMhd/GhValenciaDivClean/FiniteDifference/Tag.hpp"
#include "Framework/TestCreation.hpp"
#include "Framework/TestHelpers.hpp"
#include "Helpers/Evolution/Systems/GrMhd/GhValenciaDivClean/FiniteDifference/PrimReconstructor.hpp"
#include "NumericalAlgorithms/FiniteDifference/FallbackReconstructorType.hpp"

SPECTRE_TEST_CASE("Unit.Evolution.Systems.GrMhd.GhValenciaDivClean.Fd.Ppao",
                  "[Unit][Evolution]") {
  namespace helpers = TestHelpers::grmhd::GhValenciaDivClean::fd;
  PUPable_reg(SINGLE_ARG(
      grmhd::GhValenciaDivClean::fd::PositivityPreservingAdaptiveOrderPrim));
  const auto ppao_from_options_base = TestHelpers::test_factory_creation<
      grmhd::GhValenciaDivClean::fd::Reconstructor,
      grmhd::GhValenciaDivClean::fd::OptionTags::Reconstructor>(
      "PositivityPreservingAdaptiveOrderPrim:\n"
      "  Alpha5: 3.7\n"
      "  Alpha7: None\n"
      "  Alpha9: None\n"
      "  LowOrderReconstructor: MonotonisedCentral\n");
  const auto ppao_deserialized =
      serialize_and_deserialize(ppao_from_options_base);
  auto* const ppao_from_options =
      dynamic_cast<const grmhd::GhValenciaDivClean::fd::
                       PositivityPreservingAdaptiveOrderPrim*>(
          ppao_deserialized.get());
  REQUIRE(ppao_from_options != nullptr);
  CHECK(*ppao_from_options ==
        grmhd::GhValenciaDivClean::fd::PositivityPreservingAdaptiveOrderPrim{
            3.7, std::nullopt, std::nullopt,
            fd::reconstruction::FallbackReconstructorType::MonotonisedCentral});
  test_move_semantics(
      grmhd::GhValenciaDivClean::fd::PositivityPreservingAdaptiveOrderPrim{
          3.7, std::nullopt, std::nullopt,
          fd::reconstruction::FallbackReconstructorType::MonotonisedCentral},
      grmhd::GhValenciaDivClean::fd::PositivityPreservingAdaptiveOrderPrim{
          3.7, std::nullopt, std::nullopt,
          fd::reconstruction::FallbackReconstructorType::MonotonisedCentral});
  helpers::test_prim_reconstructor(10, *ppao_from_options);
}
