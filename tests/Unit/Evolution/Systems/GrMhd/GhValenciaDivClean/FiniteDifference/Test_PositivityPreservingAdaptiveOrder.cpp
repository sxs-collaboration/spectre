// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include "Evolution/Systems/GrMhd/GhValenciaDivClean/FiniteDifference/Factory.hpp"
#include "Evolution/Systems/GrMhd/GhValenciaDivClean/FiniteDifference/PositivityPreservingAdaptiveOrder.hpp"
#include "Evolution/Systems/GrMhd/GhValenciaDivClean/FiniteDifference/Reconstructor.hpp"
#include "Evolution/Systems/GrMhd/GhValenciaDivClean/FiniteDifference/Tag.hpp"
#include "Evolution/Systems/RadiationTransport/NoNeutrinos/System.hpp"
#include "Evolution/VariableFixing/FixToAtmosphere.hpp"
#include "Framework/TestCreation.hpp"
#include "Framework/TestHelpers.hpp"
#include "Helpers/Evolution/Systems/GrMhd/GhValenciaDivClean/FiniteDifference/PrimReconstructor.hpp"
#include "NumericalAlgorithms/FiniteDifference/FallbackReconstructorType.hpp"

SPECTRE_TEST_CASE("Unit.Evolution.Systems.GrMhd.GhValenciaDivClean.Fd.Ppao",
                  "[Unit][Evolution]") {
  using NeutrinoTransportSystem = RadiationTransport::NoNeutrinos::System;
  using System = grmhd::GhValenciaDivClean::System<NeutrinoTransportSystem>;

  namespace helpers = TestHelpers::grmhd::GhValenciaDivClean::fd;
  PUPable_reg(SINGLE_ARG(
      grmhd::GhValenciaDivClean::fd::PositivityPreservingAdaptiveOrderPrim<
          System>));
  const auto ppao_from_options_base = TestHelpers::test_factory_creation<
      grmhd::GhValenciaDivClean::fd::Reconstructor<System>,
      grmhd::GhValenciaDivClean::fd::OptionTags::Reconstructor<System>>(
      "PositivityPreservingAdaptiveOrderPrim:\n"
      "  Alpha5: 3.7\n"
      "  Alpha7: None\n"
      "  Alpha9: None\n"
      "  LowOrderReconstructor: MonotonisedCentral\n"
      "  AtmosphereTreatment: Never\n"
      "  ReconstructRhoTimesTemperature: true\n");
  const auto ppao_deserialized =
      serialize_and_deserialize(ppao_from_options_base);
  auto* const ppao_from_options = dynamic_cast<
      const grmhd::GhValenciaDivClean::fd::
          PositivityPreservingAdaptiveOrderPrim<System>*>(
      ppao_deserialized.get());
  REQUIRE(ppao_from_options != nullptr);
  CHECK(grmhd::GhValenciaDivClean::fd::PositivityPreservingAdaptiveOrderPrim<
            System>{
            3.7, std::nullopt, std::nullopt,
            fd::reconstruction::FallbackReconstructorType::MonotonisedCentral,
            ::VariableFixing::FixReconstructedStateToAtmosphere::Always,
            false} !=
        grmhd::GhValenciaDivClean::fd::PositivityPreservingAdaptiveOrderPrim<
            System>{
            3.7, std::nullopt, std::nullopt,
            fd::reconstruction::FallbackReconstructorType::MonotonisedCentral,
            ::VariableFixing::FixReconstructedStateToAtmosphere::Never, false});
  CHECK(
      grmhd::GhValenciaDivClean::fd::PositivityPreservingAdaptiveOrderPrim<
          System>{
          3.7, std::nullopt, std::nullopt,
          fd::reconstruction::FallbackReconstructorType::MonotonisedCentral,
          ::VariableFixing::FixReconstructedStateToAtmosphere::Always, false} !=
      grmhd::GhValenciaDivClean::fd::PositivityPreservingAdaptiveOrderPrim<
          System>{
          3.8, std::nullopt, std::nullopt,
          fd::reconstruction::FallbackReconstructorType::MonotonisedCentral,
          ::VariableFixing::FixReconstructedStateToAtmosphere::Always, false});
  // Can't use high-order reconstruction yet. We'll enable these tests later.
  //
  // CHECK(
  //     grmhd::GhValenciaDivClean::fd::PositivityPreservingAdaptiveOrderPrim<
  //         System>{
  //         3.7, std::nullopt, std::nullopt,
  //         fd::reconstruction::FallbackReconstructorType::MonotonisedCentral,
  //         ::VariableFixing::FixReconstructedStateToAtmosphere::Always, false}
  //         !=
  //     grmhd::GhValenciaDivClean::fd::PositivityPreservingAdaptiveOrderPrim<
  //         System>{
  //         3.7, 3.5, std::nullopt,
  //         fd::reconstruction::FallbackReconstructorType::MonotonisedCentral,
  //         ::VariableFixing::FixReconstructedStateToAtmosphere::Always,
  //         false});
  // CHECK(
  //     grmhd::GhValenciaDivClean::fd::PositivityPreservingAdaptiveOrderPrim<
  //         System>{
  //         3.7, std::nullopt, std::nullopt,
  //         fd::reconstruction::FallbackReconstructorType::MonotonisedCentral,
  //         ::VariableFixing::FixReconstructedStateToAtmosphere::Always, false}
  //         !=
  //     grmhd::GhValenciaDivClean::fd::PositivityPreservingAdaptiveOrderPrim<
  //         System>{
  //         3.7, std::nullopt, 3.6,
  //         fd::reconstruction::FallbackReconstructorType::MonotonisedCentral,
  //         ::VariableFixing::FixReconstructedStateToAtmosphere::Always,
  //         false});
  CHECK(grmhd::GhValenciaDivClean::fd::PositivityPreservingAdaptiveOrderPrim<
            System>{
            3.7, std::nullopt, std::nullopt,
            fd::reconstruction::FallbackReconstructorType::MonotonisedCentral,
            ::VariableFixing::FixReconstructedStateToAtmosphere::Always,
            false} !=
        grmhd::GhValenciaDivClean::fd::PositivityPreservingAdaptiveOrderPrim<
            System>{
            3.7, std::nullopt, std::nullopt,
            fd::reconstruction::FallbackReconstructorType::MonotonisedCentral,
            ::VariableFixing::FixReconstructedStateToAtmosphere::Always, true});
  CHECK(*ppao_from_options ==
        grmhd::GhValenciaDivClean::fd::PositivityPreservingAdaptiveOrderPrim<
            System>{
            3.7, std::nullopt, std::nullopt,
            fd::reconstruction::FallbackReconstructorType::MonotonisedCentral,
            ::VariableFixing::FixReconstructedStateToAtmosphere::Never, true});
  test_move_semantics(
      grmhd::GhValenciaDivClean::fd::PositivityPreservingAdaptiveOrderPrim<
          System>{
          3.7, std::nullopt, std::nullopt,
          fd::reconstruction::FallbackReconstructorType::MonotonisedCentral,
          ::VariableFixing::FixReconstructedStateToAtmosphere::Never, true},
      grmhd::GhValenciaDivClean::fd::PositivityPreservingAdaptiveOrderPrim<
          System>{
          3.7, std::nullopt, std::nullopt,
          fd::reconstruction::FallbackReconstructorType::MonotonisedCentral,
          ::VariableFixing::FixReconstructedStateToAtmosphere::Never, true});
  helpers::test_prim_reconstructor(10, *ppao_from_options);
}
