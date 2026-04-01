// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include "Evolution/Systems/GrMhd/GhValenciaDivClean/FiniteDifference/Factory.hpp"
#include "Evolution/Systems/GrMhd/GhValenciaDivClean/FiniteDifference/MonotonisedCentral.hpp"
#include "Evolution/Systems/GrMhd/GhValenciaDivClean/FiniteDifference/Tag.hpp"
#include "Evolution/Systems/RadiationTransport/NoNeutrinos/System.hpp"
#include "Evolution/VariableFixing/FixToAtmosphere.hpp"
#include "Framework/TestCreation.hpp"
#include "Framework/TestHelpers.hpp"
#include "Helpers/Evolution/Systems/GrMhd/GhValenciaDivClean/FiniteDifference/PrimReconstructor.hpp"
#include "Utilities/Serialization/Serialize.hpp"

SPECTRE_TEST_CASE(
    "Unit.Evolution.Systems.GrMhd.GhValenciaDivClean.Fd.MonotonisedCentralPrim",
    "[Unit][Evolution]") {
  namespace helpers = TestHelpers::grmhd::GhValenciaDivClean::fd;
  using NeutrinoTransportSystem = RadiationTransport::NoNeutrinos::System;
  using System = grmhd::GhValenciaDivClean::System<NeutrinoTransportSystem>;

  PUPable_reg(SINGLE_ARG(grmhd::GhValenciaDivClean::fd::MonotonisedCentralPrim<
                         System>));
  const auto mc_from_options_base = TestHelpers::test_factory_creation<
      grmhd::GhValenciaDivClean::fd::Reconstructor<System>,
      grmhd::GhValenciaDivClean::fd::OptionTags::Reconstructor<System>>(
      "MonotonisedCentralPrim:\n"
      "  AtmosphereTreatment: Never\n"
      "  ReconstructRhoTimesTemperature: false\n");
  const auto mc_deserialized = serialize_and_deserialize(mc_from_options_base);
  auto* const mc_from_options =
      dynamic_cast<const grmhd::GhValenciaDivClean::fd::MonotonisedCentralPrim<
          System>*>(mc_deserialized.get());
  REQUIRE(mc_from_options != nullptr);
  CHECK(
      grmhd::GhValenciaDivClean::fd::MonotonisedCentralPrim<System>{
          ::VariableFixing::FixReconstructedStateToAtmosphere::Always, false} !=
      grmhd::GhValenciaDivClean::fd::MonotonisedCentralPrim<System>{
          ::VariableFixing::FixReconstructedStateToAtmosphere::Never, false});
  CHECK(
      grmhd::GhValenciaDivClean::fd::MonotonisedCentralPrim<System>{
          ::VariableFixing::FixReconstructedStateToAtmosphere::Always, false} !=
      grmhd::GhValenciaDivClean::fd::MonotonisedCentralPrim<System>{
          ::VariableFixing::FixReconstructedStateToAtmosphere::Always, true});
  CHECK(*mc_from_options ==
        grmhd::GhValenciaDivClean::fd::MonotonisedCentralPrim<System>{
            ::VariableFixing::FixReconstructedStateToAtmosphere::Never, false});
  test_move_semantics(
      grmhd::GhValenciaDivClean::fd::MonotonisedCentralPrim<System>{
          ::VariableFixing::FixReconstructedStateToAtmosphere::Never, false},
      grmhd::GhValenciaDivClean::fd::MonotonisedCentralPrim<System>{
          ::VariableFixing::FixReconstructedStateToAtmosphere::Never, false});

  helpers::test_prim_reconstructor(5, *mc_from_options);
}
