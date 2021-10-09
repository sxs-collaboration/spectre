// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include "Evolution/Systems/GrMhd/GhValenciaDivClean/FiniteDifference/Factory.hpp"
#include "Evolution/Systems/GrMhd/GhValenciaDivClean/FiniteDifference/MonotisedCentral.hpp"
#include "Evolution/Systems/GrMhd/GhValenciaDivClean/FiniteDifference/Tag.hpp"
#include "Framework/TestCreation.hpp"
#include "Framework/TestHelpers.hpp"
#include "Helpers/Evolution/Systems/GrMhd/GhValenciaDivClean/FiniteDifference/PrimReconstructor.hpp"

SPECTRE_TEST_CASE(
    "Unit.Evolution.Systems.GrMhd.GhValenciaDivClean.Fd.MonotisedCentralPrim",
    "[Unit][Evolution]") {
  namespace helpers = TestHelpers::grmhd::GhValenciaDivClean::fd;
  PUPable_reg(SINGLE_ARG(grmhd::GhValenciaDivClean::fd::MonotisedCentralPrim));
  const auto mc_from_options_base = TestHelpers::test_factory_creation<
      grmhd::GhValenciaDivClean::fd::Reconstructor,
      grmhd::GhValenciaDivClean::fd::OptionTags::Reconstructor>(
      "MonotisedCentralPrim:\n");
  const auto mc_deserialized = serialize_and_deserialize(mc_from_options_base);
  auto* const mc_from_options =
      dynamic_cast<const grmhd::GhValenciaDivClean::fd::MonotisedCentralPrim*>(
          mc_deserialized.get());
  REQUIRE(mc_from_options != nullptr);
  CHECK(*mc_from_options ==
        grmhd::GhValenciaDivClean::fd::MonotisedCentralPrim{});
  test_move_semantics(grmhd::GhValenciaDivClean::fd::MonotisedCentralPrim{},
                      grmhd::GhValenciaDivClean::fd::MonotisedCentralPrim{});
  helpers::test_prim_reconstructor(5, *mc_from_options);
}
