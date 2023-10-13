// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include "Evolution/Systems/GrMhd/GhValenciaDivClean/FiniteDifference/Factory.hpp"
#include "Evolution/Systems/GrMhd/GhValenciaDivClean/FiniteDifference/Tag.hpp"
#include "Evolution/Systems/GrMhd/GhValenciaDivClean/FiniteDifference/Wcns5z.hpp"
#include "Framework/TestCreation.hpp"
#include "Framework/TestHelpers.hpp"
#include "Helpers/Evolution/Systems/GrMhd/GhValenciaDivClean/FiniteDifference/PrimReconstructor.hpp"
#include "NumericalAlgorithms/FiniteDifference/FallbackReconstructorType.hpp"

SPECTRE_TEST_CASE("Unit.Evolution.Systems.GrMhd.GhValenciaDivClean.Fd.Wcns5z",
                  "[Unit][Evolution]") {
  namespace helpers = TestHelpers::grmhd::GhValenciaDivClean::fd;
  PUPable_reg(SINGLE_ARG(grmhd::GhValenciaDivClean::fd::Wcns5zPrim));
  const auto wcns5z_from_options_base = TestHelpers::test_factory_creation<
      grmhd::GhValenciaDivClean::fd::Reconstructor,
      grmhd::GhValenciaDivClean::fd::OptionTags::Reconstructor>(
      "Wcns5zPrim:\n"
      "  NonlinearWeightExponent: 2\n"
      "  Epsilon: 1.e-42\n"
      "  FallbackReconstructor: None\n"
      "  MaxNumberOfExtrema: 0\n");
  const auto wcns5z_deserialized =
      serialize_and_deserialize(wcns5z_from_options_base);
  auto* const wcns5z_from_options =
      dynamic_cast<const grmhd::GhValenciaDivClean::fd::Wcns5zPrim*>(
          wcns5z_deserialized.get());
  REQUIRE(wcns5z_from_options != nullptr);
  CHECK(*wcns5z_from_options ==
        grmhd::GhValenciaDivClean::fd::Wcns5zPrim{
            2, 1.e-42, fd::reconstruction::FallbackReconstructorType::None, 0});
  test_move_semantics(
      grmhd::GhValenciaDivClean::fd::Wcns5zPrim{
          2, 1.e-42, fd::reconstruction::FallbackReconstructorType::None, 0},
      grmhd::GhValenciaDivClean::fd::Wcns5zPrim{
          2, 1.e-42, fd::reconstruction::FallbackReconstructorType::None, 0});
  helpers::test_prim_reconstructor(7, *wcns5z_from_options);
}
