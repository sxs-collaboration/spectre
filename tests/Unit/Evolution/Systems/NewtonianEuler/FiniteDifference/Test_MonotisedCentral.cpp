// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include "Evolution/Systems/NewtonianEuler/FiniteDifference/Factory.hpp"
#include "Evolution/Systems/NewtonianEuler/FiniteDifference/MonotisedCentral.hpp"
#include "Evolution/Systems/NewtonianEuler/FiniteDifference/Tag.hpp"
#include "Framework/TestCreation.hpp"
#include "Helpers/Evolution/Systems/NewtonianEuler/FiniteDifference/PrimReconstructor.hpp"

namespace {
template <size_t Dim>
void test() {
  namespace helpers = TestHelpers::NewtonianEuler::fd;
  const NewtonianEuler::fd::MonotisedCentralPrim<Dim> mc_recons{};
  helpers::test_prim_reconstructor<Dim>(5, mc_recons);
  const auto mc_from_options_base = TestHelpers::test_factory_creation<
      NewtonianEuler::fd::Reconstructor<Dim>,
      NewtonianEuler::fd::OptionTags::Reconstructor<Dim>>(
      "MonotisedCentralPrim:\n");
  auto* const mc_from_options =
      dynamic_cast<const NewtonianEuler::fd::MonotisedCentralPrim<Dim>*>(
          mc_from_options_base.get());
  REQUIRE(mc_from_options != nullptr);
  CHECK(*mc_from_options == mc_recons);
}
}  // namespace

SPECTRE_TEST_CASE(
    "Unit.Evolution.Systems.NewtonianEuler.Fd.MonotisedCentralPrim",
    "[Unit][Evolution]") {
  test<1>();
  test<2>();
  test<3>();
}
