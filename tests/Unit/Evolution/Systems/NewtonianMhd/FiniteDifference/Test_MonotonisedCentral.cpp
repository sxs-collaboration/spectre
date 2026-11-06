// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include "Evolution/Systems/NewtonianMhd/FiniteDifference/Factory.hpp"
#include "Evolution/Systems/NewtonianMhd/FiniteDifference/MonotonisedCentral.hpp"
#include "Evolution/Systems/NewtonianMhd/FiniteDifference/Tag.hpp"
#include "Framework/TestCreation.hpp"
#include "Helpers/Evolution/Systems/NewtonianMhd/FiniteDifference/PrimReconstructor.hpp"

namespace {
template <size_t Dim>
void test() {
  namespace helpers = TestHelpers::NewtonianMhd::fd;
  const NewtonianMhd::fd::MonotonisedCentralPrim<Dim> mc_recons{};
  helpers::test_prim_reconstructor<Dim>(5, mc_recons);
  const auto mc_from_options_base = TestHelpers::test_factory_creation<
      NewtonianMhd::fd::Reconstructor<Dim>,
      NewtonianMhd::fd::OptionTags::Reconstructor<Dim>>(
      "MonotonisedCentralPrim:\n");
  auto* const mc_from_options =
      dynamic_cast<const NewtonianMhd::fd::MonotonisedCentralPrim<Dim>*>(
          mc_from_options_base.get());
  REQUIRE(mc_from_options != nullptr);
  CHECK(*mc_from_options == mc_recons);
}
}  // namespace

SPECTRE_TEST_CASE(
    "Unit.Evolution.Systems.NewtonianMhd.Fd.MonotonisedCentralPrim",
    "[Unit][Evolution]") {
  test<1>();
  test<2>();
  test<3>();
}
