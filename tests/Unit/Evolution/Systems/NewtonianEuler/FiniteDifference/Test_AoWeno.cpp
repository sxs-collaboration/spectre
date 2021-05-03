// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include "Evolution/Systems/NewtonianEuler/FiniteDifference/AoWeno.hpp"
#include "Evolution/Systems/NewtonianEuler/FiniteDifference/Factory.hpp"
#include "Evolution/Systems/NewtonianEuler/FiniteDifference/Tag.hpp"
#include "Framework/TestCreation.hpp"
#include "Helpers/Evolution/Systems/NewtonianEuler/FiniteDifference/PrimReconstructor.hpp"

namespace {
template <size_t Dim>
void test() {
  namespace helpers = TestHelpers::NewtonianEuler::fd;
  const NewtonianEuler::fd::AoWeno53Prim<Dim> aoweno_recons{0.85, 0.8, 1.0e-12,
                                                            8};
  helpers::test_prim_reconstructor<Dim>(5, aoweno_recons);

  const auto aoweno_from_options_base = TestHelpers::test_factory_creation<
      NewtonianEuler::fd::Reconstructor<Dim>,
      NewtonianEuler::fd::OptionTags::Reconstructor<Dim>>(
      "AoWeno53Prim:\n"
      "  GammaHi: 0.85\n"
      "  GammaLo: 0.8\n"
      "  Epsilon: 1.0e-12\n"
      "  NonlinearWeightExponent: 8\n");
  auto* const aoweno_from_options =
      dynamic_cast<const NewtonianEuler::fd::AoWeno53Prim<Dim>*>(
          aoweno_from_options_base.get());
  REQUIRE(aoweno_from_options != nullptr);
  CHECK(*aoweno_from_options == aoweno_recons);

  CHECK(aoweno_recons !=
        NewtonianEuler::fd::AoWeno53Prim<Dim>(0.8, 0.8, 1.0e-12, 8));
  CHECK(aoweno_recons !=
        NewtonianEuler::fd::AoWeno53Prim<Dim>(0.85, 0.85, 1.0e-12, 8));
  CHECK(aoweno_recons !=
        NewtonianEuler::fd::AoWeno53Prim<Dim>(0.85, 0.8, 2.0e-12, 8));
  CHECK(aoweno_recons !=
        NewtonianEuler::fd::AoWeno53Prim<Dim>(0.85, 0.8, 1.0e-12, 6));
  CHECK(aoweno_recons ==
        NewtonianEuler::fd::AoWeno53Prim<Dim>(0.85, 0.8, 1.0e-12, 8));
}
}  // namespace

SPECTRE_TEST_CASE("Unit.Evolution.Systems.NewtonianEuler.Fd.AoWeno53Prim",
                  "[Unit][Evolution]") {
  test<1>();
  test<2>();
  test<3>();
}
