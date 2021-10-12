// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <cstddef>
#include <memory>

#include "Evolution/Systems/ScalarAdvection/FiniteDifference/Factory.hpp"
#include "Evolution/Systems/ScalarAdvection/FiniteDifference/Reconstructor.hpp"
#include "Evolution/Systems/ScalarAdvection/FiniteDifference/Tags.hpp"
#include "Framework/TestCreation.hpp"
#include "Framework/TestHelpers.hpp"
#include "Helpers/Evolution/Systems/ScalarAdvection/FiniteDifference/TestHelpers.hpp"

namespace {
template <size_t Dim>
void test() {
  // test creation
  const auto aoweno = ScalarAdvection::fd::AoWeno53<Dim>(0.85, 0.8, 1e-12, 8);
  CHECK(aoweno ==
        TestHelpers::test_creation<ScalarAdvection::fd::AoWeno53<Dim>>(
            "GammaHi: 0.85\n"
            "GammaLo: 0.8\n"
            "Epsilon: 1.0e-12\n"
            "NonlinearWeightExponent: 8\n"));

  // test operators
  CHECK(aoweno == ScalarAdvection::fd::AoWeno53<Dim>(0.85, 0.8, 1.0e-12, 8));
  CHECK(aoweno != ScalarAdvection::fd::AoWeno53<Dim>(0.8, 0.8, 1.0e-12, 8));
  CHECK(aoweno != ScalarAdvection::fd::AoWeno53<Dim>(0.85, 0.85, 1.0e-12, 8));
  CHECK(aoweno != ScalarAdvection::fd::AoWeno53<Dim>(0.85, 0.8, 2.0e-12, 8));
  CHECK(aoweno != ScalarAdvection::fd::AoWeno53<Dim>(0.85, 0.8, 1.0e-12, 6));

  // test serialization
  test_serialization(aoweno);

  // test derived
  const auto aoweno_from_options_base = TestHelpers::test_factory_creation<
      ScalarAdvection::fd::Reconstructor<Dim>,
      ScalarAdvection::fd::OptionTags::Reconstructor<Dim>>(
      "AoWeno53:\n"
      "  GammaHi: 0.85\n"
      "  GammaLo: 0.8\n"
      "  Epsilon: 1.0e-12\n"
      "  NonlinearWeightExponent: 8\n");
  auto* const aoweno_from_options =
      dynamic_cast<const ScalarAdvection::fd::AoWeno53<Dim>*>(
          aoweno_from_options_base.get());
  REQUIRE(aoweno_from_options != nullptr);
  CHECK(*aoweno_from_options == aoweno);

  // test reconstruction
  const size_t num_pts_per_dimension = 5;
  TestHelpers::ScalarAdvection::fd::test_reconstructor<Dim>(
      num_pts_per_dimension, aoweno);
}
}  // namespace

SPECTRE_TEST_CASE("Unit.Evolution.Systems.ScalarAdvection.Fd.AoWeno53",
                  "[Unit][Evolution]") {
  test<1>();
  test<2>();
}
