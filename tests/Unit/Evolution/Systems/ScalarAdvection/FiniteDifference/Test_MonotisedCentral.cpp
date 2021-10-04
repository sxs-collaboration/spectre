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
void test_create() {
  const auto mc =
      TestHelpers::test_creation<ScalarAdvection::fd::MonotisedCentral<Dim>>(
          "");
  CHECK(mc == ScalarAdvection::fd::MonotisedCentral<Dim>());
}

template <size_t Dim>
void test_serialize() {
  ScalarAdvection::fd::MonotisedCentral<Dim> mc;
  test_serialization(mc);
}

template <size_t Dim>
void test_move() {
  ScalarAdvection::fd::MonotisedCentral<Dim> mc;
  ScalarAdvection::fd::MonotisedCentral<Dim> mc_copy;
  test_move_semantics(std::move(mc), mc_copy);  //  NOLINT
}

template <size_t Dim>
void test_derived() {
  const ScalarAdvection::fd::MonotisedCentral<Dim> mc_recons{};
  const auto mc_from_options_base = TestHelpers::test_factory_creation<
      ScalarAdvection::fd::Reconstructor<Dim>,
      ScalarAdvection::fd::OptionTags::Reconstructor<Dim>>(
      "MonotisedCentral:\n");
  auto* const mc_from_options =
      dynamic_cast<const ScalarAdvection::fd::MonotisedCentral<Dim>*>(
          mc_from_options_base.get());
  REQUIRE(mc_from_options != nullptr);
  CHECK(*mc_from_options == mc_recons);
}

template <size_t Dim>
void test_mc() {
  const ScalarAdvection::fd::MonotisedCentral<Dim> mc_recons{};
  const size_t num_pts_per_dimension = 5;
  TestHelpers::ScalarAdvection::fd::test_reconstructor<Dim>(
      num_pts_per_dimension, mc_recons);
}

}  // namespace

SPECTRE_TEST_CASE("Unit.Evolution.Systems.ScalarAdvection.Fd.MonotisedCentral",
                  "[Unit][Evolution]") {
  test_create<1>();
  test_create<2>();
  test_serialize<1>();
  test_serialize<2>();
  test_move<1>();
  test_move<2>();
  test_derived<1>();
  test_derived<2>();

  test_mc<1>();
  test_mc<2>();
}
