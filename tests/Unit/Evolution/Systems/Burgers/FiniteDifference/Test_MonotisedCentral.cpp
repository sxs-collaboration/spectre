// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <cstddef>
#include <utility>

#include "Evolution/Systems/Burgers/FiniteDifference/Factory.hpp"
#include "Evolution/Systems/Burgers/FiniteDifference/Reconstructor.hpp"
#include "Evolution/Systems/Burgers/FiniteDifference/Tags.hpp"
#include "Framework/TestCreation.hpp"
#include "Framework/TestHelpers.hpp"
#include "Helpers/Evolution/Systems/Burgers/FiniteDifference/TestHelpers.hpp"

SPECTRE_TEST_CASE("Unit.Evolution.Systems.Burgers.Fd.MonotisedCentral",
                  "[Unit][Evolution]") {
  auto mc = TestHelpers::test_creation<Burgers::fd::MonotisedCentral>("");
  CHECK(mc == Burgers::fd::MonotisedCentral());

  test_serialization(mc);

  const auto mc_from_options_base = TestHelpers::test_factory_creation<
      Burgers::fd::Reconstructor, Burgers::fd::OptionTags::Reconstructor>(
      "MonotisedCentral:\n");
  auto* const mc_from_options =
      dynamic_cast<const Burgers::fd::MonotisedCentral*>(
          mc_from_options_base.get());
  REQUIRE(mc_from_options != nullptr);
  CHECK(*mc_from_options == mc);

  const size_t num_pts = 5;
  TestHelpers::Burgers::fd::test_reconstructor(num_pts, mc);

  Burgers::fd::MonotisedCentral mc_copy;
  test_move_semantics(std::move(mc), mc_copy);  //  NOLINT
}
