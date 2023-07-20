// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <memory>
#include <utility>

#include "Evolution/Systems/ForceFree/FiniteDifference/Factory.hpp"
#include "Evolution/Systems/ForceFree/FiniteDifference/MonotonisedCentral.hpp"
#include "Evolution/Systems/ForceFree/FiniteDifference/Tags.hpp"
#include "Framework/TestCreation.hpp"
#include "Framework/TestHelpers.hpp"
#include "Helpers/Evolution/Systems/ForceFree/FiniteDifference/TestHelpers.hpp"

namespace {
SPECTRE_TEST_CASE("Unit.Evolution.Systems.ForceFree.Fd.MonotonisedCentral",
                  "[Unit][Evolution]") {
  using MonotonisedCentral = ForceFree::fd::MonotonisedCentral;
  using Reconstructor = ForceFree::fd::Reconstructor;

  const MonotonisedCentral mc_recons{};
  //   TestHelpers::ForceFree::fd::test_reconstructor(5, mc_recons);

  const auto mc_from_options_base = TestHelpers::test_factory_creation<
      Reconstructor, ForceFree::fd::OptionTags::Reconstructor>(
      "MonotonisedCentral:\n");
  auto* const mc_from_options =
      dynamic_cast<const MonotonisedCentral*>(mc_from_options_base.get());
  REQUIRE(mc_from_options != nullptr);
  CHECK(*mc_from_options == mc_recons);
}
}  // namespace
