// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <optional>

#include "Evolution/Systems/ForceFree/FiniteDifference/AdaptiveOrder.hpp"
#include "Evolution/Systems/ForceFree/FiniteDifference/Factory.hpp"
#include "Evolution/Systems/ForceFree/FiniteDifference/Tags.hpp"
#include "Framework/TestCreation.hpp"
#include "Framework/TestHelpers.hpp"
#include "Helpers/Evolution/Systems/ForceFree/FiniteDifference/TestHelpers.hpp"
#include "NumericalAlgorithms/FiniteDifference/FallbackReconstructorType.hpp"

SPECTRE_TEST_CASE("Unit.Evolution.Systems.ForceFree.Fd.AdaptiveOrder",
                  "[Unit][Evolution]") {
  using AdaptiveOrder = ForceFree::fd::AdaptiveOrder;
  using Reconstructor = ForceFree::fd::Reconstructor;

  auto mc = fd::reconstruction::FallbackReconstructorType::MonotonisedCentral;

  TestHelpers::ForceFree::fd::test_reconstructor(
      8, AdaptiveOrder{4.0, 4.0, 4.0, mc});
  TestHelpers::ForceFree::fd::test_reconstructor(
      6, AdaptiveOrder{4.0, 4.0, std::nullopt, mc});
  TestHelpers::ForceFree::fd::test_reconstructor(
      8, AdaptiveOrder{4.0, std::nullopt, 4.0, mc});

  const AdaptiveOrder ao_recons{4.0, std::nullopt, std::nullopt, mc};
  TestHelpers::ForceFree::fd::test_reconstructor(4, ao_recons);

  const auto ao_from_options_base = TestHelpers::test_factory_creation<
      Reconstructor, ForceFree::fd::OptionTags::Reconstructor>(
      "AdaptiveOrder:\n"
      "  Alpha5: 4.0\n"
      "  Alpha7: None\n"
      "  Alpha9: None\n"
      "  LowOrderReconstructor: MonotonisedCentral\n");
  auto* const ao_from_options =
      dynamic_cast<const AdaptiveOrder*>(ao_from_options_base.get());
  REQUIRE(ao_from_options != nullptr);
  CHECK(*ao_from_options == ao_recons);

  CHECK(ao_recons != AdaptiveOrder(4.5, std::nullopt, std::nullopt, mc));
  CHECK(ao_recons != AdaptiveOrder(4.0, 4.0, std::nullopt, mc));
  CHECK(AdaptiveOrder(4.0, 4.0, std::nullopt, mc) !=
        AdaptiveOrder(4.0, 4.1, std::nullopt, mc));
  CHECK(ao_recons != AdaptiveOrder(4.0, std::nullopt, 4.0, mc));
  CHECK(AdaptiveOrder(4.0, std::nullopt, 4.0, mc) !=
        AdaptiveOrder(4.0, std::nullopt, 4.1, mc));
  CHECK(AdaptiveOrder(5.0, std::nullopt, 4.0, mc) ==
        AdaptiveOrder(5.0, std::nullopt, 4.0, mc));
  CHECK(AdaptiveOrder(5.0, 6.0, 4.0, mc) == AdaptiveOrder(5.0, 6.0, 4.0, mc));
  CHECK(AdaptiveOrder(5.0, 6.0, 4.0, mc) != AdaptiveOrder(5.1, 6.0, 4.0, mc));
  CHECK(AdaptiveOrder(5.0, 6.0, 4.0, mc) != AdaptiveOrder(5.0, 6.1, 4.0, mc));
  CHECK(AdaptiveOrder(5.0, 6.0, 4.0, mc) != AdaptiveOrder(5.0, 6.0, 4.1, mc));
  CHECK(AdaptiveOrder(5.0, 6.0, 4.0, mc) !=
        AdaptiveOrder(5.0, 6.0, 4.0,
                      fd::reconstruction::FallbackReconstructorType::Minmod));
  CHECK(ao_recons !=
        AdaptiveOrder(4.0, std::nullopt, std::nullopt,
                      fd::reconstruction::FallbackReconstructorType::Minmod));

  CHECK_THROWS_WITH(
      AdaptiveOrder(4.5, std::nullopt, std::nullopt,
                    fd::reconstruction::FallbackReconstructorType::None),
      Catch::Matchers::ContainsSubstring(
          "None is not an allowed low-order reconstructor."));
}
