// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include "Evolution/Systems/ForceFree/FiniteDifference/Factory.hpp"
#include "Evolution/Systems/ForceFree/FiniteDifference/Tags.hpp"
#include "Evolution/Systems/ForceFree/FiniteDifference/Wcns5z.hpp"
#include "Framework/TestCreation.hpp"
#include "Framework/TestHelpers.hpp"
#include "Helpers/Evolution/Systems/ForceFree/FiniteDifference/TestHelpers.hpp"
#include "NumericalAlgorithms/FiniteDifference/FallbackReconstructorType.hpp"

SPECTRE_TEST_CASE("Unit.Evolution.Systems.ForceFree.Fd.Wcns5z",
                  "[Unit][Evolution]") {
  using Wcns5z = ForceFree::fd::Wcns5z;
  using Reconstructor = ForceFree::fd::Reconstructor;

  auto mc = fd::reconstruction::FallbackReconstructorType::MonotonisedCentral;

  const Wcns5z wcns5z_recons{2, 2.0e-16, mc, 1};
  TestHelpers::ForceFree::fd::test_reconstructor(5, wcns5z_recons);

  const auto wcns5z_from_options_base = TestHelpers::test_factory_creation<
      Reconstructor, ForceFree::fd::OptionTags::Reconstructor>(
      "Wcns5z:\n"
      "  NonlinearWeightExponent: 2\n"
      "  Epsilon: 2.0e-16\n"
      "  FallbackReconstructor: MonotonisedCentral\n"
      "  MaxNumberOfExtrema: 1\n");
  auto* const wcns5z_from_options =
      dynamic_cast<const Wcns5z*>(wcns5z_from_options_base.get());
  REQUIRE(wcns5z_from_options != nullptr);
  CHECK(*wcns5z_from_options == wcns5z_recons);

  CHECK(wcns5z_recons != Wcns5z(1, 2.0e-16, mc, 1));
  CHECK(wcns5z_recons != Wcns5z(2, 1.0e-16, mc, 1));
  CHECK(wcns5z_recons !=
        Wcns5z(2, 2.0e-16, fd::reconstruction::FallbackReconstructorType::None,
               1));
  CHECK(wcns5z_recons != Wcns5z(2, 2.0e-16, mc, 2));
}
