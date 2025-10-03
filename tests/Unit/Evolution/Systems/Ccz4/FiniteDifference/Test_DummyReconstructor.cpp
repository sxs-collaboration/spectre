// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include "Evolution/Systems/Ccz4/FiniteDifference/DummyReconstructor.hpp"
#include "Evolution/Systems/Ccz4/FiniteDifference/Tags.hpp"
#include "Framework/TestCreation.hpp"

namespace Ccz4::fd {
namespace {

SPECTRE_TEST_CASE("Unit.Evolution.Systems.Ccz4.Fd.DummyReconstructor",
                  "[Unit][Evolution]") {
  const auto dummy_from_options_base =
      TestHelpers::test_factory_creation<Ccz4::fd::Reconstructor,
                                         Ccz4::fd::OptionTags::Reconstructor>(
          "DummyReconstructor:\n");
  auto* const dummy_from_options =
      dynamic_cast<const Ccz4::fd::DummyReconstructor*>(
          dummy_from_options_base.get());
  REQUIRE(dummy_from_options != nullptr);
  CHECK(dummy_from_options->ghost_zone_size() == 3);
}

}  // namespace
}  // namespace Ccz4::fd
