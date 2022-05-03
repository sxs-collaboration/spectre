// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <optional>

#include "Framework/TestCreation.hpp"
#include "Framework/TestHelpers.hpp"
#include "Parallel/ResourceInfo.hpp"

namespace Parallel {
namespace {
void test_singleton_info() {
  SingletonInfoHolder info_holder{};
  CHECK(info_holder.proc() == std::nullopt);
  CHECK_FALSE(info_holder.is_exclusive());

  info_holder = TestHelpers::test_creation<Parallel::SingletonInfoHolder>(
      "Proc: 0\n"
      "Exclusive: false\n");
  CHECK(info_holder.proc().value() == 0);
  CHECK_FALSE(info_holder.is_exclusive());

  auto serialized_info_holder = serialize_and_deserialize(info_holder);
  CHECK(info_holder.proc().value() == serialized_info_holder.proc().value());
  CHECK(info_holder.is_exclusive() == serialized_info_holder.is_exclusive());

  info_holder = TestHelpers::test_creation<Parallel::SingletonInfoHolder>(
      "Proc: Auto\n"
      "Exclusive: false\n");
  CHECK(info_holder.proc() == std::nullopt);
  CHECK_FALSE(info_holder.is_exclusive());

  info_holder = TestHelpers::test_creation<Parallel::SingletonInfoHolder>(
      "Proc: 4\n"
      "Exclusive: true\n");
  CHECK(info_holder.proc().value() == 4);
  CHECK(info_holder.is_exclusive());

  CHECK_THROWS_WITH(
      ([]() {
        auto info_holder_error =
            TestHelpers::test_creation<Parallel::SingletonInfoHolder>(
                "Proc: -2\n"
                "Exclusive: true\n");
        (void)info_holder_error;
      })(),
      Catch::Contains("Proc must be a non-negative integer."));
}
}  // namespace

SPECTRE_TEST_CASE("Unit.Parallel.ResourceInfo", "[Unit][Parallel]") {
  test_singleton_info();
}
}  // namespace Parallel
