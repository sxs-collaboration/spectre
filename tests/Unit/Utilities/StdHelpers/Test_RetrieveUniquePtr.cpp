// Distributed under the MIT License.
// See LICENSE.txt for detai

#include "Framework/TestingFramework.hpp"

#include <memory>

#include "Utilities/StdHelpers/RetrieveUniquePtr.hpp"

SPECTRE_TEST_CASE("Unit.StdHelpers.RetrieveUniquePtr", "[Utilities][Unit]") {
  const int value = 7;
  CHECK(StdHelpers::retrieve(value) == 7);
  CHECK(&StdHelpers::retrieve(value) == &value);
  const auto pointer = std::make_unique<int>(10);
  CHECK(StdHelpers::retrieve(pointer) == 10);
  CHECK(&StdHelpers::retrieve(pointer) == pointer.get());

  int mutable_value = 1;
  CHECK(StdHelpers::retrieve(mutable_value) == 1);
  CHECK(&StdHelpers::retrieve(mutable_value) == &mutable_value);
  auto mutable_pointer = std::make_unique<int>(10);
  CHECK(StdHelpers::retrieve(mutable_pointer) == 10);
  CHECK(&StdHelpers::retrieve(mutable_pointer) == mutable_pointer.get());
  static_assert(
      std::is_same_v<decltype(StdHelpers::retrieve(mutable_pointer)), int&>);
  static_assert(
      std::is_same_v<decltype(StdHelpers::retrieve(mutable_value)), int&>);
}
