// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <cmath>
#include <limits>
#include <ostream>
#include <string>
#include <type_traits>

#include "DataStructures/DataVector.hpp"
#include "Utilities/Expected.hpp"

namespace {
enum class SqrtError { Negative, Infinite };

std::ostream& operator<<(std::ostream& os, const SqrtError error) {
  return os << (error == SqrtError::Negative ? "Negative" : "Infinite");
}

// A type without a stream operator
struct Unprintable {
  int code{};
};

// [expected_example]
Expected<double, SqrtError> checked_sqrt(const double x) {
  if (x < 0.0) {
    return Unexpected{SqrtError::Negative};
  }
  if (x == std::numeric_limits<double>::infinity()) {
    return Unexpected{SqrtError::Infinite};
  }
  return sqrt(x);
}

double sqrt_or_zero(const double x) {
  const auto result = checked_sqrt(x);
  if (not result.has_value()) {
    return result.error() == SqrtError::Negative ? 0.0 : x;
  }
  return result.value();
}
// [expected_example]

void test_value_and_error() {
  const auto root = checked_sqrt(4.0);
  CHECK(root.has_value());
  CHECK(static_cast<bool>(root));
  CHECK(root.value() == 2.0);
  CHECK_THROWS_WITH(root.error(), Catch::Matchers::ContainsSubstring(
                                      "Expected holds a value, not an error."));

  const auto negative = checked_sqrt(-1.0);
  CHECK_FALSE(negative.has_value());
  CHECK_FALSE(static_cast<bool>(negative));
  CHECK(negative.error() == SqrtError::Negative);
  CHECK(checked_sqrt(std::numeric_limits<double>::infinity()).error() ==
        SqrtError::Infinite);

  CHECK(sqrt_or_zero(9.0) == 3.0);
  CHECK(sqrt_or_zero(-9.0) == 0.0);
}

void test_error_messages() {
  // The message includes the error if it can be streamed
  CHECK_THROWS_WITH(
      checked_sqrt(-1.0).value(),
      Catch::Matchers::ContainsSubstring(
          "Expected holds an error, not a value. The error is:\nNegative"));
  const Expected<double, Unprintable> unprintable = Unexpected{Unprintable{3}};
  CHECK(unprintable.error().code == 3);
  CHECK_THROWS_WITH(unprintable.value(),
                    Catch::Matchers::ContainsSubstring(
                        "Expected holds an error, not a value."));
}

void test_same_value_and_error_type() {
  // `Unexpected` distinguishes errors from values of the same type
  const Expected<std::string, std::string> value{"value"};
  CHECK(value.has_value());
  CHECK(value.value() == "value");
  const Expected<std::string, std::string> error =
      Unexpected{std::string{"error"}};
  CHECK_FALSE(error.has_value());
  CHECK(error.error() == "error");
}

void test_non_trivial_value() {
  Expected<DataVector, SqrtError> vector{DataVector{1.0, 2.0}};
  CHECK(vector.value() == DataVector{1.0, 2.0});
  // Modify the held value
  vector.value()[0] = 3.0;
  CHECK(vector.value() == DataVector{3.0, 2.0});
  // Copy and move the held value out
  const auto copy = vector;
  CHECK(copy.value() == DataVector{3.0, 2.0});
  const DataVector moved = std::move(vector).value();
  CHECK(moved == DataVector{3.0, 2.0});

  const Expected<DataVector, SqrtError> error = Unexpected{SqrtError::Infinite};
  CHECK(error.error() == SqrtError::Infinite);
  CHECK_THROWS_WITH(error.value(), Catch::Matchers::ContainsSubstring(
                                       "The error is:\nInfinite"));
}

// Expected of trivially copyable types stays trivially copyable, so it can be
// copied to and from the device
static_assert(std::is_trivially_copyable_v<Expected<double, SqrtError>>);
static_assert(not std::is_trivially_copyable_v<Expected<DataVector, int>>);

// Expected can be used in constant expressions
static_assert(Expected<int, SqrtError>{2}.value() == 2);
static_assert(not Expected<int, SqrtError>{Unexpected{SqrtError::Negative}});
static_assert(Expected<int, SqrtError>{Unexpected{SqrtError::Negative}}
                  .error() == SqrtError::Negative);
}  // namespace

SPECTRE_TEST_CASE("Unit.Utilities.Expected", "[Unit][Utilities]") {
  test_value_and_error();
  test_error_messages();
  test_same_value_and_error_type();
  test_non_trivial_value();
}
