// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <random>
#include <set>
#include <vector>

#include "Framework/TestHelpers.hpp"
#include "Utilities/ErrorHandling/Assert.hpp"
#include "Utilities/FractionUtilities.hpp"
#include "Utilities/Rational.hpp"

namespace {
template <typename Source>
std::vector<typename Source::value_type> collect(
    Source source, const size_t limit = std::numeric_limits<size_t>::max()) {
  std::vector<typename Source::value_type> result;
  for (size_t count = 0; count < limit and source; ++count, ++source) {
    result.push_back(*source);
  }
  return result;
}
}  // namespace

SPECTRE_TEST_CASE("Unit.Utilities.FractionUtilities.ContinuedFraction",
                  "[Utilities][Unit]") {
  CHECK((std::vector<int32_t>{1, 2, 4, 2}) ==
        collect(ContinuedFraction<Rational>(Rational(29, 20))));
  CHECK((std::vector<int64_t>{0, 8}) ==
        collect(ContinuedFraction<double>(0.125)));
  CHECK((std::vector<int64_t>{-1, 1, 7}) ==
        collect(ContinuedFraction<double>(-0.125)));
  CHECK(std::vector<int64_t>(20, 1) ==
        collect(ContinuedFraction<double>(0.5 * (1. + sqrt(5.))), 20));
  CHECK((std::vector<int32_t>{0}) ==
        collect(ContinuedFraction<Rational>(Rational(0))));
  CHECK((std::vector<int64_t>{0}) == collect(ContinuedFraction<double>(0.)));

  // Check that the iterator terminates because of precision loss.
  CHECK(collect(ContinuedFraction<double>(0.5 * (1. + sqrt(5.))), 100).size() <
        100);

  // Check that the iterator doesn't terminate prematurely
  // because of precision loss.
  const int64_t two_to_the_fourty = 1099511627776;
  CHECK(std::vector<int64_t>{0, two_to_the_fourty} ==
        collect(ContinuedFraction<double>(1. / two_to_the_fourty)));
  CHECK(std::vector<int64_t>{1, two_to_the_fourty} ==
        collect(ContinuedFraction<double>(1 + 1. / two_to_the_fourty)));
  CHECK(std::vector<int64_t>{0, two_to_the_fourty, 2} ==
        collect(ContinuedFraction<double>(1. / (two_to_the_fourty + 0.5))));

  MAKE_GENERATOR(gen);
  {
    std::uniform_real_distribution<> dist(-10., 10.);
    const double value = dist(gen);
    // Set the scale because the fractional part of a negative number
    // (defined as `x - floor(x)`) can be larger than the number.
    auto approx_value =
        approx.scale(std::max(value, std::abs(std::floor(value))))(value);
    ContinuedFractionSummer<Rational> summer;
    bool should_be_smaller = true;
    std::vector<int64_t> terms{};  // Only for output
    std::vector<double> convergents{};  // Only for output
    for (ContinuedFraction<double> source(value); source; ++source) {
      summer.insert(*source);
      terms.push_back(*source);
      convergents.push_back(summer.value().value());
      CAPTURE(terms);
      CAPTURE(convergents);

      // Convergents to a continued fraction always alternate between
      // over- and underestimates.
      if (convergents.back() != approx_value) {
        if (should_be_smaller) {
          CHECK(convergents.back() <= value);
        } else {
          CHECK(convergents.back() >= value);
        }
      }
      should_be_smaller = !should_be_smaller;
    }
    CAPTURE(terms);
    CAPTURE(convergents);
    CHECK(convergents.back() == approx_value);
  }
}

SPECTRE_TEST_CASE("Unit.Utilities.FractionUtilities.ContinuedFractionSummer",
                  "[Utilities][Unit]") {
  const auto check = [](const int num, const int denom) {
    const Rational value(num, denom);
    ContinuedFractionSummer<Rational> summer;
    for (ContinuedFraction<Rational> source(value); source; ++source) {
      summer.insert(*source);
    }
    CHECK(summer.value() == value);
  };

  check(0, 1);
  check(1, 1);
  check(2, 1);
  check(1, 2);
  check(2, 3);
  check(29, 20);

  // Check overflow is handled correctly
  {
    ContinuedFractionSummer<Rational> summer;
    summer.insert(1000);
    const auto first_value = summer.value();
    summer.insert(std::numeric_limits<std::int32_t>::max() / 10);
    CHECK(summer.value() == first_value);
    summer.insert(1);
    CHECK(summer.value() == first_value);
  }

  // Check a case that returns a semiconvergent
  {
    ContinuedFractionSummer<Rational> summer;
    summer.insert(0);
    summer.insert(2);
    summer.insert(std::numeric_limits<std::int32_t>::max() / 2 + 10);
    // Remember that max() is odd, so this isn't 1/2.
    CHECK(summer.value() ==
          Rational(std::numeric_limits<std::int32_t>::max() / 2,
                   std::numeric_limits<std::int32_t>::max()));
  }
}

SPECTRE_TEST_CASE(
    "Unit.Utilities.FractionUtilities.simplest_fraction_in_interval",
    "[Utilities][Unit]") {
  const int denom_max = 20;

  std::set<Rational> fractions;
  for (int d = 1; d <= denom_max; ++d) {
    for (int n = 0; n <= d; ++n) {
      fractions.emplace(n, d);
    }
  }

  for (auto end1 = fractions.begin(); end1 != fractions.end(); ++end1) {
    // Worse than any considered value so will be immediately replaced.
    Rational simplest(1, denom_max + 1);
    for (auto end2 = end1; end2 != fractions.end(); ++end2) {
      if (*end1 == 0 and *end2 == 1) {
        // Correct answer not clear for the entire interval
        continue;
      }
      ASSERT(end2->denominator() != simplest.denominator(),
             "Answer is not unique");
      if (end2->denominator() < simplest.denominator()) {
        simplest = *end2;
      }
      CHECK(simplest_fraction_in_interval<Rational>(*end1, *end2) == simplest);
      CHECK(simplest_fraction_in_interval<Rational>(*end2, *end1) == simplest);
    }
  }

  // Quick check of non-exact input
  CHECK(simplest_fraction_in_interval<Rational>(0.6, 0.9) == Rational(2, 3));
}
