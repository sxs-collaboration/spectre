// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <array>
#include <cstdint>
#include <memory>
#include <optional>
#include <pup.h>
#include <vector>

#include "Framework/TestCreation.hpp"
#include "Framework/TestHelpers.hpp"
#include "Parallel/RegisterDerivedClassesWithCharm.hpp"
#include "Time/TimeSequence.hpp"

namespace {
void test_evenly_spaced() noexcept {
  {
    const TimeSequences::EvenlySpaced<std::uint64_t> constructed(3, 5);
    const auto factory =
        TestHelpers::test_factory_creation<TimeSequence<std::uint64_t>>(
            "EvenlySpaced:\n"
            "  Interval: 3\n"
            "  Offset: 5\n");
    const auto check = [&constructed, &factory](
                           const std::uint64_t arg,
                           const std::uint64_t result) noexcept {
      using Result = std::array<std::optional<std::uint64_t>, 3>;
      const auto expected = result >= 3
                                ? Result{{result - 3, result, result + 3}}
                                : Result{{{}, result, result + 3}};
      CHECK(constructed.times_near(arg) == expected);
      CHECK(factory->times_near(arg) == expected);
      CHECK(serialize_and_deserialize(factory)->times_near(arg) == expected);
    };
    check(0, 2);
    check(1, 2);
    check(2, 2);
    check(3, 2);
    check(4, 5);
    check(5, 5);
    check(6, 5);
    check(7, 8);
    check(8, 8);
    check(9, 8);
  }
  {
    const TimeSequences::EvenlySpaced<double> constructed(3.75, 4.0625);
    const auto factory =
        TestHelpers::test_factory_creation<TimeSequence<double>>(
            "EvenlySpaced:\n"
            "  Interval: 3.75\n"
            "  Offset: 4.0625\n");
    const auto check = [&constructed, &factory](const double arg,
                                                const double result) noexcept {
      const std::array<std::optional<double>, 3> expected{
          {result - 3.75, result, result + 3.75}};
      CHECK(constructed.times_near(arg) == expected);
      CHECK(factory->times_near(arg) == expected);
      CHECK(serialize_and_deserialize(factory)->times_near(arg) == expected);
    };
    check(4.0625, 4.0625);
    check(0.0, 4.0625 - 3.75);
    check(19.0, 4.0625 + 4.0 * 3.75);
    check(-11.0, 4.0625 - 4.0 * 3.75);
  }
}

void test_specified() noexcept {
  {
    using Result = std::array<std::optional<std::uint64_t>, 3>;
    const TimeSequences::Specified<std::uint64_t> constructed(
        {4, 8, 0, 4, 4, 3, 4});
    const auto factory =
        TestHelpers::test_factory_creation<TimeSequence<std::uint64_t>>(
            "Specified:\n"
            "  Values: [4, 8, 0, 4, 4, 3, 4]\n");
    const auto check = [&constructed, &factory](
                           const std::uint64_t arg,
                           const Result& expected) noexcept {
      CHECK(constructed.times_near(arg) == expected);
      CHECK(factory->times_near(arg) == expected);
      CHECK(serialize_and_deserialize(factory)->times_near(arg) == expected);
    };
    check(0, {{{}, 0, 3}});
    check(1, {{{}, 0, 3}});
    check(2, {{0, 3, 4}});
    check(3, {{0, 3, 4}});
    check(4, {{3, 4, 8}});
    check(5, {{3, 4, 8}});
    // 6 is a halfway point, and we choose not to specify which side
    // it prefers.
    check(7, {{4, 8, {}}});
    check(8, {{4, 8, {}}});
    check(9, {{4, 8, {}}});

    const TimeSequences::Specified<std::uint64_t> zero_elements({});
    CHECK(zero_elements.times_near(5) == Result{});
    const TimeSequences::Specified<std::uint64_t> one_element({3});
    CHECK(one_element.times_near(5) == Result{{{}, 3, {}}});
    const TimeSequences::Specified<std::uint64_t> two_elements({3, 6});
    CHECK(two_elements.times_near(2) == Result{{{}, 3, 6}});
    CHECK(two_elements.times_near(3) == Result{{{}, 3, 6}});
    CHECK(two_elements.times_near(4) == Result{{{}, 3, 6}});
    CHECK(two_elements.times_near(5) == Result{{3, 6, {}}});
    CHECK(two_elements.times_near(6) == Result{{3, 6, {}}});
    CHECK(two_elements.times_near(7) == Result{{3, 6, {}}});
  }
  {
    using Result = std::array<std::optional<double>, 3>;
    const TimeSequences::Specified<double> constructed(
        {-5.1, 7.2, -2.3, 0.0, -5.1, -5.1, 3.4, 4.5});
    const auto factory =
        TestHelpers::test_factory_creation<TimeSequence<double>>(
            "Specified:\n"
            "  Values: [-5.1, 7.2, -2.3, 0.0, -5.1, -5.1, 3.4, 4.5]\n");
    const auto check = [&constructed, &factory](
                           const double arg, const Result& expected) noexcept {
      CHECK(constructed.times_near(arg) == expected);
      CHECK(factory->times_near(arg) == expected);
      CHECK(serialize_and_deserialize(factory)->times_near(arg) == expected);
    };
    check(-8.9, {{{}, -5.1, -2.3}});
    check(-5.1, {{{}, -5.1, -2.3}});
    check(-4.7, {{{}, -5.1, -2.3}});
    check(-2.8, {{-5.1, -2.3, 0.0}});
    check(-2.3, {{-5.1, -2.3, 0.0}});
    check(4.5, {{3.4, 4.5, 7.2}});
    check(4.6, {{3.4, 4.5, 7.2}});
    check(7.0, {{4.5, 7.2, {}}});
    check(7.2, {{4.5, 7.2, {}}});
    check(8.1, {{4.5, 7.2, {}}});
  }
}
}  // namespace

SPECTRE_TEST_CASE("Unit.Time.TimeSequence", "[Unit][Time]") {
  Parallel::register_derived_classes_with_charm<TimeSequence<std::uint64_t>>();
  Parallel::register_derived_classes_with_charm<TimeSequence<double>>();

  test_evenly_spaced();
  test_specified();
}
