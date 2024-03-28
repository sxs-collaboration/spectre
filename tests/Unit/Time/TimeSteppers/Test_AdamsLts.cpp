// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Utilities/StdHelpers.hpp"

#include "Framework/TestingFramework.hpp"

#include <cstddef>
#include <cstdint>
#include <utility>

#include "DataStructures/DataVector.hpp"
#include "Time/BoundaryHistory.hpp"
#include "Time/Slab.hpp"
#include "Time/TimeStepId.hpp"
#include "Time/TimeSteppers/AdamsLts.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/MakeWithValue.hpp"

namespace {
namespace adams_lts = TimeSteppers::adams_lts;

void test_lts_coefficients_struct() {
  const Slab slab(0.0, 1.0);
  const TimeStepId id1(true, 0, slab.start());
  const TimeStepId id2 = id1.next_substep(slab.duration() / 2, 1.0);
  const TimeStepId id3 = id2.next_step(slab.duration() / 2);

  const adams_lts::LtsCoefficients coefs1{
      {id1, id1, 2.0}, {id1, id2, 3.0}, {id1, id3, 4.0}};
  const adams_lts::LtsCoefficients coefs2{
      {id1, id1, 5.0}, {id1, id3, 7.0}, {id2, id1, 9.0}};

  const adams_lts::LtsCoefficients sum{
      {id1, id1, 7.0}, {id1, id2, 3.0}, {id1, id3, 11.0}, {id2, id1, 9.0}};
  const adams_lts::LtsCoefficients difference{
      {id1, id1, -3.0}, {id1, id2, 3.0}, {id1, id3, -3.0}, {id2, id1, -9.0}};

  {
    auto s = coefs1;
    CHECK(&(s += coefs2) == &s);
    CHECK(s == sum);
  }
  {
    auto d = coefs1;
    CHECK(&(d -= coefs2) == &d);
    CHECK(d == difference);
  }

  CHECK(coefs1 + coefs2 == sum);
  CHECK(coefs1 - coefs2 == difference);
  CHECK(adams_lts::LtsCoefficients(coefs1) + coefs2 == sum);
  CHECK(adams_lts::LtsCoefficients(coefs1) - coefs2 == difference);
  CHECK(coefs1 + adams_lts::LtsCoefficients(coefs2) == sum);
  CHECK(coefs1 - adams_lts::LtsCoefficients(coefs2) == difference);
  CHECK(adams_lts::LtsCoefficients(coefs1) +
            adams_lts::LtsCoefficients(coefs2) ==
        sum);
  CHECK(adams_lts::LtsCoefficients(coefs1) -
            adams_lts::LtsCoefficients(coefs2) ==
        difference);

  adams_lts::LtsCoefficients allocated_coefs{};
  for (size_t i = 0; i < adams_lts::lts_coefficients_static_size + 1; ++i) {
    allocated_coefs.emplace_back(
        id1,
        TimeStepId(
            true, static_cast<int64_t>(i),
            Slab(static_cast<double>(i), static_cast<double>(i + 1)).start()),
        i);
  }

  {
    auto a = allocated_coefs;
    const auto* const data = a.data();
    const auto res = std::move(a) + allocated_coefs;
    CHECK(res.data() == data);
    CHECK(res == allocated_coefs + allocated_coefs);
  }
  {
    auto a = allocated_coefs;
    const auto* const data = a.data();
    const auto res = std::move(a) - allocated_coefs;
    CHECK(res.data() == data);
    CHECK(res == allocated_coefs - allocated_coefs);
  }
  {
    auto a = allocated_coefs;
    const auto* const data = a.data();
    const auto res = allocated_coefs + std::move(a);
    CHECK(res.data() == data);
    CHECK(res == allocated_coefs + allocated_coefs);
  }
  {
    auto a = allocated_coefs;
    const auto* const data = a.data();
    const auto res = allocated_coefs - std::move(a);
    CHECK(res.data() == data);
    CHECK(res == allocated_coefs - allocated_coefs);
  }
  {
    auto a = allocated_coefs;
    auto b = allocated_coefs;
    const auto* const a_data = a.data();
    const auto* const b_data = a.data();
    const auto res = std::move(a) + std::move(b);
    CHECK((res.data() == a_data or res.data() == b_data));
    CHECK(res == allocated_coefs + allocated_coefs);
  }
  {
    auto a = allocated_coefs;
    auto b = allocated_coefs;
    const auto* const a_data = a.data();
    const auto* const b_data = a.data();
    const auto res = std::move(a) - std::move(b);
    CHECK((res.data() == a_data or res.data() == b_data));
    CHECK(res == allocated_coefs - allocated_coefs);
  }
}

template <typename T>
void test_apply_coefficients(const T& used_for_size) {
  const Slab slab(1.2, 3.4);

  TimeSteppers::BoundaryHistory<double, double, T> history{};
  history.local().insert(TimeStepId(true, 0, slab.start()), 1, 1.0);
  history.local().insert(TimeStepId(true, 0, slab.end()), 1, 10.0);
  history.remote().insert(TimeStepId(true, 0, slab.start()), 1, 100.0);
  history.remote().insert(TimeStepId(true, 0, slab.end()), 1, 10000.0);

  const adams_lts::LtsCoefficients coefficients{
      {history.local()[0], history.remote()[0], 1.0},
      {history.local()[0], history.remote()[1], 2.0},
      {history.local()[1], history.remote()[1], 3.0}};

  const auto coupling = [&used_for_size](const double local,
                                         const double remote) {
    return make_with_value<T>(used_for_size, local * remote);
  };

  auto result = make_with_value<T>(used_for_size, 2.0);
  adams_lts::apply_coefficients(make_not_null(&result), coefficients,
                                history.evaluator(coupling));

  CHECK(result == make_with_value<T>(used_for_size, 320102.0));
}

SPECTRE_TEST_CASE("Unit.Time.TimeSteppers.AdamsLts", "[Unit][Time]") {
  test_lts_coefficients_struct();
  test_apply_coefficients(0.0);
  test_apply_coefficients(DataVector(5, 0.0));
}
}  // namespace
