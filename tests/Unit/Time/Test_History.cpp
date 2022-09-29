// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <cmath>
#include <cstddef>
#include <utility>

#include "Framework/TestHelpers.hpp"
#include "Time/History.hpp"
#include "Time/Slab.hpp"
#include "Time/Time.hpp"
#include "Time/TimeStepId.hpp"
#include "Utilities/ErrorHandling/Assert.hpp"

namespace {

Time make_time(const double t) { return Slab(t, t + 0.5).start(); }

TimeStepId make_time_id(const double t) {
  constexpr size_t substeps = 2;
  return {true, 0, make_time(substeps * std::floor(t / substeps)),
          static_cast<size_t>(std::fmod(t, substeps) + substeps) % substeps,
          make_time(t)};
}

// Requires `it` to point at 0. in the sequence of times -1., 0., 1., 2.
template <typename Iterator>
void check_iterator(Iterator it) {
  CHECK(*it == make_time(0.));
  CHECK(it->value() == 0.);
  CHECK(it[0] == make_time(0.));
  CHECK(it[1] == make_time(1.));
  CHECK(it[-1] == make_time(-1.));

  CHECK(*++it == make_time(1.));
  CHECK(*it == make_time(1.));
  CHECK(*--it == make_time(0.));
  CHECK(*it == make_time(0.));
  CHECK(*it++ == make_time(0.));
  CHECK(*it == make_time(1.));
  CHECK(*it-- == make_time(1.));
  CHECK(*it == make_time(0.));

  CHECK(*(it += 0) == make_time(0.));
  CHECK(*it == make_time(0.));
  CHECK(*(it += 2) == make_time(2.));
  CHECK(*it == make_time(2.));
  CHECK(*(it += -2) == make_time(0.));
  CHECK(*it == make_time(0.));

  CHECK(*(it -= 0) == make_time(0.));
  CHECK(*it == make_time(0.));
  CHECK(*(it -= -2) == make_time(2.));
  CHECK(*it == make_time(2.));
  CHECK(*(it -= 2) == make_time(0.));
  CHECK(*it == make_time(0.));

  auto next_it = it;
  ++next_it;
  CHECK(it + 0 == it);
  CHECK(it + 1 == next_it);
  CHECK(it - (-1) == next_it);
  CHECK(next_it - 0 == next_it);
  CHECK(next_it - 1 == it);
  CHECK(next_it + (-1) == it);
  CHECK(0 + it == it);
  CHECK(1 + it == next_it);
  CHECK((-1) + next_it == it);

  CHECK(it - it == 0);
  CHECK(next_it - it == 1);
  CHECK(it - next_it == -1);

  check_cmp(it, it + 1);
}

using HistoryType = TimeSteppers::History<double>;

void check_history_state(const HistoryType& hist) {
  CHECK(hist.size() == 4);
  {
    auto it = hist.begin();
    for (size_t i = 0; i < hist.size(); ++i, ++it) {
      const auto entry_num = static_cast<double>(i) - 1.0;
      CHECK(*it == hist[i]);
      CHECK(it.time_step_id() == make_time_id(entry_num));
      CHECK((*it).value() == entry_num);
      CHECK(it->value() == entry_num);
      CHECK(*it.derivative() == entry_num + 0.5);
    }
    CHECK(it == hist.end());
  }

  CHECK(hist.front() == hist[0]);
  CHECK(hist.back() == hist[3]);
}
}  // namespace

SPECTRE_TEST_CASE("Unit.Time.History", "[Unit][Time]") {
  HistoryType history{3};

  CHECK(history.integration_order() == 3);
  CHECK(std::as_const(history).integration_order() == 3);
  history.integration_order(2);
  CHECK(history.integration_order() == 2);

  CHECK(history.size() == 0);
  CHECK(history.capacity() == 0);
  CHECK(history.begin() == history.end());
  CHECK(history.begin() == history.cbegin());
  CHECK(history.end() == history.cend());

  history.insert(make_time_id(0.), 0.5);
  history.insert_initial(make_time_id(-1.), -0.5);
  history.insert(make_time_id(1.), 1.5);
  history.insert_initial(make_time_id(-2.), -1.5);
  history.insert_initial(make_time_id(-3.), -2.5);

  history.mark_unneeded(history.begin());
  CHECK(history.size() == 5);
  CHECK(history.capacity() == 5);
  history.mark_unneeded(history.begin() + 2);
  CHECK(history.size() == 3);
  CHECK(history.capacity() == 5);

  history.insert(make_time_id(2.), 2.5);
  CHECK(history.size() == 4);
  CHECK(history.capacity() == 5);

  check_history_state(history);

  // Test DerivIterator
  {
    CHECK(static_cast<size_t>(history.derivatives_end() -
                              history.derivatives_begin()) == history.size());
    auto it = history.begin();
    auto deriv_it = history.derivatives_begin();
    ASSERT(history.size() > 1,
           "Test logic error: Not all conditions will be reached.");
    for (size_t i = 0; i < history.size(); ++i, ++it, ++deriv_it) {
      CHECK(deriv_it == deriv_it);
      CHECK_FALSE(deriv_it != deriv_it);
      CHECK_FALSE(deriv_it < deriv_it);
      CHECK_FALSE(deriv_it > deriv_it);
      CHECK(deriv_it <= deriv_it);
      CHECK(deriv_it >= deriv_it);

      CHECK(deriv_it.time_step_id() == it.time_step_id());
      CHECK(*deriv_it == *it.derivative());
      CHECK(deriv_it.operator->() == &*deriv_it);
      CHECK(&deriv_it[0] == &*deriv_it);
      CHECK(&history.derivatives_begin()[static_cast<int>(i)] == &*deriv_it);
      if (i > 0) {
        auto copy = deriv_it;
        auto previous = deriv_it;
        CHECK(&*--previous == &deriv_it[-1]);
        CHECK(&*(copy--) == &*deriv_it);
        CHECK(copy == previous);
        if (i > 1) {
          auto copy2 = deriv_it;
          CHECK(&(copy2 -= 2) == &copy2);
          CHECK(copy2 == --copy);
          CHECK(deriv_it - 2 == copy2);
        }
        CHECK_FALSE(previous == deriv_it);
        CHECK(previous != deriv_it);
        CHECK(previous < deriv_it);
        CHECK_FALSE(previous > deriv_it);
        CHECK(previous <= deriv_it);
        CHECK_FALSE(previous >= deriv_it);
      }
      if (i < history.size() - 1) {
        auto copy = deriv_it;
        auto next = deriv_it;
        CHECK(&*++next == &deriv_it[1]);
        CHECK(&*(copy++) == &*deriv_it);
        CHECK(copy == next);
        if (i < history.size() - 2) {
          auto copy2 = deriv_it;
          CHECK(&(copy2 += 2) == &copy2);
          CHECK(copy2 == ++copy);
          CHECK(deriv_it + 2 == copy2);
          CHECK(2 + deriv_it == copy2);
        }
        CHECK_FALSE(next == deriv_it);
        CHECK(next != deriv_it);
        CHECK_FALSE(next < deriv_it);
        CHECK(next > deriv_it);
        CHECK_FALSE(next <= deriv_it);
        CHECK(next >= deriv_it);
      }
    }
    CHECK(deriv_it == history.derivatives_end());
  }

  // We check this later, to make sure we don't somehow depend on the
  // original object.
  const auto copy = serialize_and_deserialize(history);

  history.mark_unneeded(history.begin() + 2);
  CHECK(history.size() == 2);
  history.shrink_to_fit();
  CHECK(history.size() == 2);
  CHECK(history.capacity() == 2);

  {
    auto it = history.begin();
    for (size_t i = 0; i < 2; ++i, ++it) {
      const auto entry_num = static_cast<double>(i) + 1.0;
      CHECK(*it == history[i]);
      CHECK(it.time_step_id() == make_time_id(entry_num));
      CHECK((*it).value() == entry_num);
      CHECK(it->value() == entry_num);
      CHECK(*it.derivative() == entry_num + 0.5);
    }
    CHECK(it == history.end());
  }

  history.mark_unneeded(history.end());
  CHECK(history.size() == 0);

  check_history_state(copy);
  check_iterator(copy.begin() + 1);
  CHECK(copy.integration_order() == 2);
}
