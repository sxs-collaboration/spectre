// Distributed under the MIT License.
// See LICENSE.txt for details.

#include <string>

#include "Time/History.hpp"
#include "Time/Slab.hpp"
#include "Time/Time.hpp"
#include "tests/Unit/TestHelpers.hpp"

namespace {
Time make_time(const double t) noexcept {
  return Slab(t, t + 0.5).start();
}

// Requires `it` to point at 0. in the sequence of times -1., 0., 1., 2.
template <typename Iterator>
void check_iterator(Iterator it) noexcept {
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
}  // namespace

SPECTRE_TEST_CASE("Unit.Time.History", "[Unit][Time]") {
  using HistoryType = TimeSteppers::History<double, std::string>;
  HistoryType history;

  CHECK(history.size() == 0);
  CHECK(history.capacity() == 0);
  CHECK(history.begin() == history.end());
  CHECK(history.begin() == history.cbegin());
  CHECK(history.end() == history.cend());

  history.insert(make_time(0.), 0., get_output(0));
  history.insert_initial(make_time(-1.), -1., get_output(-1));
  {
    auto tmp = get_output(1);
    history.insert(make_time(1.), 1., std::move(tmp));
    // clang-tidy: misc-use-after-move
    CHECK(tmp == get_output(1));  // NOLINT
  }
  history.insert_initial(make_time(-2.), -2., get_output(-2));
  history.insert_initial(make_time(-3.), -3., get_output(-3));

  history.mark_unneeded(history.begin());
  CHECK(history.size() == 5);
  CHECK(history.capacity() == 5);
  history.mark_unneeded(history.begin() + 2);
  CHECK(history.size() == 3);
  CHECK(history.capacity() == 5);

  {
    auto tmp = get_output(2);
    history.insert(make_time(2.), 2., std::move(tmp));
    // clang-tidy: misc-use-after-move
    CHECK(tmp == get_output(-3));  // NOLINT
  }
  CHECK(history.size() == 4);
  CHECK(history.capacity() == 5);

  const auto check_state = [](const HistoryType& hist) noexcept {
    CHECK(hist.size() == 4);
    {
      auto it = hist.begin();
      for (size_t i = 0; i < hist.size(); ++i) {
        CHECK(*it == hist[i]);
        const auto entry_num = static_cast<ssize_t>(i) - 1;
        CHECK((*it).value() == entry_num);
        CHECK(it->value() == entry_num);
        CHECK(it.value() == entry_num);
        CHECK(it.derivative() == get_output(entry_num));
        ++it;
      }
      CHECK(it == hist.end());
    }

    CHECK(hist.front() == hist[0]);
    CHECK(hist.back() == hist[3]);
  };
  check_state(history);

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
    for (size_t i = 0; i < 2; ++i) {
      CHECK(*it == history[i]);
      const auto entry_num = static_cast<ssize_t>(i) + 1;
      CHECK((*it).value() == entry_num);
      CHECK(it->value() == entry_num);
      CHECK(it.value() == entry_num);
      CHECK(it.derivative() == get_output(entry_num));
      ++it;
    }
    CHECK(it == history.end());
  }

  history.mark_unneeded(history.end());
  CHECK(history.size() == 0);

  check_state(copy);
  check_iterator(copy.begin() + 1);
}
