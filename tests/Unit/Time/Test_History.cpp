// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <boost/iterator/transform_iterator.hpp>
#include <cmath>
#include <cstddef>
#include <deque>
#include <string>
#include <vector>

#include "Framework/TestHelpers.hpp"
#include "Time/BoundaryHistory.hpp"
#include "Time/History.hpp"
#include "Time/Slab.hpp"
#include "Time/Time.hpp"
#include "Time/TimeStepId.hpp"
#include "Utilities/GetOutput.hpp"
#include "Utilities/Gsl.hpp"

namespace {

Time make_time(const double t) noexcept { return Slab(t, t + 0.5).start(); }

TimeStepId make_time_id(const double t) noexcept {
  constexpr size_t substeps = 2;
  return {true, 0, make_time(substeps * std::floor(t / substeps)),
          static_cast<size_t>(std::fmod(t, substeps) + substeps) % substeps,
          make_time(t)};
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

using HistoryType = TimeSteppers::History<double, std::string>;

void check_history_state(const HistoryType& hist) noexcept {
  CHECK(hist.size() == 4);
  {
    auto it = hist.begin();
    for (size_t i = 0; i < hist.size(); ++i, ++it) {
      const auto entry_num = static_cast<double>(i) - 1.0;
      CHECK(*it == hist[i]);
      CHECK(it.time_step_id() == make_time_id(entry_num));
      CHECK((*it).value() == entry_num);
      CHECK(it->value() == entry_num);
      CHECK(it.value() == entry_num);
      CHECK(it.derivative() == get_output(entry_num));
    }
    CHECK(it == hist.end());
  }

  CHECK(hist.front() == hist[0]);
  CHECK(hist.back() == hist[3]);
}
}  // namespace

SPECTRE_TEST_CASE("Unit.Time.History", "[Unit][Time]") {
  HistoryType history;

  CHECK(history.size() == 0);
  CHECK(history.capacity() == 0);
  CHECK(history.begin() == history.end());
  CHECK(history.begin() == history.cbegin());
  CHECK(history.end() == history.cend());

  history.insert(make_time_id(0.), 0., get_output(0));
  history.insert_initial(make_time_id(-1.), -1., get_output(-1));
  history.insert(make_time_id(1.), 1., get_output(1));
  history.insert_initial(make_time_id(-2.), -2., get_output(-2));
  history.insert_initial(make_time_id(-3.), -3., get_output(-3));

  history.mark_unneeded(history.begin());
  CHECK(history.size() == 5);
  CHECK(history.capacity() == 5);
  history.mark_unneeded(history.begin() + 2);
  CHECK(history.size() == 3);
  CHECK(history.capacity() == 5);

  history.insert(make_time_id(2.), 2., get_output(2));
  CHECK(history.size() == 4);
  CHECK(history.capacity() == 5);

  check_history_state(history);

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
      CHECK(it.value() == entry_num);
      CHECK(it.derivative() == get_output(entry_num));
    }
    CHECK(it == history.end());
  }

  history.mark_unneeded(history.end());
  CHECK(history.size() == 0);

  check_history_state(copy);
  check_iterator(copy.begin() + 1);
}

namespace {
using BoundaryHistoryType =
    TimeSteppers::BoundaryHistory<std::string, std::vector<int>, double>;

// Must take a non-const arg for coupling caching
size_t check_boundary_state(
    const gsl::not_null<BoundaryHistoryType*> hist) noexcept {
  CHECK(hist->local_size() == 4);
  {
    auto it = hist->local_begin();
    for (size_t i = 0; i < hist->local_size(); ++i, ++it) {
      const auto entry_num = static_cast<double>(i) - 1.0;
      CHECK((*it).value() == entry_num);
      CHECK(it->value() == entry_num);
    }
    CHECK(it == hist->local_end());
  }

  CHECK(hist->remote_size() == 6);
  {
    auto it = hist->remote_begin();
    for (size_t i = 0; i < hist->remote_size(); ++i, ++it) {
      const auto entry_num = static_cast<double>(i) - 2.0;
      CHECK((*it).value() == entry_num);
      CHECK(it->value() == entry_num);
    }
    CHECK(it == hist->remote_end());
  }

  std::string local_arg;
  std::vector<int> remote_arg;
  double coupling_return;
  const auto coupling = [&local_arg, &remote_arg, &coupling_return ](
      const std::string& local, const std::vector<int>& remote) noexcept {
    local_arg = local;
    remote_arg = remote;
    return coupling_return;
  };

  size_t coupling_calls = 0;
  coupling_return = 3.5;
  CHECK(3.5 ==
        hist->coupling(coupling, hist->local_begin(), hist->remote_begin()));
  if (local_arg.empty()) {
    CHECK(remote_arg.empty());
  } else {
    ++coupling_calls;
    CHECK(local_arg == get_output(-1));
    CHECK(remote_arg == std::vector<int>{-2});
  }
  local_arg.clear();
  remote_arg.clear();

  coupling_return = 6.5;
  CHECK(6.5 == hist->coupling(coupling, hist->local_begin() + 3,
                              hist->remote_begin() + 2));
  if (local_arg.empty()) {
    CHECK(remote_arg.empty());
  } else {
    ++coupling_calls;
    CHECK(local_arg == get_output(2));
    CHECK(remote_arg == std::vector<int>{0});
  }

  return coupling_calls;
}
}  // namespace

SPECTRE_TEST_CASE("Unit.Time.BoundaryHistory", "[Unit][Time]") {
  BoundaryHistoryType history;

  CHECK(history.local_size() == 0);
  CHECK(history.local_begin() == history.local_end());
  CHECK(history.remote_size() == 0);
  CHECK(history.remote_begin() == history.remote_end());

  history.local_insert(make_time_id(0.), get_output(0));
  history.local_insert(make_time_id(1.), get_output(1));
  history.local_insert_initial(make_time_id(-1.), get_output(-1));
  history.local_insert(make_time_id(2.), get_output(2));

  {
    INFO("Test `local_data` member function");
    CHECK(history.local_data(make_time_id(0.)) == get_output(0));
    CHECK(history.local_data(make_time_id(-1.)) == get_output(-1));
    CHECK(history.local_data(make_time_id(2.)) == get_output(2));
  }

  history.remote_insert(make_time_id(1.), std::vector<int>{1});
  history.remote_insert_initial(make_time_id(0.), std::vector<int>{0});
  history.remote_insert_initial(make_time_id(-1.), std::vector<int>{-1});
  history.remote_insert(make_time_id(2.), std::vector<int>{2});
  history.remote_insert_initial(make_time_id(-2.), std::vector<int>{-2});
  history.remote_insert(make_time_id(3.), std::vector<int>{3});

  CHECK(check_boundary_state(&history) == 2);

  // We check this later, to make sure we don't somehow depend on the
  // original object.
  auto copy = serialize_and_deserialize(history);

  history.local_mark_unneeded(history.local_begin());
  CHECK(history.local_size() == 4);
  history.local_mark_unneeded(history.local_begin() + 2);
  CHECK(history.local_size() == 2);

  history.remote_mark_unneeded(history.remote_begin());
  CHECK(history.remote_size() == 6);
  history.remote_mark_unneeded(history.remote_begin() + 2);
  CHECK(history.remote_size() == 4);

  {
    auto it = history.local_begin();
    for (size_t i = 0; i < history.local_size(); ++i, ++it) {
      const auto entry_num = static_cast<double>(i) + 1.0;
      CHECK((*it).value() == entry_num);
      CHECK(it->value() == entry_num);
    }
    CHECK(it == history.local_end());
  }

  history.local_mark_unneeded(history.local_end());
  CHECK(history.local_size() == 0);

  CHECK(history.remote_size() == 4);
  {
    auto it = history.remote_begin();
    for (size_t i = 0; i < history.remote_size(); ++i, ++it) {
      const auto entry_num = static_cast<double>(i);
      CHECK((*it).value() == entry_num);
      CHECK(it->value() == entry_num);
    }
    CHECK(it == history.remote_end());
  }

  CHECK(check_boundary_state(&copy) == 0);
  check_iterator(copy.local_begin() + 1);
  check_iterator(copy.remote_begin() + 2);
}
