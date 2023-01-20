// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <cmath>
#include <cstddef>
#include <limits>
#include <string>
#include <utility>
#include <vector>

#include "Framework/TestHelpers.hpp"
#include "Time/BoundaryHistory.hpp"
#include "Time/Slab.hpp"
#include "Time/Time.hpp"
#include "Time/TimeStepId.hpp"
#include "Utilities/GetOutput.hpp"
#include "Utilities/Gsl.hpp"

namespace {
using BoundaryHistoryType =
    TimeSteppers::BoundaryHistory<std::string, std::vector<int>, double>;

Time make_time(const double t) { return Slab(t, t + 0.5).start(); }

TimeStepId make_time_id(const double t) { return {true, 0, make_time(t)}; }

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

template <bool EvaluatorOwnsCoupling>
size_t check_boundary_state(const BoundaryHistoryType& hist) {
  std::string local_arg;
  std::vector<int> remote_arg;
  double coupling_return = std::numeric_limits<double>::signaling_NaN();

  struct NonCopyableCoupling {
    NonCopyableCoupling(const NonCopyableCoupling&) = delete;
    NonCopyableCoupling& operator=(const NonCopyableCoupling&) = delete;
    double operator()(const std::string& local,
                      const std::vector<int>& remote) const {
      local_arg_ = local;
      remote_arg_ = remote;
      return coupling_return_;
    }

    std::string& local_arg_;
    std::vector<int>& remote_arg_;
    double& coupling_return_;
  };
  const NonCopyableCoupling unowned_coupling{local_arg, remote_arg,
                                             coupling_return};

  // Test correct handling of lvalues and rvalues.
  struct CouplingChooser {
    auto operator()(std::true_type) const {
      // Capture pointers by value to try to ensure that the lambda
      // object has actual member variables that will go out of scope
      // if the lambda is destroyed.  I'm not sure that references
      // could not be resolved at compile time, but these are harder
      // to optimize out.
      return [local_arg_capture = local_arg_, remote_arg_capture = remote_arg_,
              coupling_return_capture = coupling_return_](
                 const std::string& local, const std::vector<int>& remote) {
        *local_arg_capture = local;
        *remote_arg_capture = remote;
        return *coupling_return_capture;
      };
    };
    const auto& operator()(std::false_type) const { return unowned_coupling_; }

    gsl::not_null<std::string*> local_arg_;
    gsl::not_null<std::vector<int>*> remote_arg_;
    gsl::not_null<const double*> coupling_return_;
    const NonCopyableCoupling& unowned_coupling_;
  };
  const CouplingChooser coupling_chooser{&local_arg, &remote_arg,
                                         &coupling_return, unowned_coupling};

  const auto evaluator = hist.evaluator(
      coupling_chooser(std::bool_constant<EvaluatorOwnsCoupling>{}));

  CHECK(hist.local_size() == 4);
  {
    auto it = hist.local_begin();
    for (size_t i = 0; i < hist.local_size(); ++i, ++it) {
      const auto entry_num = static_cast<double>(i) - 1.0;
      CHECK((*it).value() == entry_num);
      CHECK(it->value() == entry_num);
    }
    CHECK(it == hist.local_end());
  }

  CHECK(hist.remote_size() == 6);
  {
    auto it = hist.remote_begin();
    for (size_t i = 0; i < hist.remote_size(); ++i, ++it) {
      const auto entry_num = static_cast<double>(i) - 2.0;
      CHECK((*it).value() == entry_num);
      CHECK(it->value() == entry_num);
    }
    CHECK(it == hist.remote_end());
  }

  size_t coupling_calls = 0;
  coupling_return = 3.5;
  CHECK(3.5 == *evaluator(hist.local_begin(), hist.remote_begin()));
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
  CHECK(6.5 == *evaluator(hist.local_begin() + 3, hist.remote_begin() + 2));
  if (local_arg.empty()) {
    CHECK(remote_arg.empty());
  } else {
    ++coupling_calls;
    CHECK(local_arg == get_output(2));
    CHECK(remote_arg == std::vector<int>{0});
  }

  CHECK(hist.integration_order() == evaluator.integration_order());
  CHECK(hist.local_begin() == evaluator.local_begin());
  CHECK(hist.local_end() == evaluator.local_end());
  CHECK(hist.remote_begin() == evaluator.remote_begin());
  CHECK(hist.remote_end() == evaluator.remote_end());
  CHECK(hist.local_size() == evaluator.local_size());
  CHECK(hist.remote_size() == evaluator.remote_size());

  return coupling_calls;
}

template <bool EvaluatorOwnsCoupling>
void test_boundary_history() {
  BoundaryHistoryType history{3};

  CHECK(history.integration_order() == 3);
  CHECK(std::as_const(history).integration_order() == 3);
  history.integration_order(2);
  CHECK(history.integration_order() == 2);

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

  CHECK(check_boundary_state<EvaluatorOwnsCoupling>(history) == 2);

  std::string expected_output = MakeString{} << "Integration order: 2\n"
                                             << "Local Data:\n"
                                             << "Time: Slab[-1,-0.5]:0/1\n"
                                             << "Data: -1\n"
                                             << "Time: Slab[0,0.5]:0/1\n"
                                             << "Data: 0\n"
                                             << "Time: Slab[1,1.5]:0/1\n"
                                             << "Data: 1\n"
                                             << "Time: Slab[2,2.5]:0/1\n"
                                             << "Data: 2\n"
                                             << "Remote Data:\n"
                                             << "Time: Slab[-2,-1.5]:0/1\n"
                                             << "Data: (-2)\n"
                                             << "Time: Slab[-1,-0.5]:0/1\n"
                                             << "Data: (-1)\n"
                                             << "Time: Slab[0,0.5]:0/1\n"
                                             << "Data: (0)\n"
                                             << "Time: Slab[1,1.5]:0/1\n"
                                             << "Data: (1)\n"
                                             << "Time: Slab[2,2.5]:0/1\n"
                                             << "Data: (2)\n"
                                             << "Time: Slab[3,3.5]:0/1\n"
                                             << "Data: (3)\n";
  CHECK(get_output(history) == expected_output);

  // We check this later, to make sure we don't somehow depend on the
  // original object.
  auto copy = serialize_and_deserialize(history);

  const auto cleaner = history.cleaner();
  cleaner.local_mark_unneeded(history.local_begin());
  CHECK(history.local_size() == 4);
  cleaner.local_mark_unneeded(history.local_begin() + 2);
  CHECK(history.local_size() == 2);

  cleaner.remote_mark_unneeded(history.remote_begin());
  CHECK(history.remote_size() == 6);
  cleaner.remote_mark_unneeded(history.remote_begin() + 2);
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

  cleaner.local_mark_unneeded(history.local_end());
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

  CHECK(history.integration_order() == cleaner.integration_order());
  CHECK(history.local_begin() == cleaner.local_begin());
  CHECK(history.local_end() == cleaner.local_end());
  CHECK(history.remote_begin() == cleaner.remote_begin());
  CHECK(history.remote_end() == cleaner.remote_end());
  CHECK(history.local_size() == cleaner.local_size());
  CHECK(history.remote_size() == cleaner.remote_size());

  CHECK(check_boundary_state<EvaluatorOwnsCoupling>(copy) == 0);
  check_iterator(copy.local_begin() + 1);
  check_iterator(copy.remote_begin() + 2);
  CHECK(copy.integration_order() == 2);
}
}  // namespace

SPECTRE_TEST_CASE("Unit.Time.BoundaryHistory", "[Unit][Time]") {
  test_boundary_history<true>();
  test_boundary_history<false>();
}
