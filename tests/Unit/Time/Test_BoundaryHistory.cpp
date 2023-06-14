// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <cstddef>
#include <limits>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

#include "Framework/TestHelpers.hpp"
#include "Time/BoundaryHistory.hpp"
#include "Time/Slab.hpp"
#include "Time/TimeStepId.hpp"
#include "Utilities/GetOutput.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/StdHelpers.hpp"

namespace {
using BoundaryHistoryType =
    TimeSteppers::BoundaryHistory<std::string, std::vector<int>, double>;

TimeStepId make_time_id(const double t) {
  return {true, 0, Slab(t, t + 0.5).start()};
}

// Check the expected state at one particular point in the test.
template <bool IsConcrete, typename T>
void check_local(const T& local_times) {
  CHECK(local_times.size() == 4);

  CHECK(local_times[0] == make_time_id(-1.));
  CHECK(local_times[1] == make_time_id(0.));
  CHECK(local_times[2] == make_time_id(1.));
  CHECK(local_times[3] == make_time_id(2.));

  if constexpr (IsConcrete) {
    CHECK(local_times.data(make_time_id(-1.)) == get_output(-1));
    CHECK(local_times.data(make_time_id(0.)) == get_output(0));
    CHECK(local_times.data(make_time_id(1.)) == get_output(1));
    CHECK(local_times.data(make_time_id(2.)) == get_output(2));
    CHECK(local_times.data(0) == get_output(-1));
    CHECK(local_times.data(1) == get_output(0));
    CHECK(local_times.data(2) == get_output(1));
    CHECK(local_times.data(3) == get_output(2));
  }

  CHECK(local_times.integration_order(make_time_id(-1.)) == 9);
  CHECK(local_times.integration_order(make_time_id(0.)) == 10);
  CHECK(local_times.integration_order(make_time_id(1.)) == 11);
  CHECK(local_times.integration_order(make_time_id(2.)) == 12);
  CHECK(local_times.integration_order(0) == 9);
  CHECK(local_times.integration_order(1) == 10);
  CHECK(local_times.integration_order(2) == 11);
  CHECK(local_times.integration_order(3) == 12);
}

// Check the expected state at one particular point in the test.
template <bool IsConcrete, typename T>
void check_remote(const T& remote_times) {
  CHECK(remote_times[0] == make_time_id(-2.));
  CHECK(remote_times[1] == make_time_id(-1.));
  CHECK(remote_times[2] == make_time_id(0.));
  CHECK(remote_times[3] == make_time_id(1.));
  CHECK(remote_times[4] == make_time_id(2.));
  CHECK(remote_times[5] == make_time_id(3.));

  if constexpr (IsConcrete) {
    CHECK(remote_times.data(make_time_id(-2.)) == std::vector{-2});
    CHECK(remote_times.data(make_time_id(-1.)) == std::vector{-1});
    CHECK(remote_times.data(make_time_id(0.)) == std::vector{0});
    CHECK(remote_times.data(make_time_id(1.)) == std::vector{1});
    CHECK(remote_times.data(make_time_id(2.)) == std::vector{2});
    CHECK(remote_times.data(make_time_id(3.)) == std::vector{3});
    CHECK(remote_times.data(0) == std::vector{-2});
    CHECK(remote_times.data(1) == std::vector{-1});
    CHECK(remote_times.data(2) == std::vector{0});
    CHECK(remote_times.data(3) == std::vector{1});
    CHECK(remote_times.data(4) == std::vector{2});
    CHECK(remote_times.data(5) == std::vector{3});
  }

  CHECK(remote_times.integration_order(make_time_id(-2.)) == 18);
  CHECK(remote_times.integration_order(make_time_id(-1.)) == 19);
  CHECK(remote_times.integration_order(make_time_id(0.)) == 20);
  CHECK(remote_times.integration_order(make_time_id(1.)) == 21);
  CHECK(remote_times.integration_order(make_time_id(2.)) == 22);
  CHECK(remote_times.integration_order(make_time_id(3.)) == 23);
  CHECK(remote_times.integration_order(0) == 18);
  CHECK(remote_times.integration_order(1) == 19);
  CHECK(remote_times.integration_order(2) == 20);
  CHECK(remote_times.integration_order(3) == 21);
  CHECK(remote_times.integration_order(4) == 22);
  CHECK(remote_times.integration_order(5) == 23);
}

template <bool EvaluatorOwnsCoupling>
size_t check_coupling_evaluation(const BoundaryHistoryType& hist) {
  std::string local_arg;
  std::vector<int> remote_arg;
  double coupling_return = std::numeric_limits<double>::signaling_NaN();

  struct NonCopyableCoupling {
    NonCopyableCoupling(const NonCopyableCoupling&) = delete;
    NonCopyableCoupling& operator=(const NonCopyableCoupling&) = delete;
    ~NonCopyableCoupling() = default;
    double operator()(const std::string& local,
                      const std::vector<int>& remote) const {
      *local_arg_ = local;
      *remote_arg_ = remote;
      return *coupling_return_;
    }

    gsl::not_null<std::string*> local_arg_;
    gsl::not_null<std::vector<int>*> remote_arg_;
    gsl::not_null<const double*> coupling_return_;
  };
  const NonCopyableCoupling unowned_coupling{&local_arg, &remote_arg,
                                             &coupling_return};

  // Test correct handling of lvalues and rvalues.
  struct CouplingChooser {  // NOLINT(cppcoreguidelines-pro-type-member-init)
    // clang-tidy complaint is bogus.  It's not possible to fail to
    // initialize a gsl::not_null.
    auto operator()(std::true_type /*unused*/) const {
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
    const auto& operator()(std::false_type /*unused*/) const {
      return *unowned_coupling_;
    }

    gsl::not_null<std::string*> local_arg_;
    gsl::not_null<std::vector<int>*> remote_arg_;
    gsl::not_null<const double*> coupling_return_;
    gsl::not_null<const NonCopyableCoupling*> unowned_coupling_;
  };
  const CouplingChooser coupling_chooser{&local_arg, &remote_arg,
                                         &coupling_return, &unowned_coupling};

  const auto evaluator = hist.evaluator(
      coupling_chooser(std::bool_constant<EvaluatorOwnsCoupling>{}));

  size_t coupling_calls = 0;
  coupling_return = 3.5;
  const double result = *evaluator(hist.local().front(), hist.remote().front());
  if (local_arg.empty()) {
    CHECK(remote_arg.empty());
    // In one case this should use a cached result generated from the
    // call below this one in a previous call to this function.
    CHECK((result == 3.5 or result == 6.5));
  } else {
    ++coupling_calls;
    CHECK(local_arg == hist.local().data(0));
    CHECK(remote_arg == hist.remote().data(0));
    CHECK(result == 3.5);
  }
  local_arg.clear();
  remote_arg.clear();

  coupling_return = 6.5;
  CHECK(6.5 == *evaluator(hist.local()[3], hist.remote()[2]));
  if (local_arg.empty()) {
    CHECK(remote_arg.empty());
  } else {
    ++coupling_calls;
    CHECK(local_arg == hist.local().data(3));
    CHECK(remote_arg == hist.remote().data(2));
  }

  return coupling_calls;
}

template <bool EvaluatorOwnsCoupling>
void test_boundary_history() {
  BoundaryHistoryType history{};
  const BoundaryHistoryType& const_history = history;

  CHECK(const_history.local().empty());
  CHECK(const_history.remote().empty());

  {
    INFO("Local state");
    history.local().insert(make_time_id(0.), 10, get_output(0));
    history.local().insert(make_time_id(1.), 11, get_output(1));
    history.local().insert_initial(make_time_id(-1.), 9, get_output(-1));
    history.local().insert(make_time_id(2.), 12, get_output(2));

    check_local<true>(const_history.local());
    check_local<false, TimeSteppers::ConstBoundaryHistoryTimes>(
        const_history.local());
    check_local<true>(history.local());
    check_local<false, TimeSteppers::MutableBoundaryHistoryTimes>(
        history.local());
    check_local<false, TimeSteppers::ConstBoundaryHistoryTimes>(
        history.local());
  }

  {
    INFO("Remote state");
    history.remote().insert(make_time_id(1.), 21, std::vector{1});
    history.remote().insert_initial(make_time_id(0.), 20, std::vector{0});
    history.remote().insert_initial(make_time_id(-1.), 19, std::vector{-1});
    history.remote().insert(make_time_id(2.), 22, std::vector{2});
    history.remote().insert_initial(make_time_id(-2.), 18, std::vector{-2});
    history.remote().insert(make_time_id(3.), 23, std::vector{3});

    check_remote<true>(const_history.remote());
    check_remote<false, TimeSteppers::ConstBoundaryHistoryTimes>(
        const_history.remote());
    check_remote<true>(history.remote());
    check_remote<false, TimeSteppers::MutableBoundaryHistoryTimes>(
        history.remote());
    check_remote<false, TimeSteppers::ConstBoundaryHistoryTimes>(
        history.remote());
  }

  {
    INFO("Caching");
    CHECK(check_coupling_evaluation<EvaluatorOwnsCoupling>(const_history) == 2);
    CHECK(check_coupling_evaluation<EvaluatorOwnsCoupling>(const_history) == 0);
    history.clear_coupling_cache();
    CHECK(check_coupling_evaluation<EvaluatorOwnsCoupling>(const_history) == 2);
  }

  {
    INFO("Serialization");
    const auto copy = serialize_and_deserialize(history);
    check_local<true>(copy.local());
    check_remote<true>(copy.remote());
    CHECK(check_coupling_evaluation<EvaluatorOwnsCoupling>(copy) == 0);
  }

  {
    INFO("Streaming");
    const std::string expected_output_without_data =
        "Local Data:\n"
        " Time: 0:Slab[-1,-0.5]:0/1:0:-1 (order 9)\n"
        " Time: 0:Slab[0,0.5]:0/1:0:0 (order 10)\n"
        " Time: 0:Slab[1,1.5]:0/1:0:1 (order 11)\n"
        " Time: 0:Slab[2,2.5]:0/1:0:2 (order 12)\n"
        "Remote Data:\n"
        " Time: 0:Slab[-2,-1.5]:0/1:0:-2 (order 18)\n"
        " Time: 0:Slab[-1,-0.5]:0/1:0:-1 (order 19)\n"
        " Time: 0:Slab[0,0.5]:0/1:0:0 (order 20)\n"
        " Time: 0:Slab[1,1.5]:0/1:0:1 (order 21)\n"
        " Time: 0:Slab[2,2.5]:0/1:0:2 (order 22)\n"
        " Time: 0:Slab[3,3.5]:0/1:0:3 (order 23)\n";
    const std::string expected_output_with_data =
        "Local Data:\n"
        " Time: 0:Slab[-1,-0.5]:0/1:0:-1 (order 9)\n"
        "  Data: -1\n"
        " Time: 0:Slab[0,0.5]:0/1:0:0 (order 10)\n"
        "  Data: 0\n"
        " Time: 0:Slab[1,1.5]:0/1:0:1 (order 11)\n"
        "  Data: 1\n"
        " Time: 0:Slab[2,2.5]:0/1:0:2 (order 12)\n"
        "  Data: 2\n"
        "Remote Data:\n"
        " Time: 0:Slab[-2,-1.5]:0/1:0:-2 (order 18)\n"
        "  Data: (-2)\n"
        " Time: 0:Slab[-1,-0.5]:0/1:0:-1 (order 19)\n"
        "  Data: (-1)\n"
        " Time: 0:Slab[0,0.5]:0/1:0:0 (order 20)\n"
        "  Data: (0)\n"
        " Time: 0:Slab[1,1.5]:0/1:0:1 (order 21)\n"
        "  Data: (1)\n"
        " Time: 0:Slab[2,2.5]:0/1:0:2 (order 22)\n"
        "  Data: (2)\n"
        " Time: 0:Slab[3,3.5]:0/1:0:3 (order 23)\n"
        "  Data: (3)\n";
    CHECK(get_output(history) == expected_output_with_data);
    {
      std::ostringstream ss{};
      CHECK(&history.print<true>(ss) == &ss);
      CHECK(ss.str() == expected_output_with_data);
    }
    {
      std::ostringstream ss{};
      CHECK(&history.print<false>(ss) == &ss);
      CHECK(ss.str() == expected_output_without_data);
    }
  }

  {
    INFO("pop_front()");
    history.local().pop_front();
    CHECK(history.local().size() == 3);
    CHECK(history.remote().size() == 6);
    static_cast<const TimeSteppers::MutableBoundaryHistoryTimes&>(
        history.local())
        .pop_front();
    CHECK(history.local().size() == 2);
    CHECK(history.remote().size() == 6);
    history.remote().pop_front();
    CHECK(history.local().size() == 2);
    CHECK(history.remote().size() == 5);
    static_cast<const TimeSteppers::MutableBoundaryHistoryTimes&>(
        history.remote())
        .pop_front();
    CHECK(history.local().size() == 2);
    CHECK(history.remote().size() == 4);
  }

  history.local().pop_front();
  history.local().insert(make_time_id(3.), 13, get_output(3));
  history.local().insert(make_time_id(4.), 14, get_output(4));
  history.local().insert(make_time_id(5.), 15, get_output(5));
  REQUIRE(history.local().size() == 4);
  REQUIRE(history.remote().size() == 4);

  {
    INFO("Caching across steps");
    // check_coupling_evaluation tests (0, 0) and (3, 2).  We have
    // cleverly adjusted the old (3, 2) to be (0, 0).
    CHECK(check_coupling_evaluation<EvaluatorOwnsCoupling>(const_history) == 1);
  }

  {
    INFO("Data mutation");
    history.local().data(1) = "changed local 1";
    history.local().data(history.local()[2]) = "changed local 2";
    history.remote().data(1) = std::vector{1000};
    history.remote().data(history.remote()[2]) = std::vector{2000};
    CHECK(const_history.local().data(0) == get_output(2));
    CHECK(const_history.local().data(1) == "changed local 1");
    CHECK(const_history.local().data(2) == "changed local 2");
    CHECK(const_history.local().data(3) == get_output(5));
    CHECK(const_history.remote().data(0) == std::vector{0});
    CHECK(const_history.remote().data(1) == std::vector{1000});
    CHECK(const_history.remote().data(2) == std::vector{2000});
    CHECK(const_history.remote().data(3) == std::vector{3});
  }

  {
    INFO("clear()");
    static_cast<const TimeSteppers::MutableBoundaryHistoryTimes&>(
        history.local())
        .clear();
    CHECK(const_history.local().empty());
    CHECK(const_history.remote().size() == 4);

    history.local().insert(make_time_id(5.), 15, get_output(5));

    static_cast<const TimeSteppers::MutableBoundaryHistoryTimes&>(
        history.remote())
        .clear();
    CHECK(const_history.local().size() == 1);
    CHECK(const_history.remote().empty());
  }
}
}  // namespace

SPECTRE_TEST_CASE("Unit.Time.BoundaryHistory", "[Unit][Time]") {
  test_boundary_history<true>();
  test_boundary_history<false>();
}
