// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <functional>
#include <limits>

#include "Framework/TestHelpers.hpp"
#include "Time/EvolutionOrdering.hpp"

namespace {
template <int>
struct StrangeComparison {
  explicit StrangeComparison(const int v) noexcept : value(v) {}
  StrangeComparison(const StrangeComparison&) = delete;
  StrangeComparison(StrangeComparison&&) = default;
  StrangeComparison& operator=(const StrangeComparison&) = delete;
  StrangeComparison& operator=(StrangeComparison&&) = default;
  ~StrangeComparison() = default;
  int value;
};

bool operator==(const StrangeComparison<1>& x,
                const StrangeComparison<1>& y) noexcept {
  return x.value == y.value;
}

// These are for testing types and forwarding, so it isn't important
// to do something meaningful with the values.  (And we never call
// this with I1 == I2, so we don't have to handle that.)
template <int I1, int I2>
const StrangeComparison<std::min(I1, I2)>& operator<(
    const StrangeComparison<I1>& x, const StrangeComparison<I2>& y) noexcept {
  if constexpr (I1 < I2) {
    return x;
  } else {
    return y;
  }
}

template <int I1, int I2>
decltype(auto) operator>(const StrangeComparison<I1>& x,
                         const StrangeComparison<I2>& y) noexcept {
  return x < y;
}

template <int I1, int I2>
decltype(auto) operator<=(const StrangeComparison<I1>& x,
                          const StrangeComparison<I2>& y) noexcept {
  return x < y;
}

template <int I1, int I2>
decltype(auto) operator>=(const StrangeComparison<I1>& x,
                          const StrangeComparison<I2>& y) noexcept {
  return x < y;
}

template <typename EvOp, typename StdOp, typename Arg1, typename Arg2>
void check_helper(const EvOp& ev_forward, const EvOp& ev_backward) noexcept {
  CHECK(ev_forward(Arg1{1}, Arg2{2}) == StdOp{}(Arg1{1}, Arg2{2}));
  CHECK(ev_forward(Arg1{2}, Arg2{1}) == StdOp{}(Arg1{2}, Arg2{1}));
  CHECK(ev_forward(Arg1{1}, Arg2{1}) == StdOp{}(Arg1{1}, Arg2{1}));

  CHECK(ev_backward(Arg1{1}, Arg2{2}) == StdOp{}(Arg2{2}, Arg1{1}));
  CHECK(ev_backward(Arg1{2}, Arg2{1}) == StdOp{}(Arg2{1}, Arg1{2}));
  CHECK(ev_backward(Arg1{1}, Arg2{1}) == StdOp{}(Arg2{1}, Arg1{1}));
}

template <template <typename...> class EvOp, template <typename...> class StdOp>
void check_op() noexcept {
  {
    INFO("Concrete");
    check_helper<EvOp<int>, StdOp<int>, int, int>({true}, {false});

    CHECK(EvOp<double>{true}.infinity() ==
          std::numeric_limits<double>::infinity());
    CHECK(EvOp<int>{true}.template infinity<double>() ==
          std::numeric_limits<double>::infinity());
    CHECK(EvOp<double>{false}.infinity() ==
          -std::numeric_limits<double>::infinity());
    CHECK(EvOp<int>{false}.template infinity<double>() ==
          -std::numeric_limits<double>::infinity());

    INFO("Serialization");
    check_helper<EvOp<int>, StdOp<int>, int, int>(
        serialize_and_deserialize(EvOp<int>{true}),
        serialize_and_deserialize(EvOp<int>{false}));
  }
  {
    INFO("Generic");
    check_helper<EvOp<>, StdOp<>, StrangeComparison<1>, StrangeComparison<2>>(
        {true}, {false});

    CHECK(EvOp<>{true}.template infinity<double>() ==
          std::numeric_limits<double>::infinity());
    CHECK(EvOp<>{false}.template infinity<double>() ==
          -std::numeric_limits<double>::infinity());

    INFO("Serialization");
    check_helper<EvOp<>, StdOp<>, StrangeComparison<1>, StrangeComparison<2>>(
        serialize_and_deserialize(EvOp<>{true}),
        serialize_and_deserialize(EvOp<>{false}));
  }
}
}  // namespace

SPECTRE_TEST_CASE("Unit.Time.EvolutionOrdering", "[Unit][Time]") {
  {
    INFO("less");
    check_op<evolution_less, std::less>();
  }
  {
    INFO("greater");
    check_op<evolution_greater, std::greater>();
  }
  {
    INFO("less_equal");
    check_op<evolution_less_equal, std::less_equal>();
  }
  {
    INFO("greater_equal");
    check_op<evolution_greater_equal, std::greater_equal>();
  }
}
