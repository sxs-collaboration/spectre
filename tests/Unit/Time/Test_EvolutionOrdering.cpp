// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "tests/Unit/TestingFramework.hpp"

#include <functional>
#include <tuple>

#include "Time/EvolutionOrdering.hpp"
#include "tests/Unit/TestHelpers.hpp"

namespace {
template <int>
struct StrangeComparison {
  int value;
};

#define STRANGE_OP(op)                                                      \
  template <int I1, int I2>                                                 \
  inline std::tuple<bool> operator op(StrangeComparison<I1>&& x,            \
                                      StrangeComparison<I2>&& y) noexcept { \
    return std::make_tuple(x.value op y.value);                             \
  }
STRANGE_OP(<)
STRANGE_OP(>)
STRANGE_OP(<=)
STRANGE_OP(>=)
#undef STRANGE_OP

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
    INFO("Serialization");
    check_helper<EvOp<int>, StdOp<int>, int, int>(
        serialize_and_deserialize(EvOp<int>{true}),
        serialize_and_deserialize(EvOp<int>{false}));
  }
  {
    INFO("Generic");
    check_helper<EvOp<>, StdOp<>, StrangeComparison<1>, StrangeComparison<2>>(
        {true}, {false});
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
