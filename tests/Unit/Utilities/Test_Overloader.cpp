// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "tests/Unit/TestingFramework.hpp"

#include "Utilities/Overloader.hpp"
#include "Utilities/TypeTraits.hpp"

namespace {
/// [overloader_example]
struct my_type1 {
  int func(int a) { return 2 * a; }
};

struct my_type2 {};

template <class T, class = cpp17::void_t<>>
struct has_func : std::false_type {};

template <class T>
struct has_func<
    T, cpp17::void_t<decltype(std::declval<T>().func(std::declval<int>()))>>
    : std::true_type {};

template <class T>
void func(T t) {
  auto my_lambdas =
      make_overloader([](auto& /*f*/, std::integral_constant<bool, false>,
                         std::integral_constant<bool, false>) { return 0; },
                      [](auto& /*f*/, std::integral_constant<bool, true>,
                         std::integral_constant<bool, false>) { return 1; },
                      [](auto& /*f*/, std::integral_constant<bool, false>,
                         std::integral_constant<bool, true>) { return 2; },
                      [](auto& /*f*/, std::integral_constant<bool, true>,
                         std::integral_constant<bool, true>) { return 3; });
  CHECK(static_cast<int>(has_func<T>::value) ==
        (my_lambdas(t, std::integral_constant<bool, has_func<T>::value>{},
                    std::integral_constant<bool, false>{})));
  CHECK(2 + static_cast<int>(has_func<T>::value) ==
        (my_lambdas(t, std::integral_constant<bool, has_func<T>::value>{},
                    std::integral_constant<bool, true>{})));
}

void caller() {
  func(my_type1{});
  func(my_type2{});
}
/// [overloader_example]
}  // namespace

SPECTRE_TEST_CASE("Unit.Utilities.Overloader", "[Unit][Utilities]") {
  auto my_lambdas =
      make_overloader([](std::integral_constant<int, 0>) { return 0; },
                      [](std::integral_constant<int, 1>) { return 1; },
                      [](std::integral_constant<int, 2>) { return 2; },
                      [](std::integral_constant<int, 3>) { return 3; },
                      [](std::integral_constant<int, 4>) { return 4; },
                      [](std::integral_constant<int, 5>) { return 5; },
                      [](std::integral_constant<int, 6>) { return 6; },
                      [](std::integral_constant<int, 7>) { return 7; },
                      [](std::integral_constant<int, 8>) { return 8; },
                      [](std::integral_constant<int, 9>) { return 9; },
                      [](std::integral_constant<int, 10>) { return 10; });
  CHECK((0 == my_lambdas(std::integral_constant<int, 0>{})));
  CHECK((1 == my_lambdas(std::integral_constant<int, 1>{})));
  CHECK((2 == my_lambdas(std::integral_constant<int, 2>{})));
  CHECK((3 == my_lambdas(std::integral_constant<int, 3>{})));
  CHECK((4 == my_lambdas(std::integral_constant<int, 4>{})));
  CHECK((5 == my_lambdas(std::integral_constant<int, 5>{})));
  CHECK((6 == my_lambdas(std::integral_constant<int, 6>{})));
  CHECK((7 == my_lambdas(std::integral_constant<int, 7>{})));
  CHECK((8 == my_lambdas(std::integral_constant<int, 8>{})));
  CHECK((9 == my_lambdas(std::integral_constant<int, 9>{})));
  CHECK((10 == my_lambdas(std::integral_constant<int, 10>{})));

  caller();
}
