// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <string>
#include <type_traits>
#include <typeindex>
#include <typeinfo>
#include <utility>
#include <vector>

// [include]
#include "Utilities/TMPL.hpp"
// [include]

// These are disabled in TMPL.hpp, but are in the brigand.hpp include
// if not disabled.  Trying to include any Brigand files before
// TMPL.hpp has unpredictable results because there may be an extra
// TMPL.hpp include at the start from the PCH.
#include <brigand/adapted/fusion.hpp>
#include <brigand/adapted/variant.hpp>

// We want the code to be nicely formatted for the documentation, not here.
// clang-format off
namespace {

template <typename T, typename U>
void assert_same() {
  static_assert(std::is_same_v<T, U>);
}

template <typename Map, typename Supermap>
using is_submap =
    tmpl::all<tmpl::keys_as_sequence<Map, tmpl::list>,
              std::is_same<tmpl::lazy::lookup<tmpl::pin<Map>, tmpl::_1>,
                           tmpl::lazy::lookup<tmpl::pin<Supermap>, tmpl::_1>>>;

template <typename T, typename U>
void assert_maps_same() {
  static_assert(is_submap<T, U>::value and is_submap<U, T>::value);
}

template <typename Set, typename Superset>
using is_subset =
    tmpl::all<Set, tmpl::bind<tmpl::contains, tmpl::pin<Superset>, tmpl::_1>>;

template <typename T, typename U>
void assert_sets_same() {
  static_assert(is_subset<T, U>::value and is_subset<U, T>::value);
}

#define HAS_LAZY_VERSION(name) \
  assert_same<tmpl::wrap<lazy_test_arguments, tmpl::lazy::name>::type, \
              tmpl::wrap<lazy_test_arguments, tmpl::name>>()

// [example_declarations]
struct Type1;
struct Type2;
struct Type3;

// Sequence containers.  Practical applications will usually use
// tmpl::list in place of these.
template <typename...>
struct List1;
template <typename...>
struct List2;
template <typename...>
struct List3;

template <typename>
struct Wrapper;

template <typename... T>
struct lazy_make_list1 {
  using type = List1<T...>;
};

// A comparator on the above types defining Type1 < Type2 < Type3
using CompareType123 = tmpl::and_<tmpl::or_<std::is_same<tmpl::_1, Type1>,
                                            std::is_same<tmpl::_2, Type3>>,
                                  tmpl::not_<std::is_same<tmpl::_1, tmpl::_2>>>;
// [example_declarations]

namespace metafunctions {
// [metafunctions:lazy]
template <typename T>
struct lazy_make_list_of_T_and_int {
  using type = List1<T, int>;
};
// [metafunctions:lazy]

// [metafunctions:eager]
template <typename T>
using eager_make_list_of_T_and_int = List1<T, int>;
// [metafunctions:eager]

// [metafunctions:eager_from_lazy]
template <typename T>
using eager_make_list_of_T_and_int2 =
    typename lazy_make_list_of_T_and_int<T>::type;
// [metafunctions:eager_from_lazy]

// [metafunctions:call_lazy_metafunction]
template <typename Func, typename... Args>
struct apply_lazy_metafunction;
template <template <typename...> typename Func, typename... DummyArgs,
          typename... Args>
struct apply_lazy_metafunction<Func<DummyArgs...>, Args...> {
  using type = typename Func<Args...>::type;
};

template <typename Func>
using call_lazy_metafunction_twice =
    List1<typename apply_lazy_metafunction<Func, double>::type,
          typename apply_lazy_metafunction<Func, char>::type>;
// [metafunctions:call_lazy_metafunction]

// [metafunctions:call_eager_metafunction]
template <typename Func, typename... Args>
struct apply_eager_metafunction;
template <template <typename...> typename Func, typename... DummyArgs,
          typename... Args>
struct apply_eager_metafunction<Func<DummyArgs...>, Args...> {
  using type = Func<Args...>;
};

template <typename Func>
using call_eager_metafunction_twice =
    List1<typename apply_eager_metafunction<Func, double>::type,
          typename apply_eager_metafunction<Func, char>::type>;
// [metafunctions:call_eager_metafunction]

namespace pin_protect_eager {
// [tmpl::pin:protect_eager]
template <typename T>
using is_int = tmpl::apply<std::is_same<tmpl::pin<T>, int>>;
static_assert(is_int<int>::value);
// Breaks without tmpl::pin: tmpl::list<> has no ::type
static_assert(not is_int<tmpl::list<>>::value);
// [tmpl::pin:protect_eager]
}  // namespace pin_protect_eager

// [metafunctions:evens]
template <typename L>
using evens = tmpl::filter<
  L, tmpl::equal_to<tmpl::modulo<tmpl::_1, tmpl::integral_constant<int, 2>>,
                    tmpl::integral_constant<int, 0>>>;
// [metafunctions:evens]

// [metafunctions:maybe_first]
template <typename L>
using maybe_first = tmpl::apply<tmpl::apply<
  tmpl::if_<std::bool_constant<(tmpl::size<L>::value != 0)>,
            tmpl::defer<tmpl::bind<tmpl::front, tmpl::pin<L>>>,
            tmpl::no_such_type_>>>;
// [metafunctions:maybe_first]

// [metafunctions:factorial]
template <typename N>
using factorial =
  tmpl::fold<tmpl::range<typename N::value_type, 1, N::value + 1>,
             tmpl::integral_constant<typename N::value_type, 1>,
             tmpl::times<tmpl::_state, tmpl::_element>>;
// [metafunctions:factorial]

// [metafunctions:factorial_cpp]
size_t factorial_cpp(const size_t n) {
  std::vector<size_t> range(n);
  std::iota(range.begin(), range.end(), 1);
  size_t state = 1;
  for (size_t i : range) {
    state = state * i;
  }
  return state;
}
// [metafunctions:factorial_cpp]

// [metafunctions:make_subtracter]
template <typename N>
using make_subtracter =
    tmpl::apply<tmpl::defer<tmpl::minus<tmpl::_1, tmpl::parent<tmpl::_1>>>, N>;
// [metafunctions:make_subtracter]

// [metafunctions:make_subtracter_simple]
template <typename N>
using make_subtracter_simple = tmpl::minus<tmpl::_1, tmpl::pin<N>>;
// [metafunctions:make_subtracter_simple]

// [metafunctions:multiplication_table]
template <typename N>
using multiplication_table =
  tmpl::transform<
    tmpl::range<typename N::value_type, 1, N::value + 1>,
    tmpl::lazy::transform<
      tmpl::pin<tmpl::range<typename N::value_type, 1, N::value + 1>>,
      tmpl::defer<tmpl::times<tmpl::_1, tmpl::parent<tmpl::_1>>>>>;
// [metafunctions:multiplication_table]

// [metafunctions:column_with_zeros]
template <typename Lists, typename Column>
using column_with_zeros =
  tmpl::transform<
    Lists,
    tmpl::bind<
      tmpl::apply,
      tmpl::if_<
        tmpl::greater<tmpl::bind<tmpl::size, tmpl::_1>, Column>,
        tmpl::defer<  // avoid out-of-range call to `at`
          tmpl::parent<
            tmpl::bind<tmpl::at, tmpl::_1, Column>>>,
        tmpl::size_t<0>>>>;
// [metafunctions:column_with_zeros]

// [metafunctions:factorial_recursion]
template <typename N>
using factorial_recursion =
  tmpl::apply<
    tmpl::bind<tmpl::apply, tmpl::_1, tmpl::_1, N>,
    tmpl::bind<  // recursive metalambda starts here
      tmpl::apply,
      tmpl::if_<
        tmpl::not_equal_to<tmpl::_2, tmpl::size_t<0>>,
        tmpl::defer<  // prevent speculative recursion
          tmpl::parent<
            tmpl::times<
              tmpl::_2,
              tmpl::bind<tmpl::apply, tmpl::_1,
                         tmpl::_1, tmpl::prev<tmpl::_2>>>>>,
        tmpl::integral_constant<typename N::value_type, 1>>>>;
// [metafunctions:factorial_recursion]

// [metafunctions:primes]
template <typename N>
using zero = tmpl::integral_constant<typename N::value_type, 0>;

// Return Sequence with the Nth element replaced by NewType.
template <typename Sequence, typename N, typename NewType>
using replace_at =
  tmpl::append<tmpl::front<tmpl::split_at<Sequence, N>>,
               tmpl::list<NewType>,
               tmpl::pop_front<tmpl::back<tmpl::split_at<Sequence, N>>>>;

template <typename Start, typename End>
using range_from_types =
  tmpl::range<typename Start::value_type, Start::value, End::value>;

template <typename N>
using primes =
  tmpl::remove<
    tmpl::fold<
      tmpl::range<typename N::value_type, 2, N::value>,
      tmpl::push_front<
        tmpl::range<typename N::value_type, 2, N::value>, zero<N>, zero<N>>,
      tmpl::bind<
        tmpl::apply,
        tmpl::if_<  // Skip work for known-composite entries
          tmpl::or_<
            std::is_same<
              tmpl::bind<tmpl::at, tmpl::_state, tmpl::_element>, zero<N>>,
            // Only iteration up to sqrt(N) is necessary
            tmpl::greater_equal<
              tmpl::times<tmpl::_element, tmpl::_element>, N>>,
          tmpl::defer<  // Match other branch (don't execute the state)
            tmpl::parent<tmpl::_state>>,
          tmpl::defer<
            tmpl::parent<
              tmpl::lazy::fold<
                tmpl::bind<
                  range_from_types,
                  tmpl::_element,
                  tmpl::next<tmpl::divides<tmpl::prev<N>, tmpl::_element>>>,
                tmpl::_state,
                tmpl::defer<  // Passed as a closure to the inner fold
                  tmpl::bind<
                    replace_at,
                    tmpl::_state,
                    tmpl::times<tmpl::parent<tmpl::_element>, tmpl::_element>,
                    zero<N>>>>>>>>>,
    zero<N>>;
// [metafunctions:primes]

// [metafunctions:primes_cpp]
std::vector<size_t> primes_cpp(const size_t n) {
  std::vector<size_t> sieve(n, 0);
  std::iota(sieve.begin() + 2, sieve.end(), 2);
  for (size_t i = 2; i < n; ++i) {
    if (not (sieve[i] == 0 or i * i >= n)) {
      for (size_t j = i; j < (n - 1) / i + 1; ++j) {
        sieve[i * j] = 0;
      }
    }
  }
  std::vector<size_t> result;
  std::copy_if(sieve.begin(), sieve.end(),
               std::insert_iterator(result, result.begin()),
               [](const size_t x) { return x != 0; });
  return result;
}
// [metafunctions:primes_cpp]

void run() {
// [metafunctions:agreement]
assert_same<eager_make_list_of_T_and_int<double>,
            eager_make_list_of_T_and_int2<double>>();
// [metafunctions:agreement]

// [metafunctions:call_lazy_metafunction_assert]
struct Dummy;

assert_same<call_lazy_metafunction_twice<lazy_make_list_of_T_and_int<Dummy>>,
            List1<List1<double, int>, List1<char, int>>>();
// [metafunctions:call_lazy_metafunction_assert]

// [metafunctions:call_eager_metafunction_assert]
assert_same<call_eager_metafunction_twice<eager_make_list_of_T_and_int<Dummy>>,
            List1<List1<double>, List1<char>>>();
// [metafunctions:call_eager_metafunction_assert]

// [tmpl::args]
static_assert(not std::is_same_v<tmpl::_1, tmpl::args<0>>);
static_assert(not std::is_same_v<tmpl::_2, tmpl::args<1>>);
static_assert(std::is_same_v<tmpl::_3, tmpl::args<2>>);
static_assert(std::is_same_v<tmpl::_4, tmpl::args<3>>);
// [tmpl::args]

// [tmpl::args:eval]
assert_same<tmpl::apply<tmpl::_1, Type1, Type2>, Type1>();
assert_same<tmpl::apply<tmpl::_2, Type1, Type2>, Type2>();
assert_same<tmpl::apply<tmpl::args<0>, Type1, Type2>, Type1>();
// [tmpl::args:eval]

// [metalambda_lazy]
assert_same<tmpl::apply<lazy_make_list1<tmpl::_1, tmpl::_2>, Type1, Type2>,
            List1<Type1, Type2>>();
// [metalambda_lazy]

// [tmpl::bind]
assert_same<tmpl::apply<tmpl::bind<List1, tmpl::_1, tmpl::_2>, Type1, Type2>,
            List1<Type1, Type2>>();
// [tmpl::bind]

// [tmpl::pin]
assert_same<tmpl::apply<tmpl::pin<List1<Type1>>>, List1<Type1>>();
// Error: List1 is not a lazy metafunction
// assert_same<tmpl::apply<List1<Type1>>, List1<Type1>>();
assert_same<tmpl::apply<tmpl::pin<tmpl::_1>, Type1>, tmpl::_1>();
// [tmpl::pin]

// [tmpl::defer]
assert_same<
  tmpl::apply<tmpl::apply<tmpl::defer<tmpl::_1>,
                          Type1>,
              Type2>,
  Type2>();
// [tmpl::defer]

// [tmpl::parent]
assert_same<
  tmpl::apply<tmpl::apply<tmpl::defer<tmpl::parent<tmpl::_1>>,
                          Type1>,
              Type2>,
  Type1>();
// [tmpl::parent]

// [metalambda_constant]
assert_same<tmpl::apply<Type1>, Type1>();
// [metalambda_constant]

// [metafunctions:evens:asserts]
assert_same<evens<tmpl::integral_list<int, 1, 1, 2, 3, 5, 8, 13>>,
            tmpl::integral_list<int, 2, 8>>();
// [metafunctions:evens:asserts]

// [metafunctions:maybe_first:asserts]
assert_same<maybe_first<tmpl::list<Type1>>, Type1>();
assert_same<maybe_first<tmpl::list<Type1, Type2>>, Type1>();
assert_same<maybe_first<tmpl::list<>>, tmpl::no_such_type_>();
// [metafunctions:maybe_first:asserts]

// [metafunctions:factorial:asserts]
assert_same<factorial<tmpl::size_t<5>>, tmpl::size_t<120>>();
// [metafunctions:factorial:asserts]
CHECK(factorial_cpp(5) == 120);

{
// [metafunctions:make_subtracter:asserts]
using subtract_three = make_subtracter<tmpl::size_t<3>>;
assert_same<tmpl::apply<subtract_three, tmpl::size_t<5>>, tmpl::size_t<2>>();
// [metafunctions:make_subtracter:asserts]
assert_same<tmpl::apply<make_subtracter_simple<tmpl::size_t<3>>,
                        tmpl::size_t<5>>,
            tmpl::size_t<2>>();
}

// [metafunctions:multiplication_table:asserts]
assert_same<multiplication_table<tmpl::size_t<5>>,
            tmpl::list<tmpl::integral_list<size_t, 1, 2, 3, 4, 5>,
                       tmpl::integral_list<size_t, 2, 4, 6, 8, 10>,
                       tmpl::integral_list<size_t, 3, 6, 9, 12, 15>,
                       tmpl::integral_list<size_t, 4, 8, 12, 16, 20>,
                       tmpl::integral_list<size_t, 5, 10, 15, 20, 25>>>();
// [metafunctions:multiplication_table:asserts]

// [metafunctions:factorial_recursion:asserts]
assert_same<factorial_recursion<tmpl::size_t<5>>, tmpl::size_t<120>>();
// [metafunctions:factorial_recursion:asserts]

// [metafunctions:column_with_zeros:asserts]
assert_same<
  column_with_zeros<
    tmpl::list<tmpl::integral_list<size_t, 11, 12, 13>,
               tmpl::integral_list<size_t, 21, 22, 23, 24, 25>,
               tmpl::integral_list<size_t, 31, 32, 33, 34>>,
    tmpl::size_t<3>>,
  tmpl::integral_list<size_t, 0, 24, 34>>();
// [metafunctions:column_with_zeros:asserts]

// [metafunctions:primes:asserts]
assert_same<
  primes<tmpl::size_t<100>>,
  tmpl::integral_list<size_t, 2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41,
                      43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97>>();
// [metafunctions:primes:asserts]
// Check edge cases:
// Prime
assert_same<
  primes<tmpl::size_t<11>>,
  tmpl::integral_list<size_t, 2, 3, 5, 7>>();
// Prime + 1
assert_same<
  primes<tmpl::size_t<12>>,
  tmpl::integral_list<size_t, 2, 3, 5, 7, 11>>();
// Prime^2
assert_same<
  primes<tmpl::size_t<25>>,
  tmpl::integral_list<size_t, 2, 3, 5, 7, 11, 13, 17, 19, 23>>();

CHECK(primes_cpp(100) ==
      std::vector<size_t>{2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41,
                          43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97});
CHECK(primes_cpp(11) == std::vector<size_t>{2, 3, 5, 7});
CHECK(primes_cpp(12) == std::vector<size_t>{2, 3, 5, 7, 11});
CHECK(primes_cpp(25) == std::vector<size_t>{2, 3, 5, 7, 11, 13, 17, 19, 23});
}
}  // namespace metafunctions

namespace guidelines {
namespace lazy_eager {
// [guidelines:lazy_eager]
template <typename T>
struct always_true : std::true_type {};
template <typename T>
using always_true_t = typename always_true<T>::type;
template <typename T>
constexpr bool always_true_v = always_true<T>::value;
// [guidelines:lazy_eager]
static_assert(std::is_same_v<always_true_t<int>, std::true_type>);
static_assert(always_true_v<int>);
}  // namespace lazy_eager
}  // namespace guidelines

namespace containers {
void run() {
// [tmpl::integral_constant]
using T = tmpl::integral_constant<int, 3>;
assert_same<T::value_type, int>();
assert_same<T::type, T>();
static_assert(T::value == 3);

// At runtime only
CHECK(T{} == 3);
CHECK(T{}() == 3);
// [tmpl::integral_constant]

// [tmpl::integral_constant::abbreviations]
assert_same<tmpl::int8_t<3>, tmpl::integral_constant<int8_t, 3>>();
assert_same<tmpl::int16_t<3>, tmpl::integral_constant<int16_t, 3>>();
assert_same<tmpl::int32_t<3>, tmpl::integral_constant<int32_t, 3>>();
assert_same<tmpl::int64_t<3>, tmpl::integral_constant<int64_t, 3>>();

assert_same<tmpl::uint8_t<3>, tmpl::integral_constant<uint8_t, 3>>();
assert_same<tmpl::uint16_t<3>, tmpl::integral_constant<uint16_t, 3>>();
assert_same<tmpl::uint32_t<3>, tmpl::integral_constant<uint32_t, 3>>();
assert_same<tmpl::uint64_t<3>, tmpl::integral_constant<uint64_t, 3>>();

assert_same<tmpl::size_t<3>, tmpl::integral_constant<size_t, 3>>();
assert_same<tmpl::ptrdiff_t<3>, tmpl::integral_constant<ptrdiff_t, 3>>();
assert_same<tmpl::bool_<true>, tmpl::integral_constant<bool, true>>();
// [tmpl::integral_constant::abbreviations]

// [tmpl::list]
static_assert(not std::is_same_v<tmpl::list<Type1, Type2>,
                                 tmpl::list<Type2, Type1>>);
// [tmpl::list]

// [tmpl::map]
assert_same<tmpl::lookup<tmpl::map<tmpl::pair<Type1, int>,
                                   tmpl::pair<Type2, double>>,
                         Type1>,
            int>();
// [tmpl::map]

// [tmpl::pair]
assert_same<tmpl::pair<Type1, Type2>::first_type, Type1>();
assert_same<tmpl::pair<Type1, Type2>::second_type, Type2>();
// [tmpl::pair]

// [tmpl::set]
assert_same<tmpl::contains<tmpl::set<Type1, Type2>, Type1>, tmpl::true_type>();
// [tmpl::set]

// [tmpl::type_]
assert_same<tmpl::type_<Type1>::type, Type1>();
// [tmpl::type_]
}
}  // namespace containers

namespace constants {
void run() {
// [tmpl::empty_base]
assert_same<tmpl::inherit_linearly<List1<>, List2<>>, tmpl::empty_base>();
// [tmpl::empty_base]

// [tmpl::empty_sequence]
assert_same<tmpl::empty_sequence, tmpl::list<>>();
// [tmpl::empty_sequence]

// [tmpl::false_type]
assert_same<tmpl::false_type, tmpl::bool_<false>>();
// [tmpl::false_type]

// [tmpl::no_such_type_]
assert_same<tmpl::index_of<List1<>, Type1>, tmpl::no_such_type_>();
// [tmpl::no_such_type_]

// [tmpl::true_type]
assert_same<tmpl::true_type, tmpl::bool_<true>>();
// [tmpl::true_type]
}
}  // namespace constants

namespace list_constructors {
void run() {
// [tmpl::filled_list]
assert_same<tmpl::filled_list<Type1, 3, List1>, List1<Type1, Type1, Type1>>();
assert_same<tmpl::filled_list<Type1, 3>, tmpl::list<Type1, Type1, Type1>>();
// [tmpl::filled_list]

// [tmpl::integral_list]
assert_same<tmpl::integral_list<int, 3, 2, 1>,
            tmpl::list<tmpl::integral_constant<int, 3>,
                       tmpl::integral_constant<int, 2>,
                       tmpl::integral_constant<int, 1>>>();
// [tmpl::integral_list]

// [tmpl::make_sequence]
assert_same<tmpl::make_sequence<tmpl::size_t<5>, 3>,
            tmpl::list<tmpl::size_t<5>, tmpl::size_t<6>, tmpl::size_t<7>>>();
assert_same<tmpl::make_sequence<Type1, 3, lazy_make_list1<tmpl::_1>, List2>,
            List2<Type1, List1<Type1>, List1<List1<Type1>>>>();
// [tmpl::make_sequence]

// [tmpl::range]
assert_same<tmpl::range<size_t, 4, 7>,
            tmpl::list<tmpl::size_t<4>, tmpl::size_t<5>, tmpl::size_t<6>>>();
assert_same<tmpl::range<size_t, 4, 4>, tmpl::list<>>();
// [tmpl::range]

// [tmpl::reverse_range]
assert_same<tmpl::reverse_range<size_t, 7, 4>,
            tmpl::list<tmpl::size_t<7>, tmpl::size_t<6>, tmpl::size_t<5>>>();
assert_same<tmpl::reverse_range<size_t, 7, 7>, tmpl::list<>>();
// [tmpl::reverse_range]
}
}  // namespace list_constructors

namespace list_query {
void run() {
// [tmpl::all]
assert_same<tmpl::all<List1<tmpl::size_t<0>, tmpl::size_t<1>, tmpl::size_t<2>>>,
            tmpl::false_type>();
assert_same<tmpl::all<List1<tmpl::size_t<1>, tmpl::size_t<1>, tmpl::size_t<2>>>,
            tmpl::true_type>();
assert_same<tmpl::all<List1<tmpl::size_t<0>, tmpl::size_t<1>, tmpl::size_t<2>>,
                      tmpl::less<tmpl::_1, tmpl::size_t<2>>>,
            tmpl::false_type>();
assert_same<tmpl::all<List1<tmpl::size_t<0>, tmpl::size_t<1>, tmpl::size_t<0>>,
                      tmpl::less<tmpl::_1, tmpl::size_t<2>>>,
            tmpl::true_type>();
assert_same<tmpl::all<List1<>>, tmpl::true_type>();
// [tmpl::all]

// [tmpl::all:inhomogeneous]
assert_same<tmpl::all<List1<std::true_type, tmpl::true_type>, tmpl::_1>,
            tmpl::false_type>();
// [tmpl::all:inhomogeneous]

// [tmpl::any]
assert_same<tmpl::any<List1<tmpl::size_t<0>, tmpl::size_t<1>, tmpl::size_t<2>>>,
            tmpl::true_type>();
assert_same<tmpl::any<List1<tmpl::size_t<0>, tmpl::size_t<0>, tmpl::size_t<0>>>,
            tmpl::false_type>();
assert_same<tmpl::any<List1<tmpl::size_t<0>, tmpl::size_t<1>, tmpl::size_t<2>>,
                      tmpl::less<tmpl::_1, tmpl::size_t<2>>>,
            tmpl::true_type>();
assert_same<tmpl::any<List1<tmpl::size_t<4>, tmpl::size_t<3>, tmpl::size_t<2>>,
                      tmpl::less<tmpl::_1, tmpl::size_t<2>>>,
            tmpl::false_type>();
assert_same<tmpl::any<List1<>>, tmpl::false_type>();
// [tmpl::any]

// [tmpl::any:inhomogeneous]
assert_same<tmpl::any<List1<std::false_type, tmpl::false_type>, tmpl::_1>,
            tmpl::true_type>();
// [tmpl::any:inhomogeneous]

// [tmpl::at]
assert_same<tmpl::at<List1<Type1, Type2, Type3>, tmpl::size_t<0>>, Type1>();
// [tmpl::at]

// [tmpl::at_c]
assert_same<tmpl::at_c<List1<Type1, Type2, Type3>, 0>, Type1>();
// [tmpl::at_c]

// [tmpl::back]
assert_same<tmpl::back<List1<Type1, Type2, Type3>>, Type3>();
// [tmpl::back]

// [tmpl::count_if]
assert_same<tmpl::count_if<List1<Type1, Type2, Type1>,
                           std::is_same<tmpl::_1, Type1>>,
            tmpl::integral_constant<size_t, 2>>();
// [tmpl::count_if]

{
using lazy_test_arguments =
  tmpl::list<List2<Type1, Type2>, Type3,
             lazy_make_list1<tmpl::_state, tmpl::_element>>;
// [tmpl::fold]
assert_same<tmpl::fold<List2<Type1, Type2>, Type3,
                       lazy_make_list1<tmpl::_state, tmpl::_element>>,
            List1<List1<Type3, Type1>, Type2>>();
HAS_LAZY_VERSION(fold);
// [tmpl::fold]
}

// [tmpl::found]
assert_same<
  tmpl::found<List1<Type1, Type2, Type2, Type3>, std::is_same<tmpl::_1, Type2>>,
  tmpl::true_type>();
assert_same<
  tmpl::found<List1<Type1, Type1, Type1, Type3>, std::is_same<tmpl::_1, Type2>>,
  tmpl::false_type>();
assert_same<
  tmpl::found<List1<tmpl::size_t<0>, tmpl::size_t<1>, tmpl::size_t<2>>>,
  tmpl::true_type>();
// [tmpl::found]

// [tmpl::front]
assert_same<tmpl::front<List1<Type1, Type2, Type3>>, Type1>();
// [tmpl::front]

// [tmpl::index_if]
assert_same<tmpl::index_if<List1<Type1, Type2, Type3>,
                           std::is_same<Type3, tmpl::_1>>,
            tmpl::size_t<2>>();
assert_same<tmpl::index_if<List1<Type1, Type3, Type3>,
                           std::is_same<Type3, tmpl::_1>>,
            tmpl::size_t<1>>();
assert_same<tmpl::index_if<List1<Type1>, std::is_same<Type3, tmpl::_1>>,
            tmpl::no_such_type_>();
assert_same<tmpl::index_if<List1<Type1>, std::is_same<Type3, tmpl::_1>, Type2>,
            Type2>();
// [tmpl::index_if]

// [tmpl::index_of]
assert_same<tmpl::index_of<List1<Type1, Type2, Type3>, Type3>,
            tmpl::size_t<2>>();
assert_same<tmpl::index_of<List1<Type1, Type3, Type3>, Type3>,
            tmpl::size_t<1>>();
assert_same<tmpl::index_of<List1<Type1>, Type2>,
            tmpl::no_such_type_>();
// [tmpl::index_of]

// [tmpl::list_contains]
assert_same<tmpl::list_contains<List1<Type1, Type2>, Type1>, tmpl::true_type>();
static_assert(tmpl::list_contains_v<List1<Type1, Type2>, Type1>);
static_assert(not tmpl::list_contains_v<List1<Type2, Type2>, Type1>);
// [tmpl::list_contains]

// [tmpl::none]
assert_same<
  tmpl::none<List1<tmpl::size_t<0>, tmpl::size_t<1>, tmpl::size_t<2>>>,
  tmpl::false_type>();
assert_same<
  tmpl::none<List1<tmpl::size_t<0>, tmpl::size_t<0>, tmpl::size_t<0>>>,
  tmpl::true_type>();
assert_same<
  tmpl::none<List1<tmpl::size_t<0>, tmpl::size_t<1>, tmpl::size_t<2>>,
             tmpl::less<tmpl::_1, tmpl::size_t<2>>>,
  tmpl::false_type>();
assert_same<
  tmpl::none<List1<tmpl::size_t<4>, tmpl::size_t<3>, tmpl::size_t<2>>,
             tmpl::less<tmpl::_1, tmpl::size_t<2>>>,
  tmpl::true_type>();
assert_same<tmpl::none<List1<>>, tmpl::true_type>();
// [tmpl::none]

// [tmpl::none:inhomogeneous]
assert_same<tmpl::none<List1<std::false_type, tmpl::false_type>, tmpl::_1>,
            tmpl::false_type>();
// [tmpl::none:inhomogeneous]

// [tmpl::not_found]
assert_same<
  tmpl::not_found<List1<Type1, Type2, Type2, Type3>,
                  std::is_same<tmpl::_1, Type2>>,
  tmpl::false_type>();
assert_same<
  tmpl::not_found<List1<Type1, Type1, Type1, Type3>,
                  std::is_same<tmpl::_1, Type2>>,
  tmpl::true_type>();
assert_same<
  tmpl::not_found<List1<tmpl::size_t<0>, tmpl::size_t<1>, tmpl::size_t<2>>>,
  tmpl::false_type>();
// [tmpl::not_found]

// [tmpl::size]
assert_same<tmpl::size<List1<Type1, Type1>>,
            tmpl::integral_constant<unsigned int, 2>>();
// [tmpl::size]
}
}  // namespace list_query

namespace list_from_list {
void run() {
{
using lazy_test_arguments =
  tmpl::list<List1<Type1, Type2>, List2<>, List2<Type2>>;
// [tmpl::append]
assert_same<tmpl::append<List1<Type1, Type2>, List2<>, List2<Type2>>,
            List1<Type1, Type2, Type2>>();
assert_same<tmpl::append<>, tmpl::list<>>();
HAS_LAZY_VERSION(append);
// [tmpl::append]
}

// [tmpl::clear]
assert_same<tmpl::clear<List1<Type1>>, List1<>>();
// [tmpl::clear]

// [tmpl::erase]
assert_same<tmpl::erase<List1<Type1, Type2, Type3>, tmpl::size_t<1>>,
            List1<Type1, Type3>>();
// [tmpl::erase]

// [tmpl::erase_c]
assert_same<tmpl::erase_c<List1<Type1, Type2, Type3>, 1>,
            List1<Type1, Type3>>();
// [tmpl::erase_c]

{
using lazy_test_arguments = tmpl::list<List1<Type1, Type2, Type1, Type3>,
                                       std::is_same<Type1, tmpl::_1>>;
// [tmpl::filter]
assert_same<tmpl::filter<List1<Type1, Type2, Type1, Type3>,
                         std::is_same<Type1, tmpl::_1>>,
            List1<Type1, Type1>>();
HAS_LAZY_VERSION(filter);
// [tmpl::filter]
}

{
using lazy_test_arguments =
  tmpl::list<List1<Type1, Type2, Type2, Type3>, std::is_same<tmpl::_1, Type2>>;
// [tmpl::find]
assert_same<
  tmpl::find<List1<Type1, Type2, Type2, Type3>, std::is_same<tmpl::_1, Type2>>,
  List1<Type2, Type2, Type3>>();
assert_same<
  tmpl::find<List1<Type1, Type1, Type1, Type3>, std::is_same<tmpl::_1, Type2>>,
  List1<>>();
assert_same<
  tmpl::find<List1<tmpl::size_t<0>, tmpl::size_t<1>, tmpl::size_t<2>>>,
  List1<tmpl::size_t<1>, tmpl::size_t<2>>>();
HAS_LAZY_VERSION(find);
// [tmpl::find]
}

{
using lazy_test_arguments =
  tmpl::list<List1<List1<Type1, List1<Type2>>, List2<List1<Type3>>>>;
// [tmpl::flatten]
assert_same<
  tmpl::flatten<List1<List1<Type1, List1<Type2>>, List2<List1<Type3>>>>,
  List1<Type1, Type2, List2<List1<Type3>>>>();
HAS_LAZY_VERSION(flatten);
// [tmpl::flatten]
}

{
using lazy_test_arguments =
  tmpl::list<List3<List1<Type1, Type2>, List2<>, List2<Type2>>>;
// [tmpl::join]
assert_same<tmpl::join<List3<List1<Type1, Type2>, List2<>, List2<Type2>>>,
            List1<Type1, Type2, Type2>>();
assert_same<tmpl::join<List1<>>, tmpl::list<>>();
HAS_LAZY_VERSION(join);
// [tmpl::join]
}

// [tmpl::list_difference]
assert_same<tmpl::list_difference<List1<Type1, Type2, Type1, Type2>,
                                  List2<Type3, Type2>>,
            List1<Type1, Type1>>();
// [tmpl::list_difference]

{
// [tmpl::merge]
assert_same<
  tmpl::merge<List1<tmpl::size_t<1>, tmpl::size_t<2>, tmpl::size_t<5>>,
              List2<tmpl::size_t<1>, tmpl::size_t<3>, tmpl::size_t<6>>>,
  List1<tmpl::size_t<1>, tmpl::size_t<1>, tmpl::size_t<2>, tmpl::size_t<3>,
        tmpl::size_t<5>, tmpl::size_t<6>>>();
assert_same<tmpl::merge<List1<Type1, Type2>, List2<Type1, Type3>,
                        CompareType123>,
            List1<Type1, Type1, Type2, Type3>>();
// [tmpl::merge]

// [tmpl::merge:equiv]
assert_same<tmpl::merge<List1<Type1, Type1>, List2<Type2, Type2>,
                        std::false_type>,
            List1<Type2, Type2, Type1, Type1>>();
// [tmpl::merge:equiv]
}

// [tmpl::partition]
assert_same<tmpl::partition<List1<Type1, Type2, Type1, Type3>,
                            std::is_same<Type1, tmpl::_1>>,
            tmpl::pair<List1<Type1, Type1>, List1<Type2, Type3>>>();
// [tmpl::partition]

// [tmpl::pop_back]
assert_same<tmpl::pop_back<List1<Type1, Type2, Type3>>, List1<Type1, Type2>>();
assert_same<tmpl::pop_back<List1<Type1, Type2, Type3>,
                           tmpl::integral_constant<unsigned int, 2>>,
            List1<Type1>>();
// [tmpl::pop_back]

{
using lazy_test_arguments =
  tmpl::list<List1<Type1, Type2, Type3>,
             tmpl::integral_constant<unsigned int, 2>>;
// [tmpl::pop_front]
assert_same<tmpl::pop_front<List1<Type1, Type2, Type3>>,
            List1<Type2, Type3>>();
assert_same<tmpl::pop_front<List1<Type1, Type2, Type3>,
                            tmpl::integral_constant<unsigned int, 2>>,
            List1<Type3>>();
HAS_LAZY_VERSION(pop_front);
// [tmpl::pop_front]
}

// [tmpl::push_back]
assert_same<tmpl::push_back<List1<Type1>, Type2, Type3>,
            List1<Type1, Type2, Type3>>();
// [tmpl::push_back]

{
using lazy_test_arguments = tmpl::list<List1<Type1>, Type2, Type3>;
// [tmpl::push_front]
assert_same<tmpl::push_front<List1<Type1>, Type2, Type3>,
            List1<Type2, Type3, Type1>>();
HAS_LAZY_VERSION(push_front);
// [tmpl::push_front]
}

{
using lazy_test_arguments =
  tmpl::list<List1<Type1, Type2, Type1, Type3>, Type1>;
// [tmpl::remove]
assert_same<tmpl::remove<List1<Type1, Type2, Type1, Type3>, Type1>,
            List1<Type2, Type3>>();
HAS_LAZY_VERSION(remove);
// [tmpl::remove]
}

// [tmpl::remove_duplicates]
assert_same<tmpl::remove_duplicates<List1<Type1, Type2, Type1, Type3, Type2>>,
            List1<Type1, Type2, Type3>>();
// [tmpl::remove_duplicates]

{
using lazy_test_arguments = tmpl::list<List1<Type1, Type2, Type1, Type3>,
                                       std::is_same<Type1, tmpl::_1>>;
// [tmpl::remove_if]
assert_same<tmpl::remove_if<List1<Type1, Type2, Type1, Type3>,
                            std::is_same<Type1, tmpl::_1>>,
            List1<Type2, Type3>>();
HAS_LAZY_VERSION(remove_if);
// [tmpl::remove_if]
}

{
using lazy_test_arguments =
    tmpl::list<List1<Type1, Type2, Type1>, Type1, Type3>;
// [tmpl::replace]
assert_same<tmpl::replace<List1<Type1, Type2, Type1>, Type1, Type3>,
            List1<Type3, Type2, Type3>>();
HAS_LAZY_VERSION(replace);
// [tmpl::replace]
}

{
using lazy_test_arguments = tmpl::list<List1<Type1, Type2, Type1>,
                                       std::is_same<Type1, tmpl::_1>, Type3>;
// [tmpl::replace_if]
assert_same<tmpl::replace_if<List1<Type1, Type2, Type1>,
                             std::is_same<Type1, tmpl::_1>, Type3>,
            List1<Type3, Type2, Type3>>();
HAS_LAZY_VERSION(replace_if);
// [tmpl::replace_if]
}

{
using lazy_test_arguments = tmpl::list<List1<Type1, Type2, Type3, Type1>>;
// [tmpl::reverse]
assert_same<tmpl::reverse<List1<Type1, Type2, Type3, Type1>>,
            List1<Type1, Type3, Type2, Type1>>();
HAS_LAZY_VERSION(reverse);
// [tmpl::reverse]
}

{
using lazy_test_arguments =
  tmpl::list<List1<Type1, Type2, Type2, Type3>, std::is_same<tmpl::_1, Type2>>;
// [tmpl::reverse_find]
assert_same<
  tmpl::reverse_find<List1<Type1, Type2, Type2, Type3>,
                     std::is_same<tmpl::_1, Type2>>,
  List1<Type1, Type2, Type2>>();
assert_same<
  tmpl::reverse_find<List1<Type1, Type1, Type1, Type3>,
                     std::is_same<tmpl::_1, Type2>>,
  List1<>>();
assert_same<
  tmpl::reverse_find<List1<tmpl::size_t<0>, tmpl::size_t<1>, tmpl::size_t<2>>>,
  List1<tmpl::size_t<0>, tmpl::size_t<1>, tmpl::size_t<2>>>();
HAS_LAZY_VERSION(reverse_find);
// [tmpl::reverse_find]
}

{
using lazy_test_arguments =
  tmpl::list<List2<Type1, Type2>, Type3,
             lazy_make_list1<tmpl::_state, tmpl::_element>>;
// [tmpl::reverse_fold]
assert_same<tmpl::reverse_fold<List2<Type1, Type2>, Type3,
                               lazy_make_list1<tmpl::_state, tmpl::_element>>,
            List1<List1<Type3, Type2>, Type1>>();
HAS_LAZY_VERSION(reverse_fold);
// [tmpl::reverse_fold]
}

// [tmpl::sort]
assert_same<tmpl::sort<List1<tmpl::size_t<9>, tmpl::size_t<6>,
                             tmpl::size_t<7>, tmpl::size_t<0>>>,
            List1<tmpl::size_t<0>, tmpl::size_t<6>, tmpl::size_t<7>,
                  tmpl::size_t<9>>>();
assert_same<tmpl::sort<List1<Type2, Type3, Type3, Type1, Type2, Type3, Type2>,
                       CompareType123>,
            List1<Type1, Type2, Type2, Type2, Type3, Type3, Type3>>();
// [tmpl::sort]

// [tmpl::sort:equiv]
assert_same<tmpl::sort<List1<Type1, Type2, Type3>, std::false_type>,
            List1<Type3, Type2, Type1>>();
// [tmpl::sort:equiv]

{
using lazy_test_arguments =
  tmpl::list<List1<Type1, Type2, Type3, Type2, Type3, Type3, Type1>, Type3>;
// [tmpl::split]
assert_same<
  tmpl::split<List1<Type1, Type2, Type3, Type2, Type3, Type3, Type1>, Type3>,
  List1<List1<Type1, Type2>, List1<Type2>, List1<Type1>>>();
HAS_LAZY_VERSION(split);
// [tmpl::split]
}

{
using lazy_test_arguments =
  tmpl::list<List1<Type1, Type2, Type3>,
             tmpl::integral_constant<unsigned int, 2>>;
// [tmpl::split_at]
assert_same<tmpl::split_at<List1<Type1, Type2, Type3>,
                           tmpl::integral_constant<unsigned int, 2>>,
            List1<List1<Type1, Type2>, List1<Type3>>>();
HAS_LAZY_VERSION(split_at);
// [tmpl::split_at]
}

{
using lazy_test_arguments =
  tmpl::list<List2<Type1, Type2, Type3>, List3<Type3, Type2, Type1>,
             lazy_make_list1<tmpl::_1, tmpl::_2>>;
// [tmpl::transform]
assert_same<
  tmpl::transform<List2<Type1, Type2, Type3>, List3<Type3, Type2, Type1>,
                  lazy_make_list1<tmpl::_1, tmpl::_2>>,
  List2<List1<Type1, Type3>, List1<Type2, Type2>, List1<Type3, Type1>>>();
HAS_LAZY_VERSION(transform);
// [tmpl::transform]
}
}
}  // namespace list_from_list

namespace map_functions {
// [example_map]
using example_map =
  tmpl::map<tmpl::pair<Type1, int>, tmpl::pair<Type2, double>>;
// [example_map]

void run() {
// [tmpl::at:map]
assert_same<tmpl::at<example_map, Type1>, int>();
assert_same<tmpl::at<example_map, Type3>, tmpl::no_such_type_>();
// [tmpl::at:map]

// [tmpl::erase:map]
assert_maps_same<tmpl::erase<example_map, Type1>,
                 tmpl::map<tmpl::pair<Type2, double>>>();
assert_maps_same<tmpl::erase<example_map, Type3>, example_map>();
// [tmpl::erase:map]

// [tmpl::has_key:map]
assert_same<tmpl::has_key<example_map, Type2>, tmpl::true_type>();
assert_same<tmpl::has_key<example_map, Type3>, tmpl::false_type>();
// [tmpl::has_key:map]

// [tmpl::insert:map]
assert_maps_same<tmpl::insert<example_map, tmpl::pair<Type3, int>>,
                 tmpl::map<tmpl::pair<Type1, int>, tmpl::pair<Type2, double>,
                           tmpl::pair<Type3, int>>>();
assert_maps_same<tmpl::insert<example_map, tmpl::pair<Type1, float>>,
                 example_map>();
// [tmpl::insert:map]

// [tmpl::keys_as_sequence]
assert_sets_same<tmpl::keys_as_sequence<example_map>,
                 tmpl::set<Type1, Type2>>();
static_assert(std::is_same_v<tmpl::keys_as_sequence<example_map, List1>,
                             List1<Type1, Type2>> or
              std::is_same_v<tmpl::keys_as_sequence<example_map, List1>,
                             List1<Type2, Type1>>);
// [tmpl::keys_as_sequence]

{
using lazy_test_arguments = tmpl::list<example_map, Type1>;
// [tmpl::lookup]
assert_same<tmpl::lookup<example_map, Type1>, int>();
assert_same<tmpl::lookup<example_map, Type3>, tmpl::no_such_type_>();
HAS_LAZY_VERSION(lookup);
// [tmpl::lookup]
}

// [tmpl::lookup_at]
assert_same<tmpl::lazy::lookup_at<example_map, Type1>::type,
            tmpl::type_<int>>();
assert_same<tmpl::lazy::lookup_at<example_map, Type3>::type,
            tmpl::type_<tmpl::no_such_type_>>();
// [tmpl::lookup_at]

// [tmpl::values_as_sequence]
static_assert(std::is_same_v<tmpl::values_as_sequence<example_map>,
                             tmpl::list<int, double>> or
              std::is_same_v<tmpl::values_as_sequence<example_map>,
                             tmpl::list<double, int>>);
static_assert(std::is_same_v<tmpl::values_as_sequence<example_map, List1>,
                             List1<int, double>> or
              std::is_same_v<tmpl::values_as_sequence<example_map, List1>,
                             List1<double, int>>);
// [tmpl::values_as_sequence]
}
}  // namespace map_functions

namespace set_functions {
void run() {
// [tmpl::contains]
assert_same<tmpl::contains<tmpl::set<Type1, Type2>, Type1>, tmpl::true_type>();
assert_same<tmpl::contains<tmpl::set<Type1, Type2>, Type3>, tmpl::false_type>();
// [tmpl::contains]

// [tmpl::erase:set]
assert_sets_same<tmpl::erase<tmpl::set<Type1, Type2>, Type1>,
                 tmpl::set<Type2>>();
assert_sets_same<tmpl::erase<tmpl::set<Type1, Type2>, Type3>,
                 tmpl::set<Type1, Type2>>();
// [tmpl::erase:set]

// [tmpl::has_key:set]
assert_same<tmpl::has_key<tmpl::set<Type1, Type2>, Type2>, tmpl::true_type>();
assert_same<tmpl::has_key<tmpl::set<Type1, Type2>, Type3>, tmpl::false_type>();
// [tmpl::has_key:set]

// [tmpl::insert:set]
assert_sets_same<tmpl::insert<tmpl::set<Type1, Type2>, Type3>,
                 tmpl::set<Type1, Type2, Type3>>();
assert_sets_same<tmpl::insert<tmpl::set<Type1, Type2>, Type1>,
                 tmpl::set<Type1, Type2>>();
// [tmpl::insert:set]
}
}  // namespace set_functions

namespace math_functions {
void run() {
// [math_arithmetic]
assert_same<tmpl::plus<tmpl::size_t<10>, tmpl::size_t<3>>::type,
            tmpl::size_t<13>>();
assert_same<tmpl::minus<tmpl::size_t<10>, tmpl::size_t<3>>::type,
            tmpl::size_t<7>>();
assert_same<tmpl::times<tmpl::size_t<10>, tmpl::size_t<3>>::type,
            tmpl::size_t<30>>();
assert_same<tmpl::divides<tmpl::size_t<10>, tmpl::size_t<3>>::type,
            tmpl::size_t<3>>();
assert_same<tmpl::modulo<tmpl::size_t<10>, tmpl::size_t<3>>::type,
            tmpl::size_t<1>>();
assert_same<tmpl::negate<tmpl::int64_t<10>>::type, tmpl::int64_t<-10>>();
// [math_arithmetic]

// [math_bitwise]
assert_same<tmpl::complement<tmpl::uint8_t<0b10001111>>::type,
                             tmpl::uint8_t<0b01110000>>();
assert_same<tmpl::bitand_<tmpl::uint8_t<0b00111011>,
                          tmpl::uint8_t<0b01010110>>::type,
                          tmpl::uint8_t<0b00010010>>();
assert_same<tmpl::bitor_<tmpl::uint8_t<0b01100011>,
                         tmpl::uint8_t<0b10100111>>::type,
                         tmpl::uint8_t<0b11100111>>();
assert_same<tmpl::bitxor_<tmpl::uint8_t<0b11000011>,
                          tmpl::uint8_t<0b00000110>>::type,
                          tmpl::uint8_t<0b11000101>>();
assert_same<tmpl::shift_left<tmpl::uint8_t<0b00001110>, tmpl::size_t<3>>::type,
                             tmpl::uint8_t<0b01110000>>();
assert_same<tmpl::shift_right<tmpl::uint8_t<0b10110011>, tmpl::size_t<4>>::type,
                              tmpl::uint8_t<0b00001011>>();
// [math_bitwise]

// [math_comparison]
assert_same<tmpl::equal_to<tmpl::size_t<1>, tmpl::size_t<2>>::type,
            tmpl::false_type>();
assert_same<tmpl::not_equal_to<tmpl::size_t<1>, tmpl::size_t<2>>::type,
            tmpl::true_type>();
assert_same<tmpl::greater<tmpl::size_t<1>, tmpl::size_t<2>>::type,
            tmpl::false_type>();
assert_same<tmpl::greater_equal<tmpl::size_t<1>, tmpl::size_t<2>>::type,
            tmpl::false_type>();
assert_same<tmpl::less<tmpl::size_t<1>, tmpl::size_t<2>>::type,
            tmpl::true_type>();
assert_same<tmpl::less_equal<tmpl::size_t<1>, tmpl::size_t<2>>::type,
            tmpl::true_type>();
// [math_comparison]

// [math_logical]
assert_same<tmpl::and_<>::type, tmpl::true_type>();
assert_same<tmpl::and_<std::true_type, std::false_type>::type,
            tmpl::false_type>();
assert_same<tmpl::or_<>::type, tmpl::false_type>();
assert_same<tmpl::or_<std::true_type, std::false_type, std::false_type>::type,
            tmpl::true_type>();
assert_same<tmpl::xor_<std::true_type, std::false_type>::type,
            tmpl::true_type>();
assert_same<tmpl::not_<std::true_type>::type, tmpl::false_type>();
// [math_logical]

// [tmpl::identity]
assert_same<tmpl::identity<tmpl::size_t<10>>::type, tmpl::size_t<10>>();
// [tmpl::identity]

// [tmpl::max]
assert_same<tmpl::max<tmpl::size_t<10>, tmpl::int32_t<3>>::type,
            tmpl::size_t<10>>();
// [tmpl::max]

// [tmpl::min]
assert_same<tmpl::min<tmpl::size_t<10>, tmpl::int32_t<3>>::type,
            tmpl::size_t<3>>();
// [tmpl::min]

// [tmpl::next]
assert_same<tmpl::next<tmpl::size_t<10>>::type, tmpl::size_t<11>>();
// [tmpl::next]

// [tmpl::prev]
assert_same<tmpl::prev<tmpl::size_t<10>>::type, tmpl::size_t<9>>();
// [tmpl::prev]
}
}  // namespace math_functions

namespace miscellaneous {
// [tmpl::has_type:pack_expansion]
template <typename... T>
bool check_sizes(const T&... containers,
                 const typename tmpl::has_type<T, size_t>::type... sizes) {
  return (... and (containers.size() == sizes));
}
// [tmpl::has_type:pack_expansion]

// [tmpl::inherit:pack:definitions]
template <typename... T>
struct inherit_pack {
  struct type : T... {};
};
// [tmpl::inherit:pack:definitions]

void run() {
// [tmpl::always]
assert_same<tmpl::always<Type1>::type, Type1>();
// [tmpl::always]

// [tmpl::apply]
assert_same<tmpl::apply<std::is_convertible<tmpl::_1, tmpl::_2>,
                        const char*, std::string>,
            std::true_type>();
assert_same<tmpl::apply<std::is_convertible<tmpl::_2, tmpl::_1>,
                        const char*, std::string>,
            std::false_type>();
// [tmpl::apply]

// [tmpl::count]
assert_same<tmpl::count<Type1, Type2, Type1>,
            tmpl::integral_constant<unsigned int, 3>>();
// [tmpl::count]

// [tmpl::conditional_t]
assert_same<tmpl::conditional_t<true, Type1, Type2>, Type1>();
assert_same<tmpl::conditional_t<false, Type1, Type2>, Type2>();
// [tmpl::conditional_t]

// [tmpl::eval_if]
assert_same<tmpl::eval_if<std::true_type,
                          tmpl::plus<tmpl::size_t<1>, tmpl::size_t<2>>,
                          tmpl::plus<Type1, Type2>  // Invalid expression
                          >::type,
            tmpl::size_t<3>>();
// [tmpl::eval_if]

// [tmpl::eval_if_c]
assert_same<tmpl::eval_if_c<true,
                            tmpl::plus<tmpl::size_t<1>, tmpl::size_t<2>>,
                            tmpl::plus<Type1, Type2>  // Invalid expression
                            >::type,
            tmpl::size_t<3>>();
// [tmpl::eval_if_c]

// [tmpl::has_type]
assert_same<tmpl::has_type<Type1, Type2>::type, Type2>();
assert_same<tmpl::has_type<Type1>::type, void>();
// [tmpl::has_type]

// [tmpl::has_type:pack_expansion:asserts]
CHECK(check_sizes<std::string, std::vector<int>>("Hello", {1, 2, 3}, 5, 3));
// [tmpl::has_type:pack_expansion:asserts]

// [tmpl::if_]
assert_same<tmpl::if_<std::true_type, Type1, Type2>::type, Type1>();
// [tmpl::if_]

// [tmpl::if_c]
assert_same<tmpl::if_c<true, Type1, Type2>::type, Type1>();
// [tmpl::if_c]

// [tmpl::inherit]
// tmpl::type_ is used in this example because base classes must be
// complete types
static_assert(
    std::is_base_of_v<tmpl::type_<Type2>,
                      tmpl::inherit<tmpl::type_<Type1>, tmpl::type_<Type2>,
                                    tmpl::type_<Type3>>::type>);
// [tmpl::inherit]

// [tmpl::inherit:pack:asserts]
static_assert(
    std::is_base_of_v<tmpl::type_<Type2>,
                      inherit_pack<tmpl::type_<Type1>, tmpl::type_<Type2>,
                                   tmpl::type_<Type3>>::type>);
// [tmpl::inherit:pack:asserts]

{
using lazy_test_arguments =
  tmpl::list<List1<Type1, Type2>, List2<tmpl::_1, tmpl::_2, Type3>>;
// [tmpl::inherit_linearly]
assert_same<tmpl::inherit_linearly<List1<Type1, Type2>,
                                   List2<tmpl::_1, tmpl::_2, Type3>>,
            List2<List2<tmpl::empty_base, Type1, Type3>, Type2, Type3>>();
assert_same<tmpl::inherit_linearly<List1<Type1, Type2>,
                                   List2<tmpl::_1, tmpl::_2, Type3>, Type3>,
            List2<List2<Type3, Type1, Type3>, Type2, Type3>>();
HAS_LAZY_VERSION(inherit_linearly);
// [tmpl::inherit_linearly]
}

// [tmpl::is_set]
assert_same<tmpl::is_set<Type1, Type2, Type3>, tmpl::true_type>();
assert_same<tmpl::is_set<Type1, Type2, Type1>, tmpl::false_type>();
assert_same<tmpl::is_set<>, tmpl::true_type>();
// [tmpl::is_set]

{
// [tmpl::real_]
using three_eighths = tmpl::single_<0x3EC00000>;
using minus_one_hundred_thousand_three = tmpl::double_<0xC0F86A3000000000>;
CHECK(static_cast<float>(three_eighths{}) == 0.375f);
CHECK(static_cast<double>(minus_one_hundred_thousand_three{}) == -100003.0);
assert_same<three_eighths::value_type, float>();
assert_same<minus_one_hundred_thousand_three::value_type, double>();
// [tmpl::real_]
}

// [tmpl::repeat]
assert_same<tmpl::repeat<Wrapper, tmpl::size_t<3>, Type1>,
            Wrapper<Wrapper<Wrapper<Type1>>>>();
assert_same<tmpl::repeat<Wrapper, tmpl::size_t<0>, Type1>, Type1>();
// [tmpl::repeat]

// [tmpl::repeat:lazy]
assert_same<tmpl::lazy::repeat<Wrapper, tmpl::size_t<3>, Type1>::type,
            Wrapper<Wrapper<Wrapper<Type1>>>>();
// [tmpl::repeat:lazy]

// [tmpl::sizeof_]
assert_same<tmpl::sizeof_<double>::type,
            tmpl::integral_constant<unsigned int, sizeof(double)>>();
// [tmpl::sizeof_]

// [tmpl::substitute]
assert_same<tmpl::substitute<List1<List2<tmpl::_1, tmpl::_2, tmpl::_3>,
                                   tmpl::args<0>, tmpl::args<1>, tmpl::args<2>>,
                             List3<Type1, Type2, Type3>>,
            List1<List2<tmpl::_1, tmpl::_2, Type3>, Type1, Type2, Type3>>();
// [tmpl::substitute]

// [tmpl::type_from]
assert_same<tmpl::type_from<tmpl::type_<Type1>>, Type1>();
// [tmpl::type_from]

// [tmpl::wrap]
assert_same<tmpl::wrap<List1<Type1, Type2>, List2>, List2<Type1, Type2>>();
// [tmpl::wrap]

// [tmpl::wrap:lazy]
assert_same<tmpl::lazy::wrap<List1<Type1, Type2>, List2>::type,
            List2<Type1, Type2>>();
// [tmpl::wrap:lazy]
}
}  // namespace miscellaneous

namespace runtime {
// [runtime_declarations]
template <typename T>
struct NonCopyable {
  NonCopyable(T t = T{}) : value(std::move(t)) {}
  NonCopyable(const NonCopyable&) = delete;
  NonCopyable(NonCopyable&&) = default;
  NonCopyable& operator=(const NonCopyable&) = delete;
  NonCopyable& operator=(NonCopyable&&) = default;

  T value;

  template <typename... Args>
  decltype(auto) operator()(Args&&... args) {
    return value(std::forward<Args>(args)...);
  }
};
// [runtime_declarations]

namespace example_for_each_args {
// [tmpl::for_each_args:defs]
struct Functor {
  Functor() = default;
  Functor(const Functor&) = delete;
  Functor(Functor&&) = default;
  Functor& operator=(const Functor&) = delete;
  Functor& operator=(Functor&&) = default;

  std::vector<double> record;

  void operator()(NonCopyable<int> x) {
    record.push_back(x.value);
  }

  void operator()(const NonCopyable<double>& x) {
    record.push_back(x.value);
  }
};
// [tmpl::for_each_args:defs]
}  // namespace example_for_each_args

namespace example_for_each {
// [tmpl::for_each:defs]
struct Functor {
  Functor() = default;
  Functor(const Functor&) = delete;
  Functor(Functor&&) = default;
  Functor& operator=(const Functor&) = delete;
  Functor& operator=(Functor&&) = default;

  std::vector<std::string> record;

  template <typename T>
  void operator()(T /*t*/) {
    using type = tmpl::type_from<T>;
    if (std::is_same_v<type, int>) {
      record.push_back("int");
    } else if (std::is_same_v<type, double>) {
      record.push_back("double");
    }
  }
};
// [tmpl::for_each:defs]
}  // namespace example_for_each

void run() {
{
using example_for_each_args::Functor;
// [tmpl::for_each_args]
const NonCopyable<double> three_point_five{3.5};
CHECK(tmpl::for_each_args(Functor{}, NonCopyable<int>{2}, three_point_five)
          .record == std::vector<double>{2.0, 3.5});
// [tmpl::for_each_args]
}

{
using example_for_each::Functor;
// [tmpl::for_each]
CHECK(tmpl::for_each<List1<int, double, int>>(Functor{}).record ==
        std::vector<std::string>{"int", "double", "int"});
// [tmpl::for_each]
}

{
// [tmpl::select]
const NonCopyable<std::string> hi{"Hi"};
CHECK(tmpl::select<std::true_type>(NonCopyable<int>{3}, hi).value == 3);
CHECK(tmpl::select<std::false_type>(NonCopyable<int>{3}, hi).value == "Hi");
// [tmpl::select]
}
}
}  // namespace runtime

namespace external {
void run() {
// [boost_integration]
assert_same<tmpl::as_fusion_deque<List1<Type1, Type2, Type3>>,
            boost::fusion::deque<Type1, Type2, Type3>>();
assert_same<tmpl::as_fusion_list<List1<Type1, Type2, Type3>>,
            boost::fusion::list<Type1, Type2, Type3>>();
assert_same<tmpl::as_fusion_set<List1<Type1, Type2, Type3>>,
            boost::fusion::set<Type1, Type2, Type3>>();
assert_same<tmpl::as_fusion_vector<List1<Type1, Type2, Type3>>,
            boost::fusion::vector<Type1, Type2, Type3>>();
assert_same<tmpl::as_variant<List1<Type1, Type2, Type3>>,
            boost::variant<Type1, Type2, Type3>>();

assert_same<tmpl::fusion_deque_wrapper<Type1, Type2, Type3>,
            boost::fusion::deque<Type1, Type2, Type3>>();
assert_same<tmpl::fusion_list_wrapper<Type1, Type2, Type3>,
            boost::fusion::list<Type1, Type2, Type3>>();
assert_same<tmpl::fusion_set_wrapper<Type1, Type2, Type3>,
            boost::fusion::set<Type1, Type2, Type3>>();
assert_same<tmpl::fusion_vector_wrapper<Type1, Type2, Type3>,
            boost::fusion::vector<Type1, Type2, Type3>>();
assert_same<tmpl::variant_wrapper<Type1, Type2, Type3>,
            boost::variant<Type1, Type2, Type3>>();
// [boost_integration]

// [stl_integration]
assert_same<tmpl::as_pair<List1<Type1, Type2>>, std::pair<Type1, Type2>>();
assert_same<tmpl::as_tuple<List1<Type1, Type2, Type3>>,
            std::tuple<Type1, Type2, Type3>>();

assert_same<tmpl::pair_wrapper<Type1, Type2>, std::pair<Type1, Type2>>();
assert_same<tmpl::tuple_wrapper<Type1, Type2, Type3>,
            std::tuple<Type1, Type2, Type3>>();

assert_same<tmpl::pair_wrapper_<Type1, Type2>::type, std::pair<Type1, Type2>>();
// [stl_integration]

// [tmpl::make_integral]
assert_same<tmpl::make_integral<std::integral_constant<char, 3>>::type,
            tmpl::integral_constant<char, 3>>();
assert_same<tmpl::as_integral_list<List1<std::true_type, std::true_type,
                                         std::false_type>>,
            List1<tmpl::true_type, tmpl::true_type, tmpl::false_type>>();
// [tmpl::make_integral]

// [tmpl::as_list]
assert_same<tmpl::as_sequence<std::pair<Type1, Type2>, List1>,
            List1<Type1, Type2>>();
assert_same<tmpl::as_list<std::pair<Type1, Type2>>, tmpl::list<Type1, Type2>>();
// [tmpl::as_list]

// [tmpl::as_set]
assert_same<tmpl::as_set<std::tuple<Type1, Type2, Type3>>,
            tmpl::set<Type1, Type2, Type3>>();
assert_same<tmpl::set_wrapper<Type1, Type2, Type3>,
            tmpl::set<Type1, Type2, Type3>>();
// [tmpl::as_set]
}
}  // namespace external
}  // namespace
// clang-format on

SPECTRE_TEST_CASE("Unit.Utilities.TMPL.Documentation", "[Unit][Utilities]") {
  metafunctions::run();
  containers::run();
  constants::run();
  list_constructors::run();
  list_query::run();
  list_from_list::run();
  map_functions::run();
  set_functions::run();
  math_functions::run();
  miscellaneous::run();
  runtime::run();
  external::run();
}
