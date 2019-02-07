// Distributed under the MIT License.
// See LICENSE.txt for details.

#include <array>
#include <complex>
#include <cstddef>
#include <deque>
#include <forward_list>
#include <functional>
#include <future>
#include <iosfwd>
#include <list>
#include <map>
#include <memory>
#include <queue>
#include <set>
#include <stack>
#include <string>
#include <tuple>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include "Utilities/TMPL.hpp"
#include "Utilities/TypeTraits.hpp"
#include "tests/Unit/TestHelpers.hpp"  // IWYU pragma: keep

/// \cond
class DataVector;
/// \endcond

namespace {

/// \cond HIDDEN_SYMBOLS
class A {
 public:
  int operator()() { return 1; }
  A() = delete;
  std::unique_ptr<double> get_clone() const {
    return std::make_unique<double>();
  }
  int size() const { return 2; }
};
}  // namespace

class BClassInTestTypeTraits {
 public:
  double operator()(const int /* unused */, const double /* unused */) {
    return 0.;
  }
  std::unique_ptr<double> clone() const { return std::make_unique<double>(); }
};

class CClassInTestTypeTraits {};

class D;

namespace std {
template <>
struct hash<A> {
  size_t operator()(const A& /* a */) const { return 1; }
};
}  // namespace std

std::ostream& operator<<(std::ostream& os, const D&);

bool operator==(const CClassInTestTypeTraits&, const CClassInTestTypeTraits&);

bool operator!=(const CClassInTestTypeTraits&, const CClassInTestTypeTraits&);
/// \endcond

/// [conjunction_example]
static_assert(
    cpp17::conjunction<std::true_type, std::true_type, std::true_type>::value,
    "Failed testing type trait conjunction");
static_assert(not cpp17::conjunction<std::true_type, std::false_type,
                                     std::true_type>::value,
              "Failed testing type trait conjunction");
static_assert(not cpp17::conjunction<std::true_type, std::false_type>::value,
              "Failed testing type trait conjunction");
static_assert(not cpp17::conjunction<std::false_type, std::true_type>::value,
              "Failed testing type trait conjunction");
/// [conjunction_example]

/// [disjunction_example]
static_assert(not cpp17::disjunction<std::false_type, std::false_type,
                                     std::false_type>::value,
              "Failed testing type trait disjunction");
static_assert(
    cpp17::disjunction<std::false_type, std::true_type, std::false_type>::value,
    "Failed testing type trait disjunction");
static_assert(cpp17::disjunction<std::true_type, std::false_type>::value,
              "Failed testing type trait disjunction");
static_assert(cpp17::disjunction<std::false_type, std::true_type>::value,
              "Failed testing type trait disjunction");
/// [disjunction_example]

/// [negation_example]
static_assert(
    std::is_same<
        cpp17::bool_constant<false>,
        typename cpp17::negation<cpp17::bool_constant<true>>::type>::value,
    "Failed testing type trait negate");
static_assert(
    std::is_same<
        cpp17::bool_constant<true>,
        typename cpp17::negation<cpp17::bool_constant<false>>::type>::value,
    "Failed testing type trait negate");
static_assert(not cpp17::negation<cpp17::bool_constant<true>>::value,
              "Failed testing type trait negate");
static_assert(cpp17::negation<cpp17::bool_constant<false>>::value,
              "Failed testing type trait negate");
/// [negation_example]

/// [void_t_example]
static_assert(std::is_same<cpp17::void_t<char, bool, double>, void>::value,
              "Failed testing type trait void_t");
/// [void_t_example]

static_assert(tt::can_be_copy_constructed_v<int>,
              "Failed testing type trait is_copy_constructible");
static_assert(tt::can_be_copy_constructed_v<std::vector<int>>,
              "Failed testing type trait is_copy_constructible");
static_assert(tt::can_be_copy_constructed_v<std::unordered_map<int, int>>,
              "Failed testing type trait is_copy_constructible");
static_assert(
    not tt::can_be_copy_constructed_v<std::unordered_map<int, NonCopyable>>,
    "Failed testing type trait is_copy_constructible");

/// [is_std_array_example]
static_assert(tt::is_std_array<std::array<double, 3>>::value,
              "Failed testing type trait is_std_array");
static_assert(tt::is_std_array_t<std::array<double, 3>>::value,
              "Failed testing type trait is_std_array");
static_assert(tt::is_std_array_v<std::array<double, 3>>,
              "Failed testing type trait is_std_array");
static_assert(not tt::is_std_array<double>::value,
              "Failed testing type trait is_std_array");
static_assert(tt::is_std_array<std::array<D, 10>>::value,
              "Failed testing type trait is_std_array");
/// [is_std_array_example]

/// [is_std_array_of_size_example]
static_assert(tt::is_std_array_of_size<3, std::array<double, 3>>::value,
              "Failed testing type trait is_std_array_of_size");
static_assert(tt::is_std_array_of_size_t<3, std::array<double, 3>>::value,
              "Failed testing type trait is_std_array_of_size");
static_assert(tt::is_std_array_of_size_v<3, std::array<double, 3>>,
              "Failed testing type trait is_std_array_of_size");
static_assert(not tt::is_std_array_of_size<3, double>::value,
              "Failed testing type trait is_std_array_of_size");
static_assert(not tt::is_std_array_of_size<2, std::array<double, 3>>::value,
              "Failed testing type trait is_std_array_of_size");
static_assert(tt::is_std_array_of_size<10, std::array<D, 10>>::value,
              "Failed testing type trait is_std_array_of_size");
/// [is_std_array_of_size_example]

/// [is_a_example]
static_assert(tt::is_a<std::vector, std::vector<double>>::value,
              "Failed testing type trait is_a<vector>");
static_assert(not tt::is_a_t<std::vector, double>::value,
              "Failed testing type trait is_a<vector>");
static_assert(tt::is_a_v<std::vector, std::vector<D>>,
              "Failed testing type trait is_a<vector>");

static_assert(tt::is_a<std::deque, std::deque<double>>::value,
              "Failed testing type trait is_a<deque>");
static_assert(not tt::is_a<std::deque, double>::value,
              "Failed testing type trait is_a<deque>");
static_assert(tt::is_a<std::deque, std::deque<D>>::value,
              "Failed testing type trait is_a<deque>");

static_assert(tt::is_a<std::forward_list, std::forward_list<double>>::value,
              "Failed testing type trait is_a<forward_list>");
static_assert(not tt::is_a<std::forward_list, double>::value,
              "Failed testing type trait is_a<forward_list>");
static_assert(tt::is_a<std::forward_list, std::forward_list<D>>::value,
              "Failed testing type trait is_a<forward_list>");

static_assert(tt::is_a<std::list, std::list<double>>::value,
              "Failed testing type trait is_a<list>");
static_assert(not tt::is_a<std::list, double>::value,
              "Failed testing type trait is_a<list>");
static_assert(tt::is_a<std::list, std::list<D>>::value,
              "Failed testing type trait is_a<list>");

static_assert(tt::is_a<std::map, std::map<std::string, double>>::value,
              "Failed testing type trait is_a<map>");
static_assert(not tt::is_a<std::map, double>::value,
              "Failed testing type trait is_a<map>");
static_assert(tt::is_a<std::map, std::map<std::string, D>>::value,
              "Failed testing type trait is_a<map>");

static_assert(tt::is_a<std::unordered_map,
                       std::unordered_map<std::string, double>>::value,
              "Failed testing type trait is_a<unordered_map>");
static_assert(not tt::is_a<std::unordered_map, double>::value,
              "Failed testing type trait is_a<unordered_map>");
static_assert(
    tt::is_a<std::unordered_map, std::unordered_map<std::string, D>>::value,
    "Failed testing type trait is_a<unordered_map>");

static_assert(tt::is_a<std::set, std::set<double>>::value,
              "Failed testing type trait is_a<set>");
static_assert(not tt::is_a<std::set, double>::value,
              "Failed testing type trait is_a<set>");
static_assert(tt::is_a<std::set, std::set<D>>::value,
              "Failed testing type trait is_a<set>");

static_assert(tt::is_a<std::unordered_set, std::unordered_set<double>>::value,
              "Failed testing type trait is_a<unordered_set>");
static_assert(not tt::is_a<std::unordered_set, double>::value,
              "Failed testing type trait is_a<unordered_set>");
static_assert(tt::is_a<std::unordered_set, std::unordered_set<D>>::value,
              "Failed testing type trait is_a<unordered_set>");

static_assert(tt::is_a<std::multiset, std::multiset<double>>::value,
              "Failed testing type trait is_a<multiset>");
static_assert(not tt::is_a<std::multiset, double>::value,
              "Failed testing type trait is_a<multiset>");
static_assert(tt::is_a<std::multiset, std::multiset<D>>::value,
              "Failed testing type trait is_a<multiset>");

static_assert(
    tt::is_a<std::unordered_multiset, std::unordered_multiset<double>>::value,
    "Failed testing type trait is_a<unordered_multiset>");
static_assert(not tt::is_a<std::unordered_multiset, double>::value,
              "Failed testing type trait is_a<unordered_multiset>");
static_assert(
    tt::is_a<std::unordered_multiset, std::unordered_multiset<D>>::value,
    "Failed testing type trait is_a<unordered_multiset>");

static_assert(
    tt::is_a<std::multimap, std::multimap<std::string, double>>::value,
    "Failed testing type trait is_a<multimap>");
static_assert(not tt::is_a<std::multimap, double>::value,
              "Failed testing type trait is_a<multimap>");
static_assert(tt::is_a<std::multimap, std::multimap<std::string, D>>::value,
              "Failed testing type trait is_a<multimap>");

static_assert(tt::is_a<std::unordered_multimap,
                       std::unordered_multimap<std::string, double>>::value,
              "Failed testing type trait is_a<unordered_multimap>");
static_assert(not tt::is_a<std::unordered_multimap, double>::value,
              "Failed testing type trait is_a<unordered_multimap>");
static_assert(tt::is_a<std::unordered_multimap,
                       std::unordered_multimap<std::string, D>>::value,
              "Failed testing type trait is_a<unordered_multimap>");

static_assert(tt::is_a<std::priority_queue, std::priority_queue<double>>::value,
              "Failed testing type trait is_a<priority_queue>");
static_assert(not tt::is_a<std::priority_queue, double>::value,
              "Failed testing type trait is_a<priority_queue>");
static_assert(tt::is_a<std::priority_queue, std::priority_queue<D>>::value,
              "Failed testing type trait is_a<priority_queue>");

static_assert(tt::is_a<std::queue, std::queue<double>>::value,
              "Failed testing type trait is_a<queue>");
static_assert(not tt::is_a<std::queue, double>::value,
              "Failed testing type trait is_a<queue>");
static_assert(tt::is_a<std::queue, std::queue<D>>::value,
              "Failed testing type trait is_a<queue>");

static_assert(tt::is_a<std::stack, std::stack<double>>::value,
              "Failed testing type trait is_a<stack>");
static_assert(not tt::is_a<std::stack, double>::value,
              "Failed testing type trait is_a<stack>");
static_assert(tt::is_a<std::stack, std::stack<D>>::value,
              "Failed testing type trait is_a<stack>");

static_assert(tt::is_a<std::unique_ptr, std::unique_ptr<double>>::value,
              "Failed testing type trait is_a<unique_ptr>");
static_assert(
    tt::is_a<std::unique_ptr, std::unique_ptr<CClassInTestTypeTraits>>::value,
    "Failed testing type trait is_a<unique_ptr>");
static_assert(not tt::is_a<std::unique_ptr, std::shared_ptr<double>>::value,
              "Failed testing type trait is_a<unique_ptr>");
static_assert(not tt::is_a<std::unique_ptr, CClassInTestTypeTraits>::value,
              "Failed testing type trait is_a<unique_ptr>");

static_assert(tt::is_a<std::shared_ptr, std::shared_ptr<double>>::value,
              "Failed testing type trait is_a<shared_ptr>");
static_assert(
    tt::is_a<std::shared_ptr, std::shared_ptr<CClassInTestTypeTraits>>::value,
    "Failed testing type trait is_a<shared_ptr>");
static_assert(not tt::is_a<std::shared_ptr, std::unique_ptr<double>>::value,
              "Failed testing type trait is_a<shared_ptr>");
static_assert(not tt::is_a<std::shared_future, CClassInTestTypeTraits>::value,
              "Failed testing type trait is_a<shared_ptr>");

static_assert(tt::is_a<std::weak_ptr, std::weak_ptr<double>>::value,
              "Failed testing type trait is_a<weak_ptr>");
static_assert(
    tt::is_a<std::weak_ptr, std::weak_ptr<CClassInTestTypeTraits>>::value,
    "Failed testing type trait is_a<weak_ptr>");
static_assert(not tt::is_a<std::weak_ptr, std::unique_ptr<double>>::value,
              "Failed testing type trait is_a<weak_ptr>");
static_assert(not tt::is_a<std::weak_ptr, CClassInTestTypeTraits>::value,
              "Failed testing type trait is_a<weak_ptr>");

static_assert(tt::is_a<std::tuple, std::tuple<int, double, A>>::value,
              "Failed testing type trait is_a");
static_assert(tt::is_a<std::vector, std::vector<A>>::value,
              "Failed testing type trait is_a");

static_assert(tt::is_a<std::future, std::future<double>>::value,
              "Failed testing type trait is_a<future>");
static_assert(tt::is_a<std::future, std::future<std::vector<double>>>::value,
              "Failed testing type trait is_a<future>");
static_assert(not tt::is_a<std::future, std::shared_future<double>>::value,
              "Failed testing type trait is_a<future>");

static_assert(tt::is_a<std::shared_future, std::shared_future<double>>::value,
              "Failed testing type trait is_a<shared_future>");
static_assert(tt::is_a<std::shared_future,
                       std::shared_future<std::vector<double>>>::value,
              "Failed testing type trait is_a<shared_future>");
static_assert(not tt::is_a<std::shared_future, std::future<double>>::value,
              "Failed testing type trait is_a<shared_future>");
/// [is_a_example]

/// [is_iterable_example]
static_assert(tt::is_iterable<std::vector<double>>::value,
              "Failed testing type trait is_iterable");
static_assert(tt::is_iterable_t<std::vector<double>>::value,
              "Failed testing type trait is_iterable");
static_assert(tt::is_iterable_v<std::vector<double>>,
              "Failed testing type trait is_iterable");
static_assert(not tt::is_iterable<double>::value,
              "Failed testing type trait is_iterable");
/// [is_iterable_example]

/// [is_comparable_example]
static_assert(tt::is_comparable<std::vector<double>>::value,
              "Failed testing type trait is_comparable");
static_assert(tt::is_comparable_t<std::vector<double>>::value,
              "Failed testing type trait is_comparable");
static_assert(tt::is_comparable_v<std::vector<double>>,
              "Failed testing type trait is_comparable");
static_assert(tt::is_comparable<double>::value,
              "Failed testing type trait is_comparable");
static_assert(not tt::is_comparable<A>::value,
              "Failed testing type trait is_comparable");
/// [is_comparable_example]

/// [array_size_example]
static_assert(tt::array_size<std::array<double, 3>>::value == 3,
              "Failed testing type trait array_size");
static_assert(tt::array_size<std::array<A, 4>>::value == 4,
              "Failed testing type trait array_size");
/// [array_size_example]

/// [has_equivalence_example]
static_assert(not tt::has_equivalence<A>::value,
              "Failed testing type trait has_equivalence");
static_assert(tt::has_equivalence<CClassInTestTypeTraits>::value,
              "Failed testing type trait has_equivalence");
static_assert(tt::has_equivalence_t<CClassInTestTypeTraits>::value,
              "Failed testing type trait has_equivalence");
static_assert(tt::has_equivalence_v<CClassInTestTypeTraits>,
              "Failed testing type trait has_equivalence");
/// [has_equivalence_example]

/// [has_inequivalence_example]
static_assert(not tt::has_inequivalence<A>::value,
              "Failed testing type trait has_inequivalence");
static_assert(tt::has_inequivalence<CClassInTestTypeTraits>::value,
              "Failed testing type trait has_inequivalence");
static_assert(tt::has_inequivalence_t<CClassInTestTypeTraits>::value,
              "Failed testing type trait has_inequivalence");
static_assert(tt::has_inequivalence_v<CClassInTestTypeTraits>,
              "Failed testing type trait has_inequivalence");
/// [has_inequivalence_example]

/// [is_callable_example]
static_assert(not tt::is_callable<A, int, double>::value,
              "Failed testing type trait is_callable");
static_assert(not tt::is_callable<A, int>::value,
              "Failed testing type trait is_callable");
static_assert(tt::is_callable<A>::value,
              "Failed testing type trait is_callable");
static_assert(tt::is_callable<BClassInTestTypeTraits, int, double>::value,
              "Failed testing type trait is_callable");
static_assert(tt::is_callable_t<BClassInTestTypeTraits, int, double>::value,
              "Failed testing type trait is_callable");
static_assert(tt::is_callable_v<BClassInTestTypeTraits, int, double>,
              "Failed testing type trait is_callable");
static_assert(not tt::is_callable<BClassInTestTypeTraits>::value,
              "Failed testing type trait is_callable");
/// [is_callable_example]

namespace  {
/// [CREATE_IS_CALLABLE_EXAMPLE]
CREATE_IS_CALLABLE(foo)
CREATE_IS_CALLABLE(foobar)
struct bar {
  void foo(int /*unused*/, double /*unused*/) {}
};

static_assert(is_foo_callable_v<bar, int, double>,
              "Failed testing CREATE_IS_CALLABLE");
static_assert(not is_foo_callable_v<bar, int>,
              "Failed testing CREATE_IS_CALLABLE");
static_assert(not is_foo_callable_v<bar>,
              "Failed testing CREATE_IS_CALLABLE");
static_assert(not is_foobar_callable_v<bar, int, double>,
              "Failed testing CREATE_IS_CALLABLE");
/// [CREATE_IS_CALLABLE_EXAMPLE]
}  // namespace

/// [is_hashable_example]
static_assert(tt::is_hashable<double>::value,
              "Failed testing type trait is_hashable");
static_assert(tt::is_hashable_t<double>::value,
              "Failed testing type trait is_hashable");
static_assert(tt::is_hashable_v<double>,
              "Failed testing type trait is_hashable");
static_assert(tt::is_hashable<A>::value,
              "Failed testing type trait is_hashable");
// There is a bug in libc++ prior to version 4.0.0 that uses static_assert
// in the std::hash for enum instead of using something like enable_if
#if (defined(_LIBCPP_VERSION) && _LIBCPP_VERSION >= 4000) || \
    !(defined(_LIBCPP_VERSION))
static_assert(not tt::is_hashable<std::vector<std::unique_ptr<double>>>::value,
              "Failed testing type trait is_hashable");
#endif
/// [is_hashable_example]

/// [is_maplike_example]
static_assert(tt::is_maplike<std::unordered_map<int, double>>::value,
              "Failed testing type trait is_maplike");
static_assert(tt::is_maplike_t<std::unordered_map<int, double>>::value,
              "Failed testing type trait is_maplike");
static_assert(tt::is_maplike_v<std::unordered_map<int, double>>,
              "Failed testing type trait is_maplike");
static_assert(tt::is_maplike<std::map<int, CClassInTestTypeTraits>>::value,
              "Failed testing type trait is_maplike");
static_assert(not tt::is_maplike<std::vector<CClassInTestTypeTraits>>::value,
              "Failed testing type trait is_maplike");
static_assert(not tt::is_maplike<D>::value,
              "Failed testing type trait is_maplike");
/// [is_maplike_example]

/// [is_streamable_example]
static_assert(
    not tt::is_streamable<std::ostream, CClassInTestTypeTraits>::value,
    "Failed testing type trait is_streamable");
static_assert(
    not tt::is_streamable_t<std::ostream, CClassInTestTypeTraits>::value,
    "Failed testing type trait is_streamable");
static_assert(not tt::is_streamable_v<std::ostream, CClassInTestTypeTraits>,
              "Failed testing type trait is_streamable");
static_assert(not tt::is_streamable<std::ostream, A>::value,
              "Failed testing type trait is_streamable");
static_assert(tt::is_streamable<std::ostream, D>::value,
              "Failed testing type trait is_streamable");
/// [is_streamable_example]

/// [is_string_like_example]
static_assert(tt::is_string_like<std::string>::value,
              "Failed testing type trait is_string_like");
static_assert(tt::is_string_like_t<std::string>::value,
              "Failed testing type trait is_string_like");
static_assert(tt::is_string_like_v<std::string>,
              "Failed testing type trait is_string_like");
static_assert(tt::is_string_like<std::string&>::value,
              "Failed testing type trait is_string_like");
static_assert(tt::is_string_like<const std::string&>::value,
              "Failed testing type trait is_string_like");
static_assert(tt::is_string_like<std::string&&>::value,
              "Failed testing type trait is_string_like");
static_assert(tt::is_string_like<char*>::value,
              "Failed testing type trait is_string_like");
static_assert(tt::is_string_like<const char*>::value,
              "Failed testing type trait is_string_like");
static_assert(tt::is_string_like<char* const>::value,
              "Failed testing type trait is_string_like");
static_assert(tt::is_string_like<const char* const>::value,
              "Failed testing type trait is_string_like");
static_assert(tt::is_string_like<char>::value,
              "Failed testing type trait is_string_like");
static_assert(tt::is_string_like<const char>::value,
              "Failed testing type trait is_string_like");
static_assert(tt::is_string_like<char&&>::value,
              "Failed testing type trait is_string_like");
static_assert(tt::is_string_like<char&>::value,
              "Failed testing type trait is_string_like");
static_assert(not tt::is_string_like<int&>::value,
              "Failed testing type trait is_string_like");
static_assert(not tt::is_string_like<short&>::value,
              "Failed testing type trait is_string_like");
static_assert(not tt::is_string_like<int>::value,
              "Failed testing type trait is_string_like");
static_assert(not tt::is_string_like<short>::value,
              "Failed testing type trait is_string_like");
/// [is_string_like_example]

/// [has_get_clone_example]
static_assert(tt::has_get_clone<std::unique_ptr<A>>::value,
              "Failed testing type trait has_get_clone");
static_assert(tt::has_get_clone_t<std::unique_ptr<A>>::value,
              "Failed testing type trait has_get_clone");
static_assert(tt::has_get_clone_v<std::unique_ptr<A>>,
              "Failed testing type trait has_get_clone");
static_assert(tt::has_get_clone<A>::value,
              "Failed testing type trait has_get_clone");
static_assert(
    not tt::has_get_clone<std::unique_ptr<BClassInTestTypeTraits>>::value,
    "Failed testing type trait has_get_clone");
static_assert(not tt::has_get_clone<double>::value,
              "Failed testing type trait has_get_clone");
/// [has_get_clone_example]

/// [has_clone_example]
static_assert(tt::has_clone<BClassInTestTypeTraits>::value,
              "Failed testing type trait has_clone");
static_assert(tt::has_clone_t<BClassInTestTypeTraits>::value,
              "Failed testing type trait has_clone");
static_assert(tt::has_clone_v<BClassInTestTypeTraits>,
              "Failed testing type trait has_clone");
static_assert(tt::has_clone<std::unique_ptr<BClassInTestTypeTraits>>::value,
              "Failed testing type trait has_clone");
static_assert(not tt::has_clone<std::unique_ptr<A>>::value,
              "Failed testing type trait has_clone");
static_assert(not tt::has_clone<double>::value,
              "Failed testing type trait has_clone");
/// [has_clone_example]

/// [has_size_example]
static_assert(tt::has_size<A>::value, "Failed testing type trait has_size");
static_assert(tt::has_size_t<A>::value, "Failed testing type trait has_size");
static_assert(tt::has_size_v<A>, "Failed testing type trait has_size");
static_assert(tt::has_size<std::vector<BClassInTestTypeTraits>>::value,
              "Failed testing type trait has_size");
static_assert(not tt::has_size<BClassInTestTypeTraits>::value,
              "Failed testing type trait has_size");
static_assert(not tt::has_size<double>::value,
              "Failed testing type trait has_size");
/// [has_size_example]

/// [is_integer_example]
static_assert(tt::is_integer<short>::value,
              "Failed testing type trait is_integer");
static_assert(tt::is_integer_v<unsigned short>,
              "Failed testing type trait is_integer");
static_assert(tt::is_integer_v<int>, "Failed testing type trait is_integer");
static_assert(tt::is_integer_v<unsigned int>,
              "Failed testing type trait is_integer");
static_assert(tt::is_integer_v<long>, "Failed testing type trait is_integer");
static_assert(tt::is_integer_v<unsigned long>,
              "Failed testing type trait is_integer");
static_assert(tt::is_integer_v<long long>,
              "Failed testing type trait is_integer");
static_assert(tt::is_integer_v<unsigned long long>,
              "Failed testing type trait is_integer");
static_assert(not tt::is_integer_v<bool>,
              "Failed testing type trait is_integer");
static_assert(not tt::is_integer_v<char>,
              "Failed testing type trait is_integer");
/// [is_integer_example]

/// [remove_reference_wrapper_example]
static_assert(
    cpp17::is_same_v<const double, tt::remove_reference_wrapper_t<
                                       std::reference_wrapper<const double>>>,
    "Failed testing remove_reference_wrapper");
static_assert(cpp17::is_same_v<const double,
                               tt::remove_reference_wrapper_t<const double>>,
              "Failed testing remove_reference_wrapper");
static_assert(
    cpp17::is_same_v<
        double, tt::remove_reference_wrapper_t<std::reference_wrapper<double>>>,
    "Failed testing remove_reference_wrapper");
static_assert(cpp17::is_same_v<double, tt::remove_reference_wrapper_t<double>>,
              "Failed testing remove_reference_wrapper");
static_assert(cpp17::is_same_v<const A, tt::remove_reference_wrapper_t<
                                            std::reference_wrapper<const A>>>,
              "Failed testing remove_reference_wrapper");
static_assert(
    cpp17::is_same_v<const A, tt::remove_reference_wrapper_t<const A>>,
    "Failed testing remove_reference_wrapper");
static_assert(cpp17::is_same_v<
                  A, tt::remove_reference_wrapper_t<std::reference_wrapper<A>>>,
              "Failed testing remove_reference_wrapper");
static_assert(cpp17::is_same_v<A, tt::remove_reference_wrapper_t<A>>,
              "Failed testing remove_reference_wrapper");
static_assert(
    cpp17::is_same_v<double, tt::remove_reference_wrapper_t<
                                 const std::reference_wrapper<double>>>,
    "Failed testing remove_reference_wrapper");
static_assert(
    cpp17::is_same_v<double, tt::remove_reference_wrapper_t<
                                 volatile std::reference_wrapper<double>>>,
    "Failed testing remove_reference_wrapper");
static_assert(cpp17::is_same_v<
                  double, tt::remove_reference_wrapper_t<
                              const volatile std::reference_wrapper<double>>>,
              "Failed testing remove_reference_wrapper");
static_assert(cpp17::is_same_v<const double,
                               tt::remove_reference_wrapper_t<
                                   const std::reference_wrapper<const double>>>,
              "Failed testing remove_reference_wrapper");
static_assert(
    cpp17::is_same_v<const double,
                     tt::remove_reference_wrapper_t<
                         volatile std::reference_wrapper<const double>>>,
    "Failed testing remove_reference_wrapper");
static_assert(
    cpp17::is_same_v<const double,
                     tt::remove_reference_wrapper_t<
                         const volatile std::reference_wrapper<const double>>>,
    "Failed testing remove_reference_wrapper");
static_assert(
    cpp17::is_same_v<
        A, tt::remove_reference_wrapper_t<const std::reference_wrapper<A>>>,
    "Failed testing remove_reference_wrapper");
static_assert(
    cpp17::is_same_v<
        A, tt::remove_reference_wrapper_t<volatile std::reference_wrapper<A>>>,
    "Failed testing remove_reference_wrapper");
static_assert(
    cpp17::is_same_v<A, tt::remove_reference_wrapper_t<
                            const volatile std::reference_wrapper<A>>>,
    "Failed testing remove_reference_wrapper");
static_assert(
    cpp17::is_same_v<const A, tt::remove_reference_wrapper_t<
                                  const std::reference_wrapper<const A>>>,
    "Failed testing remove_reference_wrapper");
static_assert(
    cpp17::is_same_v<const A, tt::remove_reference_wrapper_t<
                                  volatile std::reference_wrapper<const A>>>,
    "Failed testing remove_reference_wrapper");
static_assert(cpp17::is_same_v<
                  const A, tt::remove_reference_wrapper_t<
                               const volatile std::reference_wrapper<const A>>>,
              "Failed testing remove_reference_wrapper");
/// [remove_reference_wrapper_example]

/// [remove_cvref_wrap]
static_assert(cpp17::is_same_v<tt::remove_cvref_wrap_t<int>, int>,
              "Failed testing remove_cvref_wrap");
static_assert(cpp17::is_same_v<tt::remove_cvref_wrap_t<int&>, int>,
              "Failed testing remove_cvref_wrap");
static_assert(cpp17::is_same_v<tt::remove_cvref_wrap_t<const int&>, int>,
              "Failed testing remove_cvref_wrap");
static_assert(cpp17::is_same_v<tt::remove_cvref_wrap_t<int&&>, int>,
              "Failed testing remove_cvref_wrap");
static_assert(cpp17::is_same_v<tt::remove_cvref_wrap_t<const int&&>, int>,
              "Failed testing remove_cvref_wrap");
static_assert(cpp17::is_same_v<tt::remove_cvref_wrap_t<const int>, int>,
              "Failed testing remove_cvref_wrap");
static_assert(cpp17::is_same_v<tt::remove_cvref_wrap_t<volatile int>, int>,
              "Failed testing remove_cvref_wrap");
static_assert(
    cpp17::is_same_v<tt::remove_cvref_wrap_t<const volatile int>, int>,
    "Failed testing remove_cvref_wrap");
static_assert(cpp17::is_same_v<tt::remove_cvref_wrap_t<volatile int&>, int>,
              "Failed testing remove_cvref_wrap");
static_assert(
    cpp17::is_same_v<tt::remove_cvref_wrap_t<const volatile int&>, int>,
    "Failed testing remove_cvref_wrap");
static_assert(cpp17::is_same_v<tt::remove_cvref_wrap_t<volatile int&&>, int>,
              "Failed testing remove_cvref_wrap");
static_assert(
    cpp17::is_same_v<tt::remove_cvref_wrap_t<const volatile int&&>, int>,
    "Failed testing remove_cvref_wrap");
static_assert(
    cpp17::is_same_v<tt::remove_cvref_wrap_t<std::reference_wrapper<const int>>,
                     int>,
    "Failed testing remove_cvref_wrap");
static_assert(
    cpp17::is_same_v<tt::remove_cvref_wrap_t<std::reference_wrapper<int>>, int>,
    "Failed testing remove_cvref_wrap");
static_assert(cpp17::is_same_v<
                  tt::remove_cvref_wrap_t<std::reference_wrapper<int*>>, int*>,
              "Failed testing remove_cvref_wrap");
static_assert(cpp17::is_same_v<
                  tt::remove_cvref_wrap_t<std::reference_wrapper<const int*>>,
                  const int*>,
              "Failed testing remove_cvref_wrap");
static_assert(
    cpp17::is_same_v<
        tt::remove_cvref_wrap_t<std::reference_wrapper<int* const>>, int*>,
    "Failed testing remove_cvref_wrap");
static_assert(
    cpp17::is_same_v<tt::remove_cvref_wrap_t<const std::reference_wrapper<int>>,
                     int>,
    "Failed testing remove_cvref_wrap");
static_assert(
    cpp17::is_same_v<
        tt::remove_cvref_wrap_t<volatile std::reference_wrapper<int>>, int>,
    "Failed testing remove_cvref_wrap");
static_assert(
    cpp17::is_same_v<
        tt::remove_cvref_wrap_t<const volatile std::reference_wrapper<int>>,
        int>,
    "Failed testing remove_cvref_wrap");
/// [remove_cvref_wrap]

/// [get_fundamental_type]
static_assert(
    cpp17::is_same_v<
        typename tt::get_fundamental_type<std::array<double, 2>>::type, double>,
    "Failed testing get_fundamental_type");
static_assert(
    cpp17::is_same_v<
        typename tt::get_fundamental_type_t<std::vector<std::complex<int>>>,
        int>,
    "Failed testing get_fundamental_type");
static_assert(cpp17::is_same_v<typename tt::get_fundamental_type_t<int>, int>,
              "Failed testing get_fundamental_type");
/// [get_fundamental_type]

/// [is_complex_of_fundamental]
static_assert(tt::is_complex_of_fundamental_v<std::complex<double>>,
              "Failed testing is_complex_of_fundamental");
static_assert(tt::is_complex_of_fundamental_v<std::complex<int>>,
              "Failed testing is_complex_of_fundamental");
static_assert(not tt::is_complex_of_fundamental_v<double>,
              "Failed testing is_complex_of_fundamental");
static_assert(not tt::is_complex_of_fundamental_v<std::complex<DataVector>>,
              "Failed testing is_complex_of_fundamental");
static_assert(not tt::is_complex_of_fundamental_v<DataVector>,
              "Failed testing is_complex_of_fundamental");
/// [is_complex_of_fundamental]

static_assert(tt::is_complex_or_fundamental_v<std::complex<double>>,
              "Failed testing is_complex_of_fundamental");
static_assert(tt::is_complex_or_fundamental_v<std::complex<int>>,
              "Failed testing is_complex_of_fundamental");
static_assert(tt::is_complex_or_fundamental_v<double>,
              "Failed testing is_complex_of_fundamental");
static_assert(not tt::is_complex_or_fundamental_v<std::complex<DataVector>>,
              "Failed testing is_complex_of_fundamental");
static_assert(not tt::is_complex_or_fundamental_v<DataVector>,
              "Failed testing is_complex_of_fundamental");

namespace {
// The name syntax is ReturnType_NumberOfArgs_Counter
// clang-tidy: no non-const references (we need them for testing)
void void_none() {}
double double_none() { return 1.0; }
void void_one_0(double /*unused*/) {}
void void_one_1(const double& /*unused*/) {}
void void_one_2(double& /*unused*/) {}  // NOLINT
void void_one_3(double* /*unused*/) {}
void void_one_4(const double* /*unused*/) {}
void void_one_5(const double* const /*unused*/) {}

int int_one_0(double /*unused*/) { return 0; }
int int_one_1(const double& /*unused*/) { return 0; }  // NOLINT
int int_one_2(double& /*unused*/) { return 0; }        // NOLINT
int int_one_3(double* /*unused*/) { return 0; }
int int_one_4(const double* /*unused*/) { return 0; }
int int_one_5(const double* const /*unused*/) { return 0; }

int int_two_0(double /*unused*/, char* /*unused*/) { return 0; }
int int_two_1(const double& /*unused*/, const char* /*unused*/) { return 0; }
int int_two_2(double& /*unused*/, const char* const /*unused*/) {  // NOLINT
  return 0;
}
int int_two_3(double* /*unused*/, char& /*unused*/) { return 0; }  // NOLINT
int int_two_4(const double* /*unused*/, const char& /*unused*/) { return 0; }
int int_two_5(const double* const /*unused*/, char /*unused*/) { return 0; }

// We have to NOLINT these for 2 reasons:
// - no non-const references
// - ClangTidy wants a () after the macros, which expands to not what we want
#define FUNCTION_INFO_TEST(NAME, PREFIX, POSTFIX)                              \
  struct FunctionInfoTest##NAME {                                              \
    PREFIX void void_none() POSTFIX {}                            /* NOLINT */ \
    PREFIX double double_none() POSTFIX { return 1.0; }           /* NOLINT */ \
    PREFIX void void_one_0(double /*unused*/) POSTFIX {}          /* NOLINT */ \
    PREFIX void void_one_1(const double& /*unused*/) POSTFIX {}   /* NOLINT */ \
    PREFIX void void_one_2(double& /*unused*/) POSTFIX {}         /* NOLINT */ \
    PREFIX void void_one_3(double* /*unused*/) POSTFIX {}         /* NOLINT */ \
    PREFIX void void_one_4(const double* /*unused*/) POSTFIX {}   /* NOLINT */ \
    PREFIX void void_one_5(                                       /* NOLINT */ \
                           const double* const /*unused*/)        /* NOLINT */ \
        POSTFIX {}                                                /* NOLINT */ \
    PREFIX int int_one_0(double /*unused*/) POSTFIX { return 0; } /* NOLINT */ \
    PREFIX int int_one_1(const double& /*unused*/) POSTFIX {      /* NOLINT */ \
      return 0;                                                                \
    }                                                                          \
    PREFIX int int_one_2(double& /*unused*/) POSTFIX { /* NOLINT */            \
      return 0;                                                                \
    }                                                                          \
    PREFIX int int_one_3(double* /*unused*/) POSTFIX { /* NOLINT */            \
      return 0;                                                                \
    }                                                                          \
    PREFIX int int_one_4(const double* /*unused*/) POSTFIX { /* NOLINT */      \
      return 0;                                                                \
    }                                                                          \
    PREFIX int int_one_5(const double* const /*unused*/) /* NOLINT */          \
        POSTFIX {                                        /* NOLINT */          \
      return 0;                                                                \
    }                                                                          \
    PREFIX int int_two_0(double /*unused*/,          /* NOLINT */              \
                         char* /*unused*/) POSTFIX { /* NOLINT */              \
      return 0;                                                                \
    }                                                                          \
    PREFIX int int_two_1(const double& /*unused*/,         /* NOLINT */        \
                         const char* /*unused*/) POSTFIX { /* NOLINT */        \
      return 0;                                                                \
    }                                                                          \
    PREFIX int int_two_2(double& /*unused*/,                     /* NOLINT */  \
                         const char* const /*unused*/) POSTFIX { /* NOLINT */  \
      return 0;                                                                \
    }                                                                          \
    PREFIX int int_two_3(double* /*unused*/,         /* NOLINT */              \
                         char& /*unused*/) POSTFIX { /* NOLINT */              \
      return 0;                                                                \
    }                                                                          \
    PREFIX int int_two_4(const double* /*unused*/,         /* NOLINT */        \
                         const char& /*unused*/) POSTFIX { /* NOLINT */        \
      return 0;                                                                \
    }                                                                          \
    PREFIX int int_two_5(const double* const /*unused*/, /* NOLINT */          \
                         char /*unused*/) POSTFIX {      /* NOLINT */          \
      return 0;                                                                \
    }                                                                          \
  };

FUNCTION_INFO_TEST(, , )
FUNCTION_INFO_TEST(Const, , const)
FUNCTION_INFO_TEST(Static, static, )
FUNCTION_INFO_TEST(Noexcept, , noexcept)
FUNCTION_INFO_TEST(ConstNoexcept, , const noexcept)
FUNCTION_INFO_TEST(StaticNoexcept, static, noexcept)

#undef FUNCTION_INFO_TEST

struct FunctionInfoTestVirtual {
  FunctionInfoTestVirtual(const FunctionInfoTestVirtual&) = default;
  FunctionInfoTestVirtual& operator=(const FunctionInfoTestVirtual&) = default;
  FunctionInfoTestVirtual(FunctionInfoTestVirtual&&) = default;
  FunctionInfoTestVirtual& operator=(FunctionInfoTestVirtual&&) = default;
  virtual ~FunctionInfoTestVirtual() = default;
  virtual void void_none() {}
  virtual double double_none() { return 1.0; }
  virtual void void_one_0(double /*unused*/) {}
  virtual void void_one_1(const double& /*unused*/) {}
  virtual void void_one_2(double& /*unused*/) {}  // NOLINT
  virtual void void_one_3(double* /*unused*/) {}
  virtual void void_one_4(const double* /*unused*/) {}
  virtual void void_one_5(const double* const /*unused*/) {}

  virtual int int_one_0(double /*unused*/) { return 0; }
  virtual int int_one_1(const double& /*unused*/) { return 0; }
  virtual int int_one_2(double& /*unused*/) { return 0; }  // NOLINT
  virtual int int_one_3(double* /*unused*/) { return 0; }
  virtual int int_one_4(const double* /*unused*/) { return 0; }
  virtual int int_one_5(const double* const /*unused*/) { return 0; }

  virtual int int_two_0(double /*unused*/, char* /*unused*/) { return 0; }
  virtual int int_two_1(const double& /*unused*/,  // NOLINT
                        const char* /*unused*/) {
    return 0;
  }
  virtual int int_two_2(double& /*unused*/,  // NOLINT
                        const char* const /*unused*/) {
    return 0;
  }
  virtual int int_two_3(double* /*unused*/, char& /*unused*/) {  // NOLINT
    return 0;
  }
  virtual int int_two_4(const double* /*unused*/, const char& /*unused*/) {
    return 0;
  }
  virtual int int_two_5(const double* const /*unused*/, char /*unused*/) {
    return 0;
  }
};

struct FunctionInfoTestVirtualBase {
  // Some of these are redundant because we don't need to const things in
  // forward decls. However, so that we can use the generic interface we keep
  // the functions.
  FunctionInfoTestVirtualBase(const FunctionInfoTestVirtualBase&) = default;
  FunctionInfoTestVirtualBase& operator=(const FunctionInfoTestVirtualBase&) =
      default;
  FunctionInfoTestVirtualBase(FunctionInfoTestVirtualBase&&) = default;
  FunctionInfoTestVirtualBase& operator=(FunctionInfoTestVirtualBase&&) =
      default;
  virtual ~FunctionInfoTestVirtualBase() = default;
  virtual void void_none() = 0;
  virtual double double_none() = 0;
  virtual void void_one_0(double /*unused*/) = 0;
  virtual void void_one_1(const double& /*unused*/) = 0;
  virtual void void_one_2(double& /*unused*/) = 0;  // NOLINT
  virtual void void_one_3(double* /*unused*/) = 0;
  virtual void void_one_4(const double* /*unused*/) = 0;
  virtual void void_one_5(const double* /*unused*/) = 0;

  virtual int int_one_0(double /*unused*/) = 0;
  virtual int int_one_1(const double& /*unused*/) = 0;
  virtual int int_one_2(double& /*unused*/) = 0;  // NOLINT
  virtual int int_one_3(double* /*unused*/) = 0;
  virtual int int_one_4(const double* /*unused*/) = 0;
  virtual int int_one_5(const double* /*unused*/) = 0;

  virtual int int_two_0(double /*unused*/, char* /*unused*/) = 0;
  virtual int int_two_1(const double& /*unused*/,                   // NOLINT
                        const char* /*unused*/) = 0;                // NOLINT
  virtual int int_two_2(double& /*unused*/,                         // NOLINT
                        const char* /*unused*/) = 0;                // NOLINT
  virtual int int_two_3(double* /*unused*/, char& /*unused*/) = 0;  // NOLINT
  virtual int int_two_4(const double* /*unused*/, const char& /*unused*/) = 0;
  virtual int int_two_5(const double* /*unused*/, char /*unused*/) = 0;
};

// Check that function_info works after LazyF is applied to a variety of
// different functions. LazyF is identity, add_pointer, add_const<add_pointer>,
// add_volatile<add_pointer>, and add_cv<add_pointer>. This allows us to check
// all the different combinations with a single class.
template <template <class> class LazyF>
struct check_function_info {
  template <class T>
  using F = typename LazyF<T>::type;

  template <class Function, class ReturnType, class ClassType, class... Args>
  struct Check {
    static_assert(
        cpp17::is_same_v<ReturnType,
                         typename tt::function_info<Function>::return_type>,
        "Failed testing function_info");
    static_assert(
        cpp17::is_same_v<tmpl::list<Args...>,
                         typename tt::function_info<Function>::argument_types>,
        "Failed testing function_info");
    static_assert(
        cpp17::is_same_v<ClassType,
                         typename tt::function_info<Function>::class_type>,
        "Failed testing function_info");
    static constexpr bool t = true;
  };

  static constexpr Check<F<decltype(void_none)>, void, void> t_void_none{};
  static constexpr Check<F<decltype(double_none)>, double, void>
      t_double_none{};

  static constexpr Check<F<decltype(void_one_0)>, void, void, double>
      t_void_one_0{};
  static constexpr Check<F<decltype(void_one_1)>, void, void, const double&>
      t_void_one_1{};
  static constexpr Check<F<decltype(void_one_2)>, void, void, double&>
      t_void_one_2{};
  static constexpr Check<F<decltype(void_one_3)>, void, void, double*>
      t_void_one_3{};
  static constexpr Check<F<decltype(void_one_4)>, void, void, const double*>
      t_void_one_4{};
  static constexpr Check<F<decltype(void_one_5)>, void, void, const double*>
      t_void_one_5{};

  static constexpr Check<F<decltype(int_one_0)>, int, void, double>
      t_int_one_0{};
  static constexpr Check<F<decltype(int_one_1)>, int, void, const double&>
      t_int_one_1{};
  static constexpr Check<F<decltype(int_one_2)>, int, void, double&>
      t_int_one_2{};
  static constexpr Check<F<decltype(int_one_3)>, int, void, double*>
      t_int_one_3{};
  static constexpr Check<F<decltype(int_one_4)>, int, void, const double*>
      t_int_one_4{};
  static constexpr Check<F<decltype(int_one_5)>, int, void, const double*>
      t_int_one_5{};

  static constexpr Check<F<decltype(int_two_0)>, int, void, double, char*>
      t_int_two_0{};
  static constexpr Check<F<decltype(int_two_1)>, int, void, const double&,
                         const char*>
      t_int_two_1{};
  static constexpr Check<F<decltype(int_two_2)>, int, void, double&,
                         const char*>
      t_int_two_2{};
  static constexpr Check<F<decltype(int_two_3)>, int, void, double*, char&>
      t_int_two_3{};
  static constexpr Check<F<decltype(int_two_4)>, int, void, const double*,
                         const char&>
      t_int_two_4{};
  static constexpr Check<F<decltype(int_two_5)>, int, void, const double*, char>
      t_int_two_5{};

  template <class Scope, class Class = Scope>
  struct CheckClass {
    // We have to remove the pointer since we are already adding a pointer
    // sometimes using F
    static constexpr bool t_void_none =
        Check<std::remove_pointer_t<F<decltype(&Scope::void_none)>>, void,
              Class>::t;
    static constexpr Check<
        std::remove_pointer_t<F<decltype(&Scope::double_none)>>, double, Class>
        t_double_none{};

    static constexpr Check<
        std::remove_pointer_t<F<decltype(&Scope::void_one_0)>>, void, Class,
        double>
        t_void_one_0{};
    static constexpr Check<
        std::remove_pointer_t<F<decltype(&Scope::void_one_1)>>, void, Class,
        const double&>
        t_void_one_1{};
    static constexpr Check<
        std::remove_pointer_t<F<decltype(&Scope::void_one_2)>>, void, Class,
        double&>
        t_void_one_2{};
    static constexpr Check<
        std::remove_pointer_t<F<decltype(&Scope::void_one_3)>>, void, Class,
        double*>
        t_void_one_3{};
    static constexpr Check<
        std::remove_pointer_t<F<decltype(&Scope::void_one_4)>>, void, Class,
        const double*>
        t_void_one_4{};
    static constexpr Check<
        std::remove_pointer_t<F<decltype(&Scope::void_one_5)>>, void, Class,
        const double*>
        t_void_one_5{};

    static constexpr Check<
        std::remove_pointer_t<F<decltype(&Scope::int_one_0)>>, int, Class,
        double>
        t_int_one_0{};
    static constexpr Check<
        std::remove_pointer_t<F<decltype(&Scope::int_one_1)>>, int, Class,
        const double&>
        t_int_one_1{};
    static constexpr Check<
        std::remove_pointer_t<F<decltype(&Scope::int_one_2)>>, int, Class,
        double&>
        t_int_one_2{};
    static constexpr Check<
        std::remove_pointer_t<F<decltype(&Scope::int_one_3)>>, int, Class,
        double*>
        t_int_one_3{};
    static constexpr Check<
        std::remove_pointer_t<F<decltype(&Scope::int_one_4)>>, int, Class,
        const double*>
        t_int_one_4{};
    static constexpr Check<
        std::remove_pointer_t<F<decltype(&Scope::int_one_5)>>, int, Class,
        const double*>
        t_int_one_5{};

    static constexpr Check<
        std::remove_pointer_t<F<decltype(&Scope::int_two_0)>>, int, Class,
        double, char*>
        t_int_two_0{};
    static constexpr Check<
        std::remove_pointer_t<F<decltype(&Scope::int_two_1)>>, int, Class,
        const double&, const char*>
        t_int_two_1{};
    static constexpr Check<
        std::remove_pointer_t<F<decltype(&Scope::int_two_2)>>, int, Class,
        double&, const char*>
        t_int_two_2{};
    static constexpr Check<
        std::remove_pointer_t<F<decltype(&Scope::int_two_3)>>, int, Class,
        double*, char&>
        t_int_two_3{};
    static constexpr Check<
        std::remove_pointer_t<F<decltype(&Scope::int_two_4)>>, int, Class,
        const double*, const char&>
        t_int_two_4{};
    static constexpr Check<
        std::remove_pointer_t<F<decltype(&Scope::int_two_5)>>, int, Class,
        const double*, char>
        t_int_two_5{};
  };

  static constexpr CheckClass<FunctionInfoTest> non_const_members{};
  static constexpr CheckClass<FunctionInfoTestConst> const_members{};
  static constexpr CheckClass<FunctionInfoTestConstNoexcept>
      const_noexcept_members{};
  static constexpr CheckClass<FunctionInfoTestNoexcept> noexcept_members{};
  static constexpr CheckClass<FunctionInfoTestVirtual> virtual_members{};
  static constexpr CheckClass<FunctionInfoTestVirtualBase>
      virtual_base_members{};
  static constexpr CheckClass<FunctionInfoTestStatic, void> static_members{};
  static constexpr CheckClass<FunctionInfoTestStaticNoexcept, void>
      static_noexcept_members{};
};

template <template <class> class F>
struct add_pointer_helper {
  template <class T>
  struct impl {
    using type = typename F<std::add_pointer_t<T>>::type;
  };
};

template <class T>
struct identity {
  using type = T;
};

// Scope function calls to avoid warnings
struct TestFunctions {
  TestFunctions() noexcept {
    (void)check_function_info<identity>{};
    (void)check_function_info<std::add_pointer>{};
    (void)check_function_info<add_pointer_helper<std::add_const>::impl>{};
    (void)check_function_info<add_pointer_helper<std::add_volatile>::impl>{};
    (void)check_function_info<add_pointer_helper<std::add_cv>::impl>{};
    // Use these to avoid warnings
    void_none();
    (void)double_none();

    double t_double = 1.0;
    void_one_0(t_double);
    void_one_1(t_double);
    void_one_2(t_double);
    void_one_3(&t_double);
    void_one_4(&t_double);
    void_one_5(&t_double);

    (void)int_one_0(t_double);
    (void)int_one_1(t_double);
    (void)int_one_2(t_double);
    (void)int_one_3(&t_double);
    (void)int_one_4(&t_double);
    (void)int_one_5(&t_double);

    char t_char = 'a';
    (void)int_two_0(t_double, &t_char);
    (void)int_two_1(t_double, &t_char);
    (void)int_two_2(t_double, &t_char);
    (void)int_two_3(&t_double, t_char);
    (void)int_two_4(&t_double, t_char);
    (void)int_two_5(&t_double, t_char);
  }
};
TestFunctions test_functions{};

static_assert(cpp17::is_same_v<tt::identity_t<double>, double>,
              "Failed testing tt::identity_t");
static_assert(cpp17::is_same_v<tt::identity_t<double, 10>, double>,
              "Failed testing tt::identity_t");
template <typename = std::make_index_sequence<3>, typename...>
struct IdentityPackExample;

/// [example_identity_t]
template <size_t... Is>
struct IdentityPackExample<std::index_sequence<Is...>> {
  using type = std::tuple<tt::identity_t<double, Is>...>;
};
/// [example_identity_t]
}  // namespace
