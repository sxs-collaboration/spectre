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

std::ostream& operator<<(std::ostream& os, const D&) noexcept;

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
