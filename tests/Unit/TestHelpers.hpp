// Distributed under the MIT License.
// See LICENSE.txt for details.

/// \file
/// Commonly used routines, functions and definitions shared amongst unit tests

#pragma once

#include "tests/Unit/TestingFramework.hpp"

#include <algorithm>
#include <array>
#include <boost/algorithm/string/predicate.hpp>
#include <cstddef>
#include <iterator>
#include <memory>
#include <ostream>
#include <random>
#include <string>
#include <tuple>

#include "ErrorHandling/Assert.hpp"
#include "ErrorHandling/Error.hpp"
#include "Parallel/RegisterDerivedClassesWithCharm.hpp"
#include "Parallel/Serialize.hpp"
#include "Utilities/DereferenceWrapper.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/MakeArray.hpp"
#include "Utilities/Requires.hpp"
#include "Utilities/StdArrayHelpers.hpp"  // IWYU pragma: keep
#include "Utilities/TMPL.hpp"
#include "Utilities/Tuple.hpp"
#include "Utilities/TypeTraits.hpp"

/*!
 * \ingroup TestingFrameworkGroup
 * \brief Serializes and deserializes an object `t` of type `T`
 */
template <typename T>
T serialize_and_deserialize(const T& t) {
  static_assert(
      std::is_default_constructible<T>::value,
      "Cannot use serialize_and_deserialize if a class is not default "
      "constructible.");
  return deserialize<T>(serialize<T>(t).data());
}

/// \ingroup TestingFrameworkGroup
/// \brief Tests the serialization of comparable types
/// \example
/// \snippet Test_PupStlCpp11.cpp example_serialize_comparable
template <typename T>
void test_serialization(const T& t) {
  static_assert(tt::has_equivalence_v<T>, "No operator== for T");
  CHECK(t == serialize_and_deserialize(t));
}

/// \ingroup TestingFrameworkGroup
/// \brief Test the serialization of a derived class via a base class pointer
/// \example
/// \snippet Test_PupStlCpp11.cpp example_serialize_derived
/// \tparam B the base class
/// \tparam D the derived class
/// \tparam Args deduced from `args`
/// \param args arguments passed to a constructor of the derived class
template <typename B, typename D, typename... Args>
void test_serialization_via_base(Args&&... args) {
  static_assert(cpp17::is_base_of_v<B, D>,
                "passed input type is not derived from specified base");
  static_assert(tt::has_equivalence_v<D>, "No operator== for derived class");
  Parallel::register_derived_classes_with_charm<B>();
  std::unique_ptr<B> base = std::make_unique<D>(args...);
  std::unique_ptr<B> pupped_base = serialize_and_deserialize(base);
  CHECK_FALSE(nullptr == dynamic_cast<const D*>(pupped_base.get()));
  const D derived(args...);
  CHECK(derived == dynamic_cast<const D&>(*pupped_base));
}

/// Test for copy semantics assuming operator== is implement correctly
template <typename T, Requires<tt::has_equivalence<T>::value> = nullptr>
void test_copy_semantics(const T& a) {
  static_assert(std::is_copy_assignable<T>::value,
                "Class is not copy assignable.");
  static_assert(std::is_copy_constructible<T>::value,
                "Class is not copy constructible.");
  T b = a;
  CHECK(b == a);
  // clang-tidy: intentionally not a reference to force invocation of copy
  // constructor
  const T c(a);  // NOLINT
  CHECK(c == a);
#ifndef __APPLE__
#if defined(__clang__) && __clang_major__ > 6
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wself-assign-overloaded"
#endif  // defined(__clang__) && __clang_major__ > 6
#endif  // ! __APPLE__
  // clang-tidy: self-assignment
  b = b;  // NOLINT
#ifndef __APPLE__
#if defined(__clang__) && __clang_major__ > 6
#pragma GCC diagnostic pop
#endif  // defined(__clang__) && __clang_major__ > 6
#endif  // ! __APPLE__
  CHECK(b == a);
}

/// Test for move semantics assuming operator== is implemented correctly.
/// \requires `std::is_rvalue_reference<decltype(a)>::%value` is true.
/// If T is not default constructible, you pass additional
/// arguments that are used to construct a T.
template <typename T, Requires<tt::has_equivalence<T>::value> = nullptr,
          typename... Args>
void test_move_semantics(T&& a, const T& comparison, Args&&... args) {
  static_assert(std::is_rvalue_reference<decltype(a)>::value,
                "Must move into test_move_semantics");
  static_assert(std::is_nothrow_move_assignable<T>::value,
                "Class is not nothrow move assignable.");
  static_assert(std::is_nothrow_move_constructible<T>::value,
                "Class is not nothrow move constructible.");
  if (&a == &comparison or a != comparison) {
    // We use ERROR instead of ASSERT (which we normally should be using) to
    // guard against someone writing tests in Release mode where ASSERTs don't
    // show up.
    ERROR("'a' and 'comparison' must be distinct (but equal in value) objects");
  }
  T b(std::forward<Args>(args)...);
  // clang-tidy: use std::forward instead of std::move
  b = std::move(a);  // NOLINT
  CHECK(b == comparison);
  T c(std::move(b));
  CHECK(c == comparison);
}

// Test for iterators
template <typename Container>
void test_iterators(Container& c) {
  CHECK(std::distance(c.begin(), c.end()) ==
        static_cast<decltype(std::distance(c.begin(), c.end()))>(c.size()));
  CHECK(c.begin() == c.cbegin());
  CHECK(c.end() == c.cend());

  const auto& const_c = c;
  CHECK(std::distance(const_c.begin(), const_c.end()) ==
        static_cast<decltype(std::distance(const_c.begin(), const_c.end()))>(
            const_c.size()));
  CHECK(const_c.begin() == const_c.cbegin());
  CHECK(const_c.end() == const_c.cend());
}

// Test for reverse iterators
template <typename Container>
void test_reverse_iterators(Container& c) {
  CHECK(std::distance(c.rbegin(), c.rend()) ==
        static_cast<decltype(std::distance(c.rbegin(), c.rend()))>(c.size()));

  CHECK(c.rbegin() == c.crbegin());
  CHECK(c.rend() == c.crend());

  auto it = c.begin();
  auto rit = c.rbegin();
  auto end = c.end();
  auto rend = c.rend();
  auto cit = c.cbegin();
  auto cend = c.cend();
  auto crit = c.crbegin();
  auto crend = c.crend();

  for (size_t i = 0; i < c.size(); i++) {
    CHECK(*it == *(std::prev(rend, 1)));
    CHECK(*rit == *(std::prev(end, 1)));
    CHECK(*cit == *(std::prev(crend, 1)));
    CHECK(*crit == *(std::prev(cend, 1)));
    it++;
    rit++;
    rend--;
    end--;
    crit++;
    cit++;
    crend--;
    cend--;
  }

  const auto& const_c = c;
  CHECK(std::distance(const_c.begin(), const_c.end()) ==
        static_cast<decltype(std::distance(const_c.begin(), const_c.end()))>(
            const_c.size()));
  auto c_it = const_c.begin();
  auto c_rit = const_c.rbegin();
  auto c_end = const_c.end();
  auto c_rend = const_c.rend();
  for (size_t i = 0; i < c.size(); i++) {
    CHECK(*c_it == *(std::prev(c_rend, 1)));
    CHECK(*c_rit == *(std::prev(c_end, 1)));
    c_it++;
    c_rit++;
    c_rend--;
    c_end--;
  }
}

/*!
 * \ingroup TestingFrameworkGroup
 * \brief Function to test comparison operators.  Pass values with
 * less < greater.
 */
template <typename T, typename U>
void check_cmp(const T& less, const U& greater) {
  CHECK(less == less);
  CHECK_FALSE(less == greater);
  CHECK(less != greater);
  CHECK_FALSE(less != less);
  CHECK(less < greater);
  CHECK_FALSE(greater < less);
  CHECK(greater > less);
  CHECK_FALSE(less > greater);
  CHECK(less <= greater);
  CHECK_FALSE(greater <= less);
  CHECK(greater >= less);
  CHECK_FALSE(less >= greater);
  CHECK(less <= less);
  CHECK_FALSE(less < less);
  CHECK(less >= less);
  CHECK_FALSE(less > less);
}

/*!
 * \ingroup TestingFrameworkGroup
 * \brief Check a op b == c and also the op= version.
 */
#define CHECK_OP(a, op, b, c)   \
  do {                          \
    const auto& a_ = a;         \
    const auto& b_ = b;         \
    const auto& c_ = c;         \
    CHECK(a_ op b_ == c_);      \
    auto f = a_;                \
    CHECK((f op## = b_) == c_); \
    CHECK(f == c_);             \
  } while (false)

/*!
 * \ingroup TestingFrameworkGroup
 * \brief Calculates the derivative of an Invocable at a point x - represented
 * by an array of doubles - in the domain of `map` with a sixth-order finite
 * difference method.
 *
 * \details Intended for use with CoordinateMaps taking the domain {xi,eta,zeta}
 * to the range {x,y,z}. This function calculates the derivative along the
 * direction given by `direction` with a step size of `h`.
 *
 * \requires direction be between 0 and VolumeDim
 */
template <typename Invocable, size_t VolumeDim>
std::array<double, VolumeDim> numerical_derivative(
    const Invocable& map, const std::array<double, VolumeDim>& x,
    const size_t direction, const double delta) {
  ASSERT(0 <= direction and direction < VolumeDim,
         "Trying to take derivative along axis " << direction);

  const auto dx = [direction, delta]() {
    auto d = make_array<VolumeDim>(0.);
    gsl::at(d, direction) = delta;
    return d;
  }();

  const std::array<double, VolumeDim> x_1ahead = x + dx;
  const std::array<double, VolumeDim> x_2ahead = x_1ahead + dx;
  const std::array<double, VolumeDim> x_3ahead = x_2ahead + dx;
  const std::array<double, VolumeDim> x_1behind = x - dx;
  const std::array<double, VolumeDim> x_2behind = x_1behind - dx;
  const std::array<double, VolumeDim> x_3behind = x_2behind - dx;
  return (1.0 / (60.0 * delta)) * map(x_3ahead) +
         (-3.0 / (20.0 * delta)) * map(x_2ahead) +
         (0.75 / delta) * map(x_1ahead) + (-0.75 / delta) * map(x_1behind) +
         (3.0 / (20.0 * delta)) * map(x_2behind) +
         (-1.0 / (60.0 * delta)) * map(x_3behind);
}

struct NonCopyable {
  constexpr NonCopyable() = default;
  constexpr NonCopyable(const NonCopyable&) = delete;
  constexpr NonCopyable& operator=(const NonCopyable&) = delete;
  constexpr NonCopyable(NonCopyable&&) = default;
  NonCopyable& operator=(NonCopyable&&) = default;
  ~NonCopyable() = default;
};
inline bool operator==(const NonCopyable& /*a*/,
                       const NonCopyable& /*b*/) noexcept {
  return true;
}
inline bool operator!=(const NonCopyable& a, const NonCopyable& b) noexcept {
  return not(a == b);
}
inline std::ostream& operator<<(std::ostream& os,
                                const NonCopyable& /*v*/) noexcept {
  return os << "NC";
}

class DoesNotThrow {
 public:
  DoesNotThrow() noexcept = default;
  DoesNotThrow(const DoesNotThrow&) noexcept = default;
  DoesNotThrow& operator=(const DoesNotThrow&) noexcept = default;
  DoesNotThrow(DoesNotThrow&&) noexcept = default;
  DoesNotThrow& operator=(DoesNotThrow&&) noexcept = default;
  ~DoesNotThrow() = default;
};
class DoesThrow {
 public:
  DoesThrow() noexcept(false);
  DoesThrow(const DoesThrow&) noexcept(false);
  DoesThrow& operator=(const DoesThrow&) noexcept(false);
  DoesThrow(DoesThrow&&) noexcept(false);
  DoesThrow& operator=(DoesThrow&&) noexcept(false);
  ~DoesThrow() = default;
};

/*!
 * \ingroup TestingFrameworkGroup
 * \brief Execute `func` and check that it throws an exception `expected`.
 *
 * \note The `.what()` strings of the thrown and `expected` exceptions are
 * compared for a partial match only: the `expected.what()` string must be
 * contained in (or equal to) the `.what()` string of the thrown exception.
 */
template <typename Exception, typename ThrowingFunctor>
void test_throw_exception(const ThrowingFunctor& func,
                          const Exception& expected) {
  try {
    func();
    INFO("Failed to throw any exception");
    CHECK(false);
  } catch (Exception& e) {
    CAPTURE(e.what());
    CAPTURE(expected.what());
    CHECK(boost::contains(std::string(e.what()), std::string(expected.what())));
  } catch (...) {
    INFO("Failed to throw exception of type " +
         pretty_type::get_name<Exception>());
    CHECK(false);
  }
}

/// \ingroup TestingFrameworkGroup
/// \brief Make a generator of type `std::mt19937`, stores it in a declared
/// variable of name `NAME`
///
/// \details As the generator is made, `INFO` is called to make sure failed
/// tests provide seed information.
#define MAKE_GENERATOR(NAME) \
  std::random_device r;      \
  const auto seed = r();     \
  INFO("Seed is: " << seed); \
  auto(NAME) = std::mt19937{seed};

/*!
 * \ingroup TestingFrameworkGroup
 * \brief A test utility for verifying that an element wise function, `func`
 * acts identically to the same operation applied to each element of a container
 * separately. This macro invokes `test_element_wise_function()` (which gives a
 * more complete documentation of the element wise checking operations and
 * arguments).
 */
#define CHECK_ELEMENT_WISE_FUNCTION_APPROX(func, args)                     \
  do {                                                                     \
    INFO(__FILE__ ":" + std::to_string(__LINE__) + ": " #func ", " #args); \
    test_element_wise_function(func, args);                                \
  } while (false)

/*!
 * \ingroup TestingFrameworkGroup
 * \brief Same as `CHECK_ELEMENT_WISE_FUNCTION_APPROX`, but with a user-defined
 * function `at` and `size_of`, each of which correspond to arguments of
 * `test_element_wise_function()` (which gives a more complete documentation of
 * the element wise checking operations and arguments).
 */
#define CHECK_CUSTOM_ELEMENT_WISE_FUNCTION_APPROX(func, args, at, size_of) \
  do {                                                                     \
    INFO(__FILE__ ":" + std::to_string(__LINE__) + ": " #func ", " #args); \
    test_element_wise_function(func, args, at, size_of);                   \
  } while (false)

namespace TestHelpersDetail {

// function which returns the size of a container, or if passed a non-iterable,
// returns 0.  Used for supporting e.g. checking double + DataVector works as
// intended
template <typename SizeFunc, typename T,
          Requires<tt::is_iterable_v<T>> = nullptr>
size_t maybe_size(SizeFunc size_of, T container) noexcept {
  return size_of(container);
}

template <typename SizeFunc, typename T,
          Requires<not tt::is_iterable_v<T>> = nullptr>
size_t maybe_size(SizeFunc /*size_of*/, T /*value*/) noexcept {
  return 0;
}

// function which returns the value at a specified index of a container, or if
// passed a non-iterable, simply returns the value passed. Used for supporting
// e.g. checking double + DataVector works as intended
template <typename IndexingFunc, typename T,
          Requires<tt::is_iterable_v<T>> = nullptr>
auto value_at_or_value_of(IndexingFunc at, T container, size_t i) noexcept {
  return at(container, i);
}

template <typename IndexingFunc, typename T,
          Requires<not tt::is_iterable_v<T>> = nullptr>
auto value_at_or_value_of(IndexingFunc /*at*/, T value, size_t /*i*/) noexcept {
  return value;
}

// CHECK forwarding for parameter pack expansion.
SPECTRE_ALWAYS_INLINE int call_check(bool input) noexcept {
  CHECK(input);
  return 0;
}

// internal implementation for checking that an element wise function acts
// appropriately on a container object by first trying the operation on the
// containers, then checking that the operation has been performed as it would
// by looping over the elements and performing the same operation on each. This
// also supports testing functions between containers and single elements by
// making use of the above `maybe_size` and `value_at_or_value_of`.
template <typename Func, typename IndexingFunc, typename SizeFunc,
          typename... Args, size_t... Is>
void test_element_wise_function_impl(Func func, std::tuple<Args...> args,
                                     IndexingFunc at, SizeFunc size_of,
                                     std::index_sequence<Is...> /*meta*/,
                                     Approx appx) noexcept {
  size_t size_val = std::max({maybe_size(size_of, std::get<Is>(args))...});
  tuple_fold(args, [&size_of, &size_val](auto x) {
    ASSERT(maybe_size(size_of, x) == size_val or maybe_size(size_of, x) == 0,
           "inconsistent sized arguments passed "
           "to test_element_wise_function");
  });
  // Some operators might modify the values of the arguments. We take care to
  // verify that the modifications (or lack thereof) are also as expected.
  auto original_args = std::make_tuple(std::get<Is>(args)...);
  auto result = func(std::get<Is>(args)...);
  for (size_t i = 0; i < size_val; ++i) {
    // the arguments which modify their arguments must be passed lvalues
    auto element_of_args = std::make_tuple(
        value_at_or_value_of(at, std::get<Is>(original_args), i)...);
    CHECK(appx(value_at_or_value_of(at, result, i)) ==
          func(std::get<Is>(element_of_args)...));
    // ensure that the final state of the arguments matches
    expand_pack(
        call_check(appx(value_at_or_value_of(at, std::get<Is>(args), i)) ==
                   std::get<Is>(element_of_args))...);
  }
}
}  // namespace TestHelpersDetail

/*!
 * \ingroup TestingFrameworkGroup
 * \brief Utility function for verifying the action of an element wise function
 * on containers, or on some combination of containers and compatible
 * non-containers (e.g. DataVectors with doubles).
 *
 * \details The ability to specify custom functions for `at` and `size_of` is
 * useful for more intricate containers. For instance, multidimensional types
 * can be used with this function with a size function that returns the full
 * number of elements, and an `at` function which indexes the multidimensional
 * type in a flattened fashion.
 *
 * parameters:
 * - `func` A callable which is expected to act in an element wise fashion, must
 *    be compatible both with the container and its individual elements.
 * - `args` A tuple of arguments to be tested
 * - `at` A function to override the container access function. Defaults to an
 *   object which simply calls `container.at(i)`. A custom callable must take as
 *   arguments the container(s) used in `args` and a `size_t` index (in that
 *   order), and return an element compatible with `func`. This function
 *   signature follows the convention of `gsl::at`.
 * - `size_of` A function to override the container size function. Defaults to
 *   an object which simply calls `container.size()`. A custom callable must
 *   take as argument the container(s) used in `args`, and return a size_t. This
 *   function signature follows the convention of `cpp17::size`.
 */
template <typename Func, typename AtFunc, typename SizeFunc, typename... Args>
void test_element_wise_function(Func func, std::tuple<Args...> args,
                                AtFunc at = gsl::at,
                                SizeFunc size_of = cpp17::size,
                                Approx appx = approx) noexcept {
  TestHelpersDetail::test_element_wise_function_impl(
      func, args, at, size_of, std::make_index_sequence<sizeof...(Args)>{},
      appx);
}
