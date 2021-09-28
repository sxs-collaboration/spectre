// Distributed under the MIT License.
// See LICENSE.txt for details.

/// \file
/// Commonly used routines, functions and definitions shared amongst unit tests

#pragma once

#include "Framework/TestingFramework.hpp"

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

#include "DataStructures/DataBox/TagName.hpp"
#include "DataStructures/Variables.hpp"
#include "Parallel/RegisterDerivedClassesWithCharm.hpp"
#include "Parallel/Serialize.hpp"
#include "Utilities/ContainerHelpers.hpp"
#include "Utilities/DereferenceWrapper.hpp"
#include "Utilities/ErrorHandling/Assert.hpp"
#include "Utilities/ErrorHandling/Error.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/MakeArray.hpp"
#include "Utilities/MakeString.hpp"
#include "Utilities/Requires.hpp"
#include "Utilities/StdArrayHelpers.hpp"  // IWYU pragma: keep
#include "Utilities/TMPL.hpp"
#include "Utilities/Tuple.hpp"
#include "Utilities/TypeTraits.hpp"
#include "Utilities/TypeTraits/HasEquivalence.hpp"

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

/*!
 * \ingroup TestingFrameworkGroup
 * \brief Serializes and deserializes an object `t` of type `T`
 */
template <typename T>
void serialize_and_deserialize(const gsl::not_null<T*> result, const T& t) {
  deserialize<T>(result, serialize<T>(t).data());
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
  static_assert(std::is_base_of_v<B, D>,
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
#if defined(__clang__) && __clang_major__ > 6
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wself-assign-overloaded"
#endif  // defined(__clang__) && __clang_major__ > 6
  // clang-tidy: self-assignment
  b = b;  // NOLINT
#if defined(__clang__) && __clang_major__ > 6
#pragma GCC diagnostic pop
#endif  // defined(__clang__) && __clang_major__ > 6
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
  static_assert(std::is_move_assignable<T>::value);
  static_assert(std::is_move_constructible<T>::value);
  if (&a == &comparison or a != comparison) {
    // We use ERROR instead of ASSERT (which we normally should be using) to
    // guard against someone writing tests in Release mode where ASSERTs don't
    // show up.
    ERROR("'a' and 'comparison' must be distinct (but equal in value) objects");
  }
  T b{std::forward<Args>(args)...};
  // clang-tidy: use std::forward instead of std::move
  b = std::move(a);  // NOLINT
  {
    INFO("Test move assignment");
    CHECK(b == comparison);
  }
  T c(std::move(b));
  {
    INFO("Test move construction");
    CHECK(c == comparison);
  }
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
std::result_of_t<const Invocable&(const std::array<double, VolumeDim>&)>
numerical_derivative(const Invocable& function,
                     const std::array<double, VolumeDim>& x,
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
  return (1.0 / (60.0 * delta)) * function(x_3ahead) +
         (-3.0 / (20.0 * delta)) * function(x_2ahead) +
         (0.75 / delta) * function(x_1ahead) +
         (-0.75 / delta) * function(x_1behind) +
         (3.0 / (20.0 * delta)) * function(x_2behind) +
         (-1.0 / (60.0 * delta)) * function(x_3behind);
}

struct NonCopyable {
  constexpr NonCopyable() = default;
  constexpr NonCopyable(const NonCopyable&) = delete;
  constexpr NonCopyable& operator=(const NonCopyable&) = delete;
  constexpr NonCopyable(NonCopyable&&) = default;
  NonCopyable& operator=(NonCopyable&&) = default;
  ~NonCopyable() = default;
};
inline bool operator==(const NonCopyable& /*a*/, const NonCopyable& /*b*/) {
  return true;
}
inline bool operator!=(const NonCopyable& a, const NonCopyable& b) {
  return not(a == b);
}
inline std::ostream& operator<<(std::ostream& os, const NonCopyable& /*v*/) {
  return os << "NC";
}

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

/// \cond
#define MAKE_GENERATOR_IMPL_FIRST_ARG(NAME, ...) NAME
#define MAKE_GENERATOR_IMPL_SECOND_ARG(NAME, SEED, ...) SEED
/// \endcond

/// \ingroup TestingFrameworkGroup
/// \brief `MAKE_GENERATOR(NAME [, SEED])` declares a variable of name `NAME`
/// containing a generator of type `std::mt19937`.
///
/// \details As the generator is made, `INFO` is called to make sure failed
/// tests provide seed information.  `SEED` is chosen randomly if not supplied,
/// otherwise it must be a constant expression.
// What is going on here:
//
// If this is called as MAKE_GENERATOR(NAME):
//   MAKE_GENERATOR_IMPL_FIRST_ARG(__VA_ARGS__, DUMMY_TOKEN)
//     -> MAKE_GENERATOR_IMPL_FIRST_ARG(NAME, DUMMY_TOKEN)
//     -> NAME
//   MAKE_GENERATOR_IMPL_SECOND_ARG(
//       __VA_ARGS__, std::random_device{}(), DUMMY_TOKEN)
//     -> MAKE_GENERATOR_IMPL_SECOND_ARG(
//            NAME, std::random_device{}(), DUMMY_TOKEN)
//     -> std::random_device{}()
//   So we create NAME with a random seed.
//
//   In this case DUMMY_TOKEN is needed because the "..." in the IMPL
//   macros has to match at least one thing.
//
// If this is called as MAKE_GENERATOR(NAME, SEED):
//   MAKE_GENERATOR_IMPL_FIRST_ARG(__VA_ARGS__, DUMMY_TOKEN)
//     -> MAKE_GENERATOR_IMPL_FIRST_ARG(NAME, SEED, DUMMY_TOKEN)
//     -> NAME
//   MAKE_GENERATOR_IMPL_SECOND_ARG(
//       __VA_ARGS__, std::random_device{}(), DUMMY_TOKEN)
//     -> MAKE_GENERATOR_IMPL_SECOND_ARG(
//            NAME, SEED, std::random_device{}(), DUMMY_TOKEN)
//     -> SEED
//   So we create NAME with seed SEED.
//
//   In this case the DUMMY_TOKEN is not necessary.
#define MAKE_GENERATOR(...)                                             \
  std::mt19937 MAKE_GENERATOR_IMPL_FIRST_ARG(__VA_ARGS__, DUMMY_TOKEN); \
  /* Capture everything because we don't know what passed seed uses */  \
  INFO("Seed is: " << [&]() {                                           \
    const auto MAKE_GENERATOR_seed = (MAKE_GENERATOR_IMPL_SECOND_ARG(   \
        __VA_ARGS__, std::random_device{}(), DUMMY_TOKEN));             \
    MAKE_GENERATOR_IMPL_FIRST_ARG(__VA_ARGS__, DUMMY_TOKEN)             \
        .seed(MAKE_GENERATOR_seed);                                     \
    return MakeString{} << MAKE_GENERATOR_seed << " from " __FILE__ ":" \
                        << __LINE__;                                    \
  }())

/*!
 * \ingroup TestingFrameworkGroup
 * \brief A wrapper around Catch's CHECK macro that checks approximate equality
 * of each entry in each tag within a variables.
 */
#define CHECK_VARIABLES_APPROX(a, b)                                         \
  do {                                                                       \
    INFO(__FILE__ ":" + std::to_string(__LINE__) + ": " #a " == " #b);       \
    check_variables_approx<std::common_type_t<                               \
        std::decay_t<decltype(a)>, std::decay_t<decltype(b)>>>::apply(a, b); \
  } while (false)

/*!
 * \ingroup TestingFrameworkGroup
 *  \brief Same as `CHECK_VARIABLES_APPROX`, but with a user-defined Approx.
 *  The third argument should be of type `Approx`.
 */
#define CHECK_VARIABLES_CUSTOM_APPROX(a, b, appx)                            \
  do {                                                                       \
    INFO(__FILE__ ":" + std::to_string(__LINE__) + ": " #a " == " #b);       \
    check_variables_approx<std::common_type_t<                               \
        std::decay_t<decltype(a)>, std::decay_t<decltype(b)>>>::apply(a, b,  \
                                                                      appx); \
  } while (false)

template <typename Tag, typename TagList>
const auto& extract_value_for_variables_comparison(
    const Variables<TagList>& input_vars, std::true_type /*meta*/) {
  // only Scalars of spin-weighted quantities are currently allowed. If that
  // changes, then this solution will no longer be workable.
  return get(get<Tag>(input_vars)).data();
}

template <typename Tag, typename TagList>
const auto& extract_value_for_variables_comparison(
    const Variables<TagList>& input_vars, std::false_type /*meta*/) {
  return get<Tag>(input_vars);
}

template <typename T>
struct check_variables_approx;

template <typename TagList>
struct check_variables_approx<Variables<TagList>> {
  // clang-tidy: non-const reference
  static void apply(const Variables<TagList>& a, const Variables<TagList>& b,
                    Approx& appx = approx) {  // NOLINT
    tmpl::for_each<TagList>([&a, &b, &appx](auto x) {
      using Tag = typename decltype(x)::type;
      INFO(db::tag_name<Tag>());
      const auto& a_val = extract_value_for_variables_comparison<Tag>(
          a, is_any_spin_weighted<typename Tag::type::type>{});
      const auto& b_val = extract_value_for_variables_comparison<Tag>(
          b, is_any_spin_weighted<typename Tag::type::type>{});
      CHECK_ITERABLE_CUSTOM_APPROX(a_val, b_val, appx);
    });
  }
};

// an additional overload for SpinWeighted quantities that just forwards to
// checking on the wrapped `data()` for the SpinWeighted
template <typename T>
struct check_iterable_approx<T, Requires<is_any_spin_weighted_v<T>>> {
  // clang-tidy: non-const reference
  static void apply(const T& a, const T& b, Approx& appx = approx) {  // NOLINT
    check_iterable_approx<typename T::value_type>::apply(a.data(), b.data(),
                                                         appx);
  }
};
