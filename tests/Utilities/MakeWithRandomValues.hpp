// Distributed under the MIT License.
// See LICENSE.txt for details.

/// \file
/// Helper functions for data structures used in unit tests

#pragma once

#include <complex>
#include <cstddef>  // for std::nullptr_t
#include <limits>
#include <random>

#include "DataStructures/SpinWeighted.hpp"
#include "DataStructures/Variables.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/MakeWithValue.hpp"
#include "Utilities/Requires.hpp"
#include "Utilities/TypeTraits.hpp"

namespace TestHelpers_detail {
/// \cond HIDDEN_SYMBOLS
template <typename T, typename = std::nullptr_t>
struct FillWithRandomValuesImpl;

template <typename T>
struct FillWithRandomValuesImpl<T, Requires<cpp17::is_fundamental_v<T>>> {
  template <typename UniformRandomBitGenerator,
            typename RandomNumberDistribution>
  static void apply(
      const gsl::not_null<T*> data,
      const gsl::not_null<UniformRandomBitGenerator*> generator,
      const gsl::not_null<RandomNumberDistribution*> distribution) noexcept {
    static_assert(
        cpp17::is_same_v<T, typename RandomNumberDistribution::result_type>,
        "Mismatch between data type and random number type.");
    *data = (*distribution)(*generator);
  }
};

template <typename T>
struct FillWithRandomValuesImpl<std::complex<T>> {
  template <typename UniformRandomBitGenerator,
            typename RandomNumberDistribution>
  static void apply(
      const gsl::not_null<std::complex<T>*> data,
      const gsl::not_null<UniformRandomBitGenerator*> generator,
      const gsl::not_null<RandomNumberDistribution*> distribution) noexcept {
    static_assert(
        cpp17::is_same_v<T, typename RandomNumberDistribution::result_type>,
        "Mismatch between data type and random number type.");
    data->real((*distribution)(*generator));
    data->imag((*distribution)(*generator));
  }
};

template <typename T, int Spin>
struct FillWithRandomValuesImpl<SpinWeighted<T, Spin>> {
  template <typename UniformRandomBitGenerator,
            typename RandomNumberDistribution>
  static void apply(
      const gsl::not_null<SpinWeighted<T, Spin>*> data,
      const gsl::not_null<UniformRandomBitGenerator*> generator,
      const gsl::not_null<RandomNumberDistribution*> distribution) noexcept {
    FillWithRandomValuesImpl<T>::apply(&(data->data()), generator,
                                       distribution);
  }
};

template <typename T>
struct FillWithRandomValuesImpl<
    T, Requires<not tt::is_maplike_v<T> and tt::is_iterable_v<T>>> {
  template <typename UniformRandomBitGenerator,
            typename RandomNumberDistribution>
  static void apply(
      const gsl::not_null<T*> data,
      const gsl::not_null<UniformRandomBitGenerator*> generator,
      const gsl::not_null<RandomNumberDistribution*> distribution) noexcept {
    for (auto& d : *data) {
      FillWithRandomValuesImpl<std::decay_t<decltype(d)>>::apply(&d, generator,
                                                                 distribution);
    }
  }
};

template <typename... Tags>
struct FillWithRandomValuesImpl<Variables<tmpl::list<Tags...>>,
                                std::nullptr_t> {
  template <typename UniformRandomBitGenerator,
            typename RandomNumberDistribution>
  static void apply(
      const gsl::not_null<Variables<tmpl::list<Tags...>>*> data,
      const gsl::not_null<UniformRandomBitGenerator*> generator,
      const gsl::not_null<RandomNumberDistribution*> distribution) noexcept {
    expand_pack(
        (FillWithRandomValuesImpl<std::decay_t<decltype(get<Tags>(*data))>>::
             apply(&get<Tags>(*data), generator, distribution),
         cpp17::void_type{})...);
  }
};
/// \endcond
}  // namespace TestHelpers_detail

// {@
/// \ingroup TestingFrameworkGroup
///
/// \brief A uniform distribution function object which redirects appropriately
/// to either the `std::uniform_int_distribution` or the
/// `std::uniform_real_distribution`. This also provides a convenience
/// constructor which takes a 2-element array for the bounds for either
/// floating point or int distributions.
template <typename T>
class UniformCustomDistribution
    : public tmpl::conditional_t<
          cpp17::is_integral_v<T>,
          std::uniform_int_distribution<std::remove_const_t<T>>,
          std::uniform_real_distribution<std::remove_const_t<T>>> {
  using base = tmpl::conditional_t<
      cpp17::is_integral_v<T>,
      std::uniform_int_distribution<std::remove_const_t<T>>,
      std::uniform_real_distribution<std::remove_const_t<T>>>;
  static_assert(cpp17::is_integral_v<T> or cpp17::is_floating_point_v<T>,
                "UniformCustomDistribution currently supports only floating"
                "point and integral values");

 public:
  using base::base;
  template <typename Bound>
  explicit UniformCustomDistribution(std::array<Bound, 2> arr) noexcept
      : base(arr[0], arr[1]) {}
  using base::operator=;
  using base::operator();
};
// @}

/// \ingroup TestingFrameworkGroup
/// \brief Fill an existing data structure with random values
template <typename T, typename UniformRandomBitGenerator,
          typename RandomNumberDistribution>
void fill_with_random_values(
    const gsl::not_null<T*> data,
    const gsl::not_null<UniformRandomBitGenerator*> generator,
    const gsl::not_null<RandomNumberDistribution*> distribution) noexcept {
  TestHelpers_detail::FillWithRandomValuesImpl<T>::apply(data, generator,
                                                         distribution);
}

// @{
/// \ingroup TestingFrameworkGroup
/// \brief Make a data structure and fill it with random values
///
/// \details Given an object of type `T`, create an object of type `ReturnType`
/// whose elements are initialized to random values using the given random
/// number generator and random number distribution.
///
/// \requires the type `ReturnType` to be creatable using
/// `make_with_value<ReturnType>(T)`
template <typename ReturnType, typename T, typename UniformRandomBitGenerator,
          typename RandomNumberDistribution>
ReturnType make_with_random_values(
    const gsl::not_null<UniformRandomBitGenerator*> generator,
    const gsl::not_null<RandomNumberDistribution*> distribution,
    const T& used_for_size) noexcept {
  auto result = make_with_value<ReturnType>(
      used_for_size,
      std::numeric_limits<
          tt::get_fundamental_type_t<ReturnType>>::signaling_NaN());
  fill_with_random_values(make_not_null(&result), generator, distribution);
  return result;
}
// distributions are sufficiently small to justify providing a convenience
// function that receives them by value, which is useful when obtaining pointers
// is inconvenient (e.g. for distributions that are obtained as
// rvalues). Generators should never be copied, as doing so will cause
// duplication of the pseudorandom numbers and performance hits due to the
// nontrivial size.
// clang-tidy: seems to erroneously believe this is a function declaration
// rather than a definition.
template <typename ReturnType, typename T, typename UniformRandomBitGenerator,
          typename RandomNumberDistribution>
ReturnType make_with_random_values(
    const gsl::not_null<UniformRandomBitGenerator*> generator,  // NOLINT
    RandomNumberDistribution distribution, const T& used_for_size) noexcept {
  return make_with_random_values<ReturnType>(
      generator, make_not_null(&distribution), used_for_size);
}
// @}

/// \ingroup TestingFrameworkGroup
/// \brief Make a fixed-size data structure and fill with random values
///
/// \details Given a template argument type `T`, create an object of the same
/// type, fills it with random values, and returns the result. Acts as a
/// convenience function to avoid users needing to put in constructors with
/// `signaling_NaN()`s or `max()`s themselves when making with random values.
/// Used as
/// `make_with_random_values<Type>(make_not_null(&gen),make_not_null(&dist))`
template <typename T, typename UniformRandomBitGenerator,
          typename RandomNumberDistribution>
T make_with_random_values(
    const gsl::not_null<UniformRandomBitGenerator*> generator,
    const gsl::not_null<RandomNumberDistribution*> distribution) noexcept {
  T result{};
  fill_with_random_values(make_not_null(&result), generator, distribution);
  return result;
}
