// Distributed under the MIT License.
// See LICENSE.txt for details.

/// \file
/// Helper functions for data structures used in unit tests

#pragma once

#include <cstddef>  // for std::nullptr_t
#include <limits>
#include <random>

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
      used_for_size, std::numeric_limits<double>::signaling_NaN());
  fill_with_random_values(make_not_null(&result), generator, distribution);
  return result;
}
