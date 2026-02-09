// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>

#include "NumericalAlgorithms/Spectral/MaximumNumberOfPoints.hpp"
#include "NumericalAlgorithms/Spectral/MinimumNumberOfPoints.hpp"
#include "NumericalAlgorithms/Spectral/Parity.hpp"
#include "Utilities/ErrorHandling/Assert.hpp"
#include "Utilities/StaticCache.hpp"

/// \cond
namespace Spectral {
enum class Basis : uint8_t;
enum class Quadrature : uint8_t;
}  // namespace Spectral
/// \endcond

namespace Spectral::detail {
template <Basis BasisType, Quadrature QuadratureType,
          typename SpectralQuantityGenerator>
const auto& precomputed_spectral_quantity(const size_t num_points) {
  constexpr size_t max_num_points =
      Spectral::maximum_number_of_points<BasisType>;
  constexpr size_t min_num_points =
      Spectral::minimum_number_of_points<BasisType, QuadratureType>;
  ASSERT(num_points >= min_num_points,
         "Tried to work with less than the minimum number of collocation "
         "points for this quadrature.");
  ASSERT(num_points <= max_num_points,
         "Exceeded maximum number of collocation points.");
  // We compute the quantity for all possible `num_point`s the first time this
  // function is called and keep the data around for the lifetime of the
  // program. The computation is handled by the call operator of the
  // `SpectralQuantityType` instance.
  static const auto precomputed_data =
      make_static_cache<CacheRange<min_num_points, max_num_points + 1>>(
          SpectralQuantityGenerator{});
  return precomputed_data(num_points);
}

template <Basis BasisType, Quadrature QuadratureType,
          typename SpectralQuantityGenerator>
const auto& precomputed_spectral_quantity_with_parity(const size_t num_points,
                                                      const Parity parity) {
  constexpr size_t max_num_points =
      Spectral::maximum_number_of_points<BasisType>;
  constexpr size_t min_num_points =
      Spectral::minimum_number_of_points<BasisType, QuadratureType>;
  ASSERT(num_points >= min_num_points,
         "Tried to work with less than the minimum number of collocation "
         "points for this quadrature.");
  ASSERT(num_points <= max_num_points,
         "Exceeded maximum number of collocation points.");
  ASSERT(parity != Parity::Uninitialized,
         "Tried to use a parity-based function without a definite parity");
  // We compute the quantity for all possible `num_point`s the first time this
  // function is called and keep the data around for the lifetime of the
  // program. The computation is handled by the call operator of the
  // `SpectralQuantityType` instance.
  static const auto precomputed_data =
      make_static_cache<CacheRange<min_num_points, max_num_points + 1>,
                        CacheEnumeration<Parity, Parity::Even, Parity::Odd>>(
          SpectralQuantityGenerator{});
  return precomputed_data(num_points, parity);
}

template <Basis BasisType, Quadrature QuadratureType,
          typename SpectralQuantityGenerator>
const auto& precomputed_two_indexed_spectral_quantity(const size_t num_points,
                                           const size_t m, const size_t N) {
  constexpr size_t max_num_points =
      Spectral::maximum_number_of_points<BasisType>;
  constexpr size_t min_num_points =
      Spectral::minimum_number_of_points<BasisType, QuadratureType>;
  ASSERT(num_points >= min_num_points,
         "Tried to work with less than the minimum number of collocation "
         "points for this quadrature.");
  ASSERT(num_points <= max_num_points,
         "Exceeded maximum number of collocation points.");
  // We compute the quantity for all possible `num_point`s the first time this
  // function is called and keep the data around for the lifetime of the
  // program. The computation is handled by the call operator of the
  // `SpectralQuantityType` instance.
  static const auto precomputed_data =
      make_static_cache<CacheRange<min_num_points, max_num_points + 1>,
                        CacheRange<0_st, 2 * max_num_points - 1>,
                        CacheRange<0_st, 2 * max_num_points - 1>>(
          SpectralQuantityGenerator{});
  return precomputed_data(num_points, m, N);
}
}  // namespace Spectral::detail

// clang-tidy: Macro arguments should be in parentheses, but we want to append
// template parameters here.
#define PRECOMPUTED_SPECTRAL_QUANTITY(function_name, return_type, \
                                      generator_name)             \
  template <Basis BasisType, Quadrature QuadratureType>           \
  const return_type& function_name(const size_t num_points) {     \
    return Spectral::detail::precomputed_spectral_quantity<       \
        BasisType, QuadratureType,                                \
        generator_name<BasisType, QuadratureType>>(/* NOLINT */   \
                                                   num_points);   \
  }
#define PRECOMPUTED_SPECTRAL_QUANTITY_WITH_PARITY(function_name, return_type, \
                                                  generator_name)             \
  template <Basis BasisType, Quadrature QuadratureType>                       \
  const return_type& function_name(const size_t num_points,                   \
                                   const Parity parity) {                     \
    return Spectral::detail::precomputed_spectral_quantity_with_parity<       \
        BasisType, QuadratureType,                                            \
        generator_name<BasisType, QuadratureType>>(/* NOLINT */               \
                                                   num_points, parity);       \
  }
#define PRECOMPUTED_TWO_INDEXED_SPECTRAL_QUANTITY(function_name, return_type, \
                                                  generator_name)             \
  template <Basis BasisType, Quadrature QuadratureType>                       \
  const return_type& function_name(const size_t num_points, const size_t m,   \
                                   const size_t N) {                          \
    return Spectral::detail::precomputed_two_indexed_spectral_quantity<       \
        BasisType, QuadratureType,                                            \
        generator_name<BasisType, QuadratureType>>(/* NOLINT */               \
                                                   num_points, m, N);         \
  }
