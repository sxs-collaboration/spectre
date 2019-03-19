// Distributed under the MIT License.
// See LICENSE.txt for details.

#include <array>
#include <cmath>
#include <ostream>
#include <sharp_cxx.h>
#include <utility>

#include "ErrorHandling/Assert.hpp"
#include "ErrorHandling/Error.hpp"
#include "NumericalAlgorithms/Spectral/ComplexDataView.hpp"
#include "NumericalAlgorithms/Spectral/SwshCollocation.hpp"
#include "Utilities/ForceInline.hpp"
#include "Utilities/Gsl.hpp"

namespace Spectral {
namespace Swsh {

template <ComplexRepresentation Representation>
Collocation<Representation>::Collocation(const size_t l_max) noexcept
    : l_max_{l_max} {
  sharp_geom_info* geometry_to_initialize;
  sharp_make_gauss_geom_info(
      l_max_ + 1, 2 * l_max_ + 1, 0.0,
      detail::ComplexDataView<Representation>::stride(),
      detail::ComplexDataView<Representation>::stride() * (2 * l_max_ + 1),
      &geometry_to_initialize);
  geom_info_.reset(geometry_to_initialize);
}

template <ComplexRepresentation Representation>
double Collocation<Representation>::theta(const size_t offset) const noexcept {
  ASSERT(offset < (2 * l_max_ + 1) * (l_max_ + 1),
         "invalid offset " << offset
                           << " passed to phi lookup. Must be less than (2 * "
                              "l_max + 1) * (l_max + 1) = "
                           << (2 * l_max_ + 1) * (l_max_ + 1));
  // clang-tidy pointer arithmetic
  if (offset < (2 * l_max_ + 1) * (l_max_ / 2 + 1)) {
    return (geom_info_.get())  // NOLINT
        ->pair[offset / (2 * l_max_ + 1)]
        .r1.theta;  // NOLINT
  } else {
    return (geom_info_.get())                       // NOLINT
        ->pair[l_max_ - offset / (2 * l_max_ + 1)]  // NOLINT
        .r2.theta;
  }
}

template <ComplexRepresentation Representation>
double Collocation<Representation>::phi(const size_t offset) const noexcept {
  ASSERT(offset < (2 * l_max_ + 1) * (l_max_ + 1),
         "invalid offset " << offset
                           << " passed to phi lookup. Must be less than (2 * "
                              "l_max + 1) * (l_max + 1) = "
                           << (2 * l_max_ + 1) * (l_max_ + 1));
  return 2.0 * M_PI * ((offset % (2 * l_max_ + 1)) / (2.0 * l_max_ + 1.0));
}

namespace {
// We use a `std::index_sequence` to generate functions that cache the
// collocation info for all the l's up to l_max. Each element in the cache is
// not computed until it is retrieved in order to reduce the overall memory
// footprint. However, when doing so we still need to guarantee thread-safety.
// static variables are guaranteed to be thread-safe only on construction and so
// we store a std::array of function pointers to functions (cache_impl) that
// then have a `static Collocation` that they return and is constructed lazily
// in a thread-safe manner.
template <ComplexRepresentation Representation, size_t I>
const Collocation<Representation>& cache_impl() noexcept {
  static const Collocation<Representation> precomputed_collocation{I};
  return precomputed_collocation;
}

template <ComplexRepresentation Representation, size_t... Is>
SPECTRE_ALWAYS_INLINE const Collocation<Representation>&
dispatch_to_precomputed_static_collocation_impl(
    const size_t index, std::index_sequence<Is...> /*meta*/) noexcept {
  if (UNLIKELY(index > collocation_maximum_l_max)) {
    ERROR("The provided l_max "
          << index
          << "is not below the maximum l_max to cache, which is currently "
          << collocation_maximum_l_max
          << ". Either "
             "construct the Collocation manually, or consider (with caution) "
             "increasing `collocation_maximum_l_max`.");
  }
  static const std::array<const Collocation<Representation>& (*)(),
                          sizeof...(Is)>
      cache{{&cache_impl<Representation, Is>...}};
  return gsl::at(cache, index)();
}
}  // namespace

template <ComplexRepresentation Representation>
const Collocation<Representation>& precomputed_collocation(
    const size_t l_max) noexcept {
  return dispatch_to_precomputed_static_collocation_impl<Representation>(
      l_max, std::make_index_sequence<collocation_maximum_l_max + 1>{});
}

template class Collocation<ComplexRepresentation::Interleaved>;
template class Collocation<ComplexRepresentation::RealsThenImags>;

template const Collocation<ComplexRepresentation::Interleaved>&
precomputed_collocation(const size_t l_max) noexcept;
template const Collocation<ComplexRepresentation::RealsThenImags>&
precomputed_collocation(const size_t l_max) noexcept;

}  // namespace Swsh
}  // namespace Spectral
