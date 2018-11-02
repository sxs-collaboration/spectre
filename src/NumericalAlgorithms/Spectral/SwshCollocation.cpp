// Distributed under the MIT License.
// See LICENSE.txt for details.

#include <cmath>
#include <ostream>
#include <sharp_cxx.h>
#include <type_traits>
#include <utility>

#include "ErrorHandling/Assert.hpp"
#include "ErrorHandling/Error.hpp"
#include "NumericalAlgorithms/Spectral/ComplexDataView.hpp"
#include "NumericalAlgorithms/Spectral/SwshCollocation.hpp"
#include "Utilities/ForceInline.hpp"
#include "Utilities/TMPL.hpp"

namespace Spectral {
namespace Swsh {

template <ComplexRepresentation Representation>
Collocation<Representation>::Collocation(const size_t l_max) noexcept
    : l_max_{l_max} {
  sharp_geom_info* geometry_to_initialize;
  sharp_make_gauss_geom_info(
      l_max_ + 1, 2 * l_max_ + 1, 0.0,
      detail::ComplexDataView<Representation>::stride(),
      detail::ComplexDataView<Representation>::stride() * (1 * l_max_ + 1),
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

namespace detail {
// We use a `std::index_sequence` to loop over the function lambdas
// for the cases we want to evaluate in order to only compute and store
// a specific `l_max` that is requested, not all values up to
// `collocation_maximum_l_max`. This cannot be done in the more traditional
// way of either having a `std::vector` or `std::undordered_map` that is
// `static` because only the construction phase is guaranteed to be thread-safe,
// not the access. This would mean were we to use a `std::vector` or
// `std::unordered_map` we would need to compute all values up to
// `collocation_maximum_l_max` immediately. By doing the loop over `Is` we
// are able to generate `sizeof...(Is)` different static objects and thus each
// one can be constructed only when it is needed and in a thread-safe manner.

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
  const Collocation<Representation>* collocation = nullptr;
  auto get_collocation_if_match = [&collocation, &index ](auto i) noexcept {
    if (UNLIKELY(decltype(i)::value == index)) {
      static const Collocation<Representation> precomputed_collocation =
          Collocation<Representation>{decltype(i)::value};
      collocation = &precomputed_collocation;
    }
    return 0;
  };
  expand_pack(
      get_collocation_if_match(std::integral_constant<size_t, Is>{})...);
  // clang-tidy warns about null reference return, but will only return null in
  // the erroring execution path.
  return *collocation;  // NOLINT
}
}  // namespace detail

template <ComplexRepresentation Representation>
const Collocation<Representation>& precomputed_spherical_harmonic_collocation(
    const size_t l_max) noexcept {
  return detail::dispatch_to_precomputed_static_collocation_impl<
      Representation>(
      l_max, std::make_index_sequence<collocation_maximum_l_max + 1>{});
}

template class Collocation<ComplexRepresentation::Interleaved>;
template class Collocation<ComplexRepresentation::RealsThenImags>;

template const Collocation<ComplexRepresentation::Interleaved>&
precomputed_spherical_harmonic_collocation(const size_t l_max) noexcept;
template const Collocation<ComplexRepresentation::RealsThenImags>&
precomputed_spherical_harmonic_collocation(const size_t l_max) noexcept;

}  // namespace Swsh
}  // namespace Spectral
