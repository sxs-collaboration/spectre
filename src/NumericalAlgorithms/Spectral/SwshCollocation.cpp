// Distributed under the MIT License.
// See LICENSE.txt for details.

#include <array>
#include <cmath>
#include <ostream>
#include <sharp_cxx.h>
#include <utility>

#include "NumericalAlgorithms/Spectral/ComplexDataView.hpp"
#include "NumericalAlgorithms/Spectral/SwshCollocation.hpp"
#include "Utilities/ErrorHandling/Assert.hpp"
#include "Utilities/ErrorHandling/Error.hpp"
#include "Utilities/ForceInline.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/StaticCache.hpp"

namespace Spectral {
namespace Swsh {

template <ComplexRepresentation Representation>
CollocationMetadata<Representation>::CollocationMetadata(
    const size_t l_max) noexcept
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
double CollocationMetadata<Representation>::theta(const size_t offset) const
    noexcept {
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
double CollocationMetadata<Representation>::phi(const size_t offset) const
    noexcept {
  ASSERT(offset < (2 * l_max_ + 1) * (l_max_ + 1),
         "invalid offset " << offset
                           << " passed to phi lookup. Must be less than (2 * "
                              "l_max + 1) * (l_max + 1) = "
                           << (2 * l_max_ + 1) * (l_max_ + 1));
  return 2.0 * M_PI * ((offset % (2 * l_max_ + 1)) / (2.0 * l_max_ + 1.0));
}

template <ComplexRepresentation Representation>
const CollocationMetadata<Representation>& cached_collocation_metadata(
    const size_t l_max) noexcept {
  const static auto lazy_collocation_cache =
      make_static_cache<CacheRange<0, collocation_maximum_l_max>>(
          [](const size_t generator_l_max) noexcept {
            return CollocationMetadata<Representation>{generator_l_max};
          });
  return lazy_collocation_cache(l_max);
}

template class CollocationMetadata<ComplexRepresentation::Interleaved>;
template class CollocationMetadata<ComplexRepresentation::RealsThenImags>;

template const CollocationMetadata<ComplexRepresentation::Interleaved>&
cached_collocation_metadata(const size_t l_max) noexcept;
template const CollocationMetadata<ComplexRepresentation::RealsThenImags>&
cached_collocation_metadata(const size_t l_max) noexcept;

}  // namespace Swsh
}  // namespace Spectral
