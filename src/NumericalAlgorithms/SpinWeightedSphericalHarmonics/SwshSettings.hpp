// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

namespace Spectral {
namespace Swsh {

// In the static caching mechanism, we permit an l_max up to this setting
// value. `CollocationMetadata` can still be constructed manually above this
// value.
constexpr size_t collocation_maximum_l_max = 200;

namespace detail {

// currently, the operators for the swsh derivatives and for the coefficients
// are cached to the same maximum l_max as the collocations, but it could
// conceivably be useful to have distinct representation maximums for them, so
// they have separate control parameters.
constexpr size_t swsh_derivative_maximum_l_max = collocation_maximum_l_max;
constexpr size_t coefficients_maximum_l_max = collocation_maximum_l_max;
}  // namespace detail
}  // namespace Swsh
}  // namespace Spectral
