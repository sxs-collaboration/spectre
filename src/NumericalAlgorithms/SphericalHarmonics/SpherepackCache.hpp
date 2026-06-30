// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>
#include <vector>

namespace ylm {
class Spherepack;

/// Retrieves a cached Spherepack object with m_max equal to l_max
const Spherepack& get_spherepack_cache(size_t l_max);

/*!
 * \brief SPHEREPACK spectral offsets grouped by angular degree \f$\ell\f$.
 *
 * \details `offsets` is a flat array of SPHEREPACK spectral offsets sorted
 * by angular degree. The offsets for degree \f$\ell\f$ occupy the half-open
 * range `[l_start[l], l_start[l+1])` within `offsets`.
 */
struct SphericalHarmonicModesByDegree {
  /// Flat list of SPHEREPACK spectral offsets, grouped by angular degree
  std::vector<size_t> offsets;
  /// Prefix-sum array of length \f$\ell_\mathrm{max}+2\f$; the offsets for
  /// degree \f$\ell\f$ are `offsets[l_start[l]..l_start[l+1])`
  std::vector<size_t> l_start;
};

/// Retrieves cached SPHEREPACK spectral offsets grouped by angular degree for
/// the given \p l_max.
const SphericalHarmonicModesByDegree& get_modes_by_degree_cache(size_t l_max);
}  // namespace ylm
