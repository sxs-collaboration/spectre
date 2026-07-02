// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "NumericalAlgorithms/SphericalHarmonics/SpherepackCache.hpp"

#include <cstddef>
#include <vector>

#include "NumericalAlgorithms/SphericalHarmonics/Spherepack.hpp"
#include "NumericalAlgorithms/SphericalHarmonics/SpherepackIterator.hpp"
#include "Utilities/StaticCache.hpp"

namespace ylm {
const Spherepack& get_spherepack_cache(const size_t l_max) {
  static const auto spherepack_cache = make_static_cache<CacheRange<2, 151>>(
      [](const size_t l) { return ylm::Spherepack(l, l); });
  return spherepack_cache(l_max);
}

const SphericalHarmonicModesByDegree& get_modes_by_degree_cache(
    const size_t l_max) {
  // Note the arbitrary max cache size
  static const auto cache =
      make_static_cache<CacheRange<2, 151>>([](const size_t l) {
        SphericalHarmonicModesByDegree result;
        result.l_start.assign(l + 2, 0);

        // Count modes per degree
        SpherepackIterator iter{l, l};
        while (iter) {
          ++result.l_start[iter.l() + 1];
          ++iter;
        }
        // Build prefix sums
        for (size_t ell = 1; ell <= l + 1; ++ell) {
          result.l_start[ell] += result.l_start[ell - 1];
        }
        // Fill offsets using write positions initialised from the prefix sums
        result.offsets.resize(result.l_start[l + 1]);
        std::vector<size_t> write_pos(
            result.l_start.begin(),
            result.l_start.begin() + static_cast<std::ptrdiff_t>(l + 1));
        iter.reset();
        while (iter) {
          result.offsets[write_pos[iter.l()]++] = iter();
          ++iter;
        }
        return result;
      });
  return cache(l_max);
}
}  // namespace ylm
