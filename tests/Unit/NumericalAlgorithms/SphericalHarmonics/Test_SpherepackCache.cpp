// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <algorithm>
#include <cstddef>
#include <vector>

#include "NumericalAlgorithms/SphericalHarmonics/SpherepackCache.hpp"
#include "NumericalAlgorithms/SphericalHarmonics/SpherepackIterator.hpp"

namespace {
void test_modes_by_degree(const size_t l_max) {
  const auto& modes = ylm::get_modes_by_degree_cache(l_max);

  // l_start has length l_max+2 and begins at zero
  CHECK(modes.l_start.size() == l_max + 2);
  CHECK(modes.l_start[0] == 0);

  // Each degree l has exactly 2*l+1 modes (one m=0 entry plus two for each
  // m=1..l), which is the standard count for m_max == l_max with
  // zero_m_is_real=true.
  for (size_t l = 0; l <= l_max; ++l) {
    CHECK(modes.l_start[l + 1] - modes.l_start[l] == 2 * l + 1);
  }

  // Total mode count matches the number of entries visited by
  // SpherepackIterator
  ylm::SpherepackIterator iter{l_max, l_max};
  size_t total_modes = 0;
  while (iter) {
    ++total_modes;
    ++iter;
  }
  CHECK(modes.offsets.size() == total_modes);
  CHECK(modes.l_start[l_max + 1] == total_modes);

  // Build an offset->l reference map from the iterator.
  // Use l_max+1 as a sentinel for "not a valid mode offset".
  std::vector<size_t> offset_to_l(iter.spherepack_array_size(), l_max + 1);
  iter.reset();
  while (iter) {
    offset_to_l[iter()] = iter.l();
    ++iter;
  }

  // Every entry in offsets[l_start[l]..l_start[l+1]) has degree l
  for (size_t l = 0; l <= l_max; ++l) {
    for (size_t k = modes.l_start[l]; k < modes.l_start[l + 1]; ++k) {
      CHECK(offset_to_l[modes.offsets[k]] == l);
    }
  }

  // The full set of offsets matches the iterator exactly (no duplicates, no
  // gaps)
  std::vector<size_t> sorted_cache_offsets = modes.offsets;
  std::sort(sorted_cache_offsets.begin(), sorted_cache_offsets.end());

  std::vector<size_t> sorted_iter_offsets;
  sorted_iter_offsets.reserve(total_modes);
  iter.reset();
  while (iter) {
    sorted_iter_offsets.push_back(iter());
    ++iter;
  }
  std::sort(sorted_iter_offsets.begin(), sorted_iter_offsets.end());

  CHECK(sorted_cache_offsets == sorted_iter_offsets);
}

void test_cache_identity() {
  // Same l_max must return the same object
  CHECK(&ylm::get_modes_by_degree_cache(4) ==
        &ylm::get_modes_by_degree_cache(4));
  // Different l_max must return different objects
  CHECK(&ylm::get_modes_by_degree_cache(2) !=
        &ylm::get_modes_by_degree_cache(4));
}
}  // namespace

SPECTRE_TEST_CASE("Unit.SphericalHarmonics.SpherepackCache",
                  "[NumericalAlgorithms][Unit]") {
  // Testing the SphericalHarmonicModesByDegree cache, NOT the Spherepack cache
  test_modes_by_degree(2);
  test_modes_by_degree(5);
  test_modes_by_degree(10);
  test_cache_identity();
}
