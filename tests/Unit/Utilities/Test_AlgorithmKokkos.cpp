// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <array>
#include <cstddef>

#include "Helpers/Utilities/Kokkos/CheckOnDevice.hpp"
#include "Utilities/Algorithm.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/Kokkos/KokkosCore.hpp"

namespace {
// Calls every function that can be called in Kokkos kernels in a kernel
void test_on_device() {
  TestHelpers::check_on_device(KOKKOS_LAMBDA() {
    std::array<int, 6> a{{3, 1, 4, 1, 5, 9}};
    const std::array<int, 6> b{{3, 1, 4, 1, 5, 9}};
    const auto odd = [](const int x) { return x % 2 == 1; };
    std::array<int, 40> r{};
    r[0] = alg::all_of(a, [](const int x) { return x > 0; }) ? 2 : 1;
    r[1] = alg::any_of(a, [](const int x) { return x > 8; }) ? 2 : 1;
    r[2] = alg::none_of(a, [](const int x) { return x > 9; }) ? 2 : 1;
    r[3] = static_cast<int>(alg::count(a, 1));
    r[4] = static_cast<int>(alg::count_if(a, odd));
    r[5] = static_cast<int>(alg::find(a, 4) - a.begin());
    r[6] = static_cast<int>(alg::find_if(a, [](const int x) { return x > 4; }) -
                            a.begin());
    r[7] = static_cast<int>(alg::find_if_not(a, odd) - a.begin());
    r[8] = alg::found(a, 5) ? 2 : 1;
    r[9] = alg::found_if(a, [](const int x) { return x > 9; }) ? 2 : 1;
    r[10] = alg::found_if_not(a, odd) ? 2 : 1;
    int sum = 0;
    alg::for_each(a, [&sum](const int x) { sum += x; });
    r[11] = sum;
    r[12] = alg::equal(a, b) ? 2 : 1;
    r[13] = alg::equal(a, b, [](const int x, const int y) { return x < y; })
                ? 2
                : 1;
    r[14] = *alg::min_element(a);
    r[15] = *alg::max_element(a);
    const auto [min, max] = alg::minmax_element(a);
    r[16] = static_cast<int>(min - a.begin()) + 10;
    r[17] = static_cast<int>(max - a.begin());
    std::array<int, 6> transformed{};
    alg::transform(a, b, transformed.begin(),
                   [](const int x, const int y) { return x * y; });
    r[18] = transformed[5];
    alg::transform(a, transformed.begin(), [](const int x) { return x + 1; });
    r[19] = transformed[0];
    auto removed = a;
    r[20] = static_cast<int>(alg::remove(removed, 1) - removed.begin());
    r[21] = removed[1];
    removed = a;
    r[22] = static_cast<int>(alg::remove_if(removed, odd) - removed.begin());
    r[23] = removed[0];
    alg::bubble_sort(a);
    for (size_t i = 0; i < 6; ++i) {
      gsl::at(r, 24 + i) = gsl::at(a, i);
    }
    alg::bubble_sort(a, [](const int x, const int y) { return x > y; });
    for (size_t i = 0; i < 6; ++i) {
      gsl::at(r, 30 + i) = gsl::at(a, i);
    }
    std::array<int, 4> permutation{{1, 2, 4, 3}};
    cpp20::next_permutation(permutation.begin(), permutation.end());
    r[36] = permutation[1] * 100 + permutation[2] * 10 + permutation[3];
    cpp20::reverse(permutation.begin(), permutation.end());
    r[37] = permutation[0];
    int x = 7;
    int y = 8;
    cpp20::swap(x, y);
    r[38] = x;
    cpp20::iter_swap(permutation.begin(),
                     alg::find(permutation, permutation[1]));
    r[39] = permutation[0];
    return r;
  });
}
}  // namespace

SPECTRE_TEST_CASE("Unit.Utilities.Algorithm.Kokkos", "[Unit][Utilities]") {
  test_on_device();
}
