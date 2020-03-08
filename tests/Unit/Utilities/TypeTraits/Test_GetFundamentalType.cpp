// Distributed under the MIT License.
// See LICENSE.txt for details.

#include <array>
#include <complex>
#include <vector>

#include "Utilities/TypeTraits.hpp"
#include "Utilities/TypeTraits/GetFundamentalType.hpp"

/// [get_fundamental_type]
static_assert(
    cpp17::is_same_v<
        typename tt::get_fundamental_type<std::array<double, 2>>::type, double>,
    "Failed testing get_fundamental_type");
static_assert(
    cpp17::is_same_v<
        typename tt::get_fundamental_type_t<std::vector<std::complex<int>>>,
        int>,
    "Failed testing get_fundamental_type");
static_assert(cpp17::is_same_v<typename tt::get_fundamental_type_t<int>, int>,
              "Failed testing get_fundamental_type");
/// [get_fundamental_type]
