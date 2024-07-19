// Distributed under the MIT License.
// See LICENSE.txt for details.

#include <array>
#include <complex>
#include <vector>

#include "Utilities/TypeTraits.hpp"
#include "Utilities/TypeTraits/GetFundamentalType.hpp"

// [get_fundamental_type]
static_assert(
    std::is_same_v<
        typename tt::get_fundamental_type<std::array<double, 2>>::type, double>,
    "Failed testing get_fundamental_type");
static_assert(
    std::is_same_v<
        typename tt::get_fundamental_type_t<std::vector<std::complex<double>>>,
        double>,
    "Failed testing get_fundamental_type");
static_assert(std::is_same_v<typename tt::get_fundamental_type_t<int>, int>,
              "Failed testing get_fundamental_type");
// [get_fundamental_type]

static_assert(
    std::is_same_v<
        typename tt::get_complex_or_fundamental_type_t<std::array<double, 2>>,
        double>);
static_assert(std::is_same_v<typename tt::get_complex_or_fundamental_type_t<
                                 std::vector<std::complex<double>>>,
                             std::complex<double>>);
static_assert(
    std::is_same_v<typename tt::get_complex_or_fundamental_type_t<int>, int>);
static_assert(
    std::is_same_v<
        typename tt::get_complex_or_fundamental_type_t<std::complex<double>>,
        std::complex<double>>);
