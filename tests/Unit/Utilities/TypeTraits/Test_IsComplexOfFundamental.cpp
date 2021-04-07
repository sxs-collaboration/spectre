// Distributed under the MIT License.
// See LICENSE.txt for details.

#include <complex>

#include "Utilities/TypeTraits.hpp"
#include "Utilities/TypeTraits/IsComplexOfFundamental.hpp"

class DataVector;

// [is_complex_of_fundamental]
static_assert(tt::is_complex_of_fundamental_v<std::complex<double>>,
              "Failed testing is_complex_of_fundamental");
static_assert(tt::is_complex_of_fundamental_v<std::complex<int>>,
              "Failed testing is_complex_of_fundamental");
static_assert(not tt::is_complex_of_fundamental_v<double>,
              "Failed testing is_complex_of_fundamental");
static_assert(not tt::is_complex_of_fundamental_v<std::complex<DataVector>>,
              "Failed testing is_complex_of_fundamental");
static_assert(not tt::is_complex_of_fundamental_v<DataVector>,
              "Failed testing is_complex_of_fundamental");
// [is_complex_of_fundamental]

static_assert(tt::is_complex_or_fundamental_v<std::complex<double>>,
              "Failed testing is_complex_of_fundamental");
static_assert(tt::is_complex_or_fundamental_v<std::complex<int>>,
              "Failed testing is_complex_of_fundamental");
static_assert(tt::is_complex_or_fundamental_v<double>,
              "Failed testing is_complex_of_fundamental");
static_assert(not tt::is_complex_or_fundamental_v<std::complex<DataVector>>,
              "Failed testing is_complex_of_fundamental");
static_assert(not tt::is_complex_or_fundamental_v<DataVector>,
              "Failed testing is_complex_of_fundamental");
