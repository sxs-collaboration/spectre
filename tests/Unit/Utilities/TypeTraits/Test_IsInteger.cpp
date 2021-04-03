// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Utilities/TypeTraits/IsInteger.hpp"

// [is_integer_example]
static_assert(tt::is_integer<short>::value,
              "Failed testing type trait is_integer");
static_assert(tt::is_integer_v<unsigned short>,
              "Failed testing type trait is_integer");
static_assert(tt::is_integer_v<int>, "Failed testing type trait is_integer");
static_assert(tt::is_integer_v<unsigned int>,
              "Failed testing type trait is_integer");
static_assert(tt::is_integer_v<long>, "Failed testing type trait is_integer");
static_assert(tt::is_integer_v<unsigned long>,
              "Failed testing type trait is_integer");
static_assert(tt::is_integer_v<long long>,
              "Failed testing type trait is_integer");
static_assert(tt::is_integer_v<unsigned long long>,
              "Failed testing type trait is_integer");
static_assert(not tt::is_integer_v<bool>,
              "Failed testing type trait is_integer");
static_assert(not tt::is_integer_v<char>,
              "Failed testing type trait is_integer");
// [is_integer_example]
