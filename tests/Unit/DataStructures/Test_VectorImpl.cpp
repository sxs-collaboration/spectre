// Distributed under the MIT License.
// See LICENSE.txt for details.

#include <array>
#include <complex>

#include "DataStructures/DataVector.hpp"  // IWYU pragma: keep
#include "DataStructures/VectorImpl.hpp"
#include "Utilities/TypeTraits.hpp"

/// [vector_base_type_example]
static_assert(cpp17::is_same_v<vector_base_type_t<DataVector>, double>,
              "Failed testing type trait vector_base_type");

static_assert(
    cpp17::is_same_v<vector_base_type_t<std::array<DataVector, 2>>, double>,
    "Failed testing type trait vector_base_type");

static_assert(cpp17::is_same_v<vector_base_type_t<std::complex<double>*>,
                               std::complex<double>>,
              "Failed testing type trait vector_base_type");

static_assert(cpp17::is_same_v<vector_base_type_t<DataVector&>, double>,
              "Failed testing type trait vector_base_type");
/// [vector_base_type_example]
