// Distributed under the MIT License.
// See LICENSE.txt for details.

#include <array>
#include <complex>

#include "DataStructures/DataVector.hpp"  // IWYU pragma: keep
#include "DataStructures/VectorImpl.hpp"  // IWYU pragma: keep
#include "Utilities/TypeTraits.hpp"

/// [get_vector_element_type_example]
static_assert(cpp17::is_same_v<get_vector_element_type_t<DataVector>, double>,
              "Failed testing type trait get_vector_element_type");

static_assert(cpp17::is_same_v<
                  get_vector_element_type_t<std::array<DataVector, 2>>, double>,
              "Failed testing type trait get_vector_element_type");

static_assert(cpp17::is_same_v<get_vector_element_type_t<std::complex<double>*>,
                               std::complex<double>>,
              "Failed testing type trait get_vector_element_type");

static_assert(cpp17::is_same_v<get_vector_element_type_t<DataVector&>, double>,
              "Failed testing type trait get_vector_element_type");
/// [get_vector_element_type_example]
