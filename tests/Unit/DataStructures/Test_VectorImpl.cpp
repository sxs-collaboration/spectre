// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <array>
#include <complex>
#include <vector>

#include "DataStructures/ComplexDataVector.hpp"  // IWYU pragma: keep
#include "DataStructures/DataVector.hpp"  // IWYU pragma: keep
#include "DataStructures/VectorImpl.hpp"  // IWYU pragma: keep
#include "Utilities/TypeTraits.hpp"

/// [get_vector_element_type_example]
static_assert(std::is_same_v<get_vector_element_type_t<DataVector>, double>,
              "Failed testing type trait get_vector_element_type");

static_assert(std::is_same_v<
                  get_vector_element_type_t<std::array<DataVector, 2>>, double>,
              "Failed testing type trait get_vector_element_type");

static_assert(std::is_same_v<get_vector_element_type_t<std::complex<double>*>,
                             std::complex<double>>,
              "Failed testing type trait get_vector_element_type");

static_assert(std::is_same_v<get_vector_element_type_t<DataVector&>, double>,
              "Failed testing type trait get_vector_element_type");
/// [get_vector_element_type_example]

static_assert(is_derived_of_vector_impl_v<DataVector>,
              "Failed testing type trait is_derived_of_vector_impl");
static_assert(is_derived_of_vector_impl_v<ComplexDataVector>,
              "Failed testing type trait is_derived_of_vector_impl");
static_assert(not is_derived_of_vector_impl_v<std::complex<double>>,
              "Failed testing type trait is_derived_of_vector_impl");
static_assert(not is_derived_of_vector_impl_v<std::vector<int>>,
              "Failed testing type trait is_derived_of_vector_impl");

// [[OutputRegex, Comparing an empty vector to a value]]
[[noreturn]] SPECTRE_TEST_CASE(
    "Unit.DataStructures.VectorImpl.empty_equals_double",
    "[DataStructures][Unit]") {
  ASSERTION_TEST();
#ifdef SPECTRE_DEBUG
  (void)(DataVector{} == 1.0);
  ERROR("Failed to trigger ASSERT in an assertion test");
#endif
}

// [[OutputRegex, Comparing an empty vector to a value]]
[[noreturn]] SPECTRE_TEST_CASE(
    "Unit.DataStructures.VectorImpl.empty_equals_complex",
    "[DataStructures][Unit]") {
  ASSERTION_TEST();
#ifdef SPECTRE_DEBUG
  (void)(ComplexDataVector{} != std::complex<double>{});
  ERROR("Failed to trigger ASSERT in an assertion test");
#endif
}
