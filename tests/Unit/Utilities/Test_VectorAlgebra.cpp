// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <cstddef>
#include <random>

#include "DataStructures/ComplexDataVector.hpp"
#include "DataStructures/DataVector.hpp"
#include "DataStructures/ModalVector.hpp"
#include "Framework/TestHelpers.hpp"
#include "Helpers/DataStructures/MakeWithRandomValues.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/TypeTraits/GetFundamentalType.hpp"
#include "Utilities/VectorAlgebra.hpp"

template <typename RhsVectorType, typename LhsVectorType>
void test_outer_product() {
  MAKE_GENERATOR(gen);
  UniformCustomDistribution<size_t> size_distribution{5, 10};
  UniformCustomDistribution<
      tt::get_fundamental_type_t<get_vector_element_type_t<LhsVectorType>>>
      lhs_distribution{-2.0, 2.0};
  UniformCustomDistribution<
      tt::get_fundamental_type_t<get_vector_element_type_t<RhsVectorType>>>
      rhs_distribution{-2.0, 2.0};
  const LhsVectorType lhs = make_with_random_values<LhsVectorType>(
      make_not_null(&gen), make_not_null(&lhs_distribution),
      size_distribution(gen));
  const RhsVectorType rhs = make_with_random_values<RhsVectorType>(
      make_not_null(&gen), make_not_null(&rhs_distribution),
      size_distribution(gen));
  const auto outer_test = outer_product(lhs, rhs);
  size_t product_index = 0;
  for (const auto& b : rhs) {
    for (const auto& a : lhs) {
      // Iterable approx has overloads for both real and complex types
      CHECK_ITERABLE_APPROX(outer_test[product_index], a * b);
      ++product_index;
    }
  }
}

template <typename VectorType>
void test_n_copies() {
  MAKE_GENERATOR(gen);
  UniformCustomDistribution<size_t> size_distribution{5, 10};
  UniformCustomDistribution<
      tt::get_fundamental_type_t<get_vector_element_type_t<VectorType>>>
      element_distribution{-2.0, 2.0};
  const VectorType to_repeat = make_with_random_values<VectorType>(
      make_not_null(&gen), make_not_null(&element_distribution),
      size_distribution(gen));
  size_t repeats = size_distribution(gen);
  const auto repeated = create_vector_of_n_copies(to_repeat, repeats);
  for (size_t i = 0; i < repeats; ++i) {
    for (size_t j = 0; j < to_repeat.size(); ++j) {
      CHECK(to_repeat[j] == repeated[j + i * to_repeat.size()]);
    }
  }
}

SPECTRE_TEST_CASE("Unit.DataStructures.VectorAlgebra",
                  "[Unit][DataStructures]") {
  test_outer_product<ComplexDataVector, ComplexDataVector>();
  test_outer_product<DataVector, ComplexDataVector>();
  test_outer_product<ComplexDataVector, DataVector>();
  test_outer_product<DataVector, DataVector>();

  test_n_copies<DataVector>();
  test_n_copies<ComplexDataVector>();
  test_n_copies<ModalVector>();
}
