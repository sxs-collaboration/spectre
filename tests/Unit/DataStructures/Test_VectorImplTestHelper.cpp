// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "tests/Unit/TestingFramework.hpp"

#include <array>
#include <complex>
#include <cstddef>
#include <random>
#include <tuple>

#include "DataStructures/DataVector.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/TypeTraits.hpp"
#include "tests/Unit/DataStructures/VectorImplTestHelper.hpp"
#include "tests/Unit/TestHelpers.hpp"
#include "tests/Utilities/MakeWithRandomValues.hpp"

SPECTRE_TEST_CASE("Unit.Utilities.VectorImplHelpers.ApplyTupleCombinations",
                  "[Utilities][Unit]") {
  // for more generic applications, with lambdas:
  const int zero = 0;
  const double one = 1.2;  // for general lambda, 1.2 casts to 1
  const std::complex<double> two{2.5, 5.0};  // for lambda, real and casts to 2
  std::array<std::array<bool, 3>, 3> checklist {{{{false, false, false}},
                                                 {{false, false, false}},
                                                 {{false, false, false}}}};
  const auto lambda_tuple = std::make_tuple(zero, one, two);
  TestHelpers::VectorImpl::apply_tuple_combinations<2>(
      lambda_tuple, [&checklist](auto x, auto y) noexcept {
        gsl::at(gsl::at(checklist, static_cast<size_t>(std::real(x))),
                static_cast<size_t>(std::real(y))) = true;
        CHECK(std::imag(x) + std::real(y) >= 0);
      });
  CHECK(checklist == std::array<std::array<bool, 3>, 3>{
          {{{true, true, true}}, {{true, true, true}}, {{true, true, true}}}});
}

SPECTRE_TEST_CASE("Unit.DataStructures.VectorImplHelpers.TupleHelpers",
                  "[Utilities][Unit]") {
  auto test_tup = std::make_tuple(5.5, std::complex<int>{2, 3}, 'c');
  CHECK(TestHelpers::VectorImpl::remove_nth<0>(test_tup) ==
        std::make_tuple(std::complex<int>{2, 3}, 'c'));
  CHECK(TestHelpers::VectorImpl::remove_nth<1>(test_tup) ==
        std::make_tuple(5.5, 'c'));
  CHECK(TestHelpers::VectorImpl::remove_nth<2>(test_tup) ==
        std::make_tuple(5.5, std::complex<int>{2, 3}));
  const auto addr_test_tup =
      TestHelpers::VectorImpl::addressof(make_not_null(&test_tup));
  CHECK(addr_test_tup == std::make_tuple(&std::get<0>(test_tup),
                                         &std::get<1>(test_tup),
                                         &std::get<2>(test_tup)));
}

// test the check_vectors against one another, distributions default to the cast
// to value type of [-100,100]. The second vector type should be constructible
// from the first vector type. For instance, calling in the order of (a double
// type, a complex double type) will function, but not the reverse.
template <typename VectorType1, typename VectorType2,
          typename FundamentalValueType =
              tt::get_fundamental_type_t<typename VectorType1::ElementType>>
void test_check_vectors(
    const FundamentalValueType& low = FundamentalValueType{-100},
    const FundamentalValueType& high = FundamentalValueType{100}) noexcept {
  MAKE_GENERATOR(gen);
  UniformCustomDistribution<FundamentalValueType> value_dist{low, high};
  UniformCustomDistribution<size_t> sdist{2, 20};
  const auto random_element1 =
      make_with_random_values<typename VectorType1::ElementType>(
          make_not_null(&gen), make_not_null(&value_dist));
  const typename VectorType2::ElementType element2{random_element1};
  const auto size = sdist(gen);
  const VectorType1 test_vector1{size, random_element1};
  const VectorType2 test_vector2{size, element2};
  std::array<VectorType1, 2> array_of_test_vector1;
  std::array<VectorType2, 2> array_of_test_vector2;
  std::array<std::array<VectorType1, 2>, 2> nested_array_of_test_vector1;
  std::array<std::array<VectorType2, 2>, 2> nested_array_of_test_vector2;
  array_of_test_vector1.fill(test_vector1);
  array_of_test_vector2.fill(test_vector2);
  nested_array_of_test_vector1.fill(array_of_test_vector1);
  nested_array_of_test_vector2.fill(array_of_test_vector2);
  // check vectors with vectors or values
  TestHelpers::VectorImpl::detail::check_vectors(test_vector1, test_vector2);
  TestHelpers::VectorImpl::detail::check_vectors(test_vector1, element2);
  TestHelpers::VectorImpl::detail::check_vectors(random_element1, test_vector2);
  TestHelpers::VectorImpl::detail::check_vectors(random_element1, element2);
  // check arrays with arrays
  TestHelpers::VectorImpl::detail::check_vectors(array_of_test_vector1,
                                                 array_of_test_vector2);
  TestHelpers::VectorImpl::detail::check_vectors(nested_array_of_test_vector1,
                                                 nested_array_of_test_vector2);
  // check arrays with vectors or values
  TestHelpers::VectorImpl::detail::check_vectors(array_of_test_vector1,
                                                 test_vector2);
  TestHelpers::VectorImpl::detail::check_vectors(array_of_test_vector1,
                                                 element2);
  TestHelpers::VectorImpl::detail::check_vectors(test_vector1,
                                                 array_of_test_vector2);
  TestHelpers::VectorImpl::detail::check_vectors(element2,
                                                 array_of_test_vector2);
}

// only current type we have to test with is DataVector
SPECTRE_TEST_CASE("Unit.DataStructures.VectorImplHelpers.CheckVectors",
                  "[Utilities][Unit]") {
  test_check_vectors<DataVector, DataVector>();
}
