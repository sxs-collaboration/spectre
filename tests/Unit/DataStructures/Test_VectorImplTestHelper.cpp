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
  /// [tuple_combos_lambda]
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
  /// [tuple_combos_lambda]
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
          typename FundValType1 =
              tt::get_fundamental_type_t<typename VectorType1::ElementType>,
          typename FundValType2 =
              tt::get_fundamental_type_t<typename VectorType2::ElementType>>
void test_check_vectors(const FundValType1& low1 = FundValType1{-100},
                        const FundValType1& high1 = FundValType1{
                            100}) noexcept {
  MAKE_GENERATOR(gen);
  UniformCustomDistribution<FundValType1> dist1{low1, high1};
  UniformCustomDistribution<size_t> sdist{2, 20};
  const auto v1 = make_with_random_values<typename VectorType1::ElementType>(
      make_not_null(&gen), make_not_null(&dist1));
  const typename VectorType2::ElementType v2{v1};
  const auto s1 = sdist(gen);
  const VectorType1 t1{s1, v1};
  const VectorType2 t2{s1, v2};
  std::array<VectorType1, 2> a1_1;
  std::array<VectorType2, 2> a1_2;
  std::array<std::array<VectorType1, 2>, 2> a2_1;
  std::array<std::array<VectorType2, 2>, 2> a2_2;
  a1_1.fill(t1);
  a1_2.fill(t2);
  a2_1.fill(a1_1);
  a2_2.fill(a1_2);
  // check vectors with vectors or values
  TestHelpers::VectorImpl::detail::check_vectors(t1, t2);
  TestHelpers::VectorImpl::detail::check_vectors(t1, v2);
  TestHelpers::VectorImpl::detail::check_vectors(v1, t2);
  TestHelpers::VectorImpl::detail::check_vectors(v1, v2);
  // check arrays with arrays
  TestHelpers::VectorImpl::detail::check_vectors(a1_1, a1_2);
  TestHelpers::VectorImpl::detail::check_vectors(a2_1, a2_2);
  // check arrays with vectors or values
  TestHelpers::VectorImpl::detail::check_vectors(a1_1, t2);
  TestHelpers::VectorImpl::detail::check_vectors(a1_1, v2);
  TestHelpers::VectorImpl::detail::check_vectors(t1, a1_2);
  TestHelpers::VectorImpl::detail::check_vectors(v2, a1_2);
}

// only current type we have to test with is DataVector
SPECTRE_TEST_CASE("Unit.DataStructures.VectorImplHelpers.CheckVectors",
                  "[Utilities][Unit]") {
  test_check_vectors<DataVector, DataVector>();
}
