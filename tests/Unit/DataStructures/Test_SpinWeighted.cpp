// Distributed under the MIT License.
// See LICENSE.txt for details

#include "tests/Unit/TestingFramework.hpp"

#include <complex>
#include <cstddef>
#include <random>

#include "DataStructures/ComplexDataVector.hpp"  // IWYU pragma: keep
#include "DataStructures/DataVector.hpp"         // IWYU pragma: keep
#include "DataStructures/SpinWeighted.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/TMPL.hpp"
#include "Utilities/TypeTraits.hpp"
#include "tests/Unit/TestHelpers.hpp"
#include "tests/Utilities/MakeWithRandomValues.hpp"

// IWYU pragma: no_forward_declare ComplexDataVector
// IWYU pragma: no_forward_declare DataVector
// IWYU pragma: no_forward_declare SpinWeighted

// tests for is_any_spin_weighted
static_assert(is_any_spin_weighted_v<SpinWeighted<int, 3>>,
              "failed testing is_any_spin_weighted");
static_assert(is_any_spin_weighted_v<SpinWeighted<DataVector, 0>>,
              "failed testing is_any_spin_weighted");
static_assert(not is_any_spin_weighted_v<ComplexDataVector>,
              "failed testing is_any_spin_weighted");

// tests for is_spin_weighted_of
static_assert(is_spin_weighted_of_v<DataVector, SpinWeighted<DataVector, 1>>,
              "failed testing is_spin_weighted_of");
static_assert(is_spin_weighted_of_v<ComplexDataVector,
                                    SpinWeighted<ComplexDataVector, -1>>,
              "failed testing is_spin_weighted_of");
static_assert(
    not is_spin_weighted_of_v<ComplexDataVector, SpinWeighted<DataVector, -2>>,
    "failed testing is_spin_weighted_of");
static_assert(not is_spin_weighted_of_v<ComplexDataVector, ComplexDataVector>,
              "failed testing is_spin_weighted_of");

// tests for is_spin_weighted_of_same_type
static_assert(is_spin_weighted_of_same_type_v<SpinWeighted<DataVector, -2>,
                                              SpinWeighted<DataVector, 1>>,
              "failed testing is_spin_weighted_of_same_type");
static_assert(
    is_spin_weighted_of_same_type_v<SpinWeighted<ComplexDataVector, 0>,
                                    SpinWeighted<ComplexDataVector, -1>>,
    "failed testing is_spin_weighted_of_same_type");
static_assert(not is_spin_weighted_of_same_type_v<ComplexDataVector,
                                                  SpinWeighted<DataVector, -2>>,
              "failed testing is_spin_weighted_of_same_type");
static_assert(
    not is_spin_weighted_of_same_type_v<SpinWeighted<ComplexDataVector, 1>,
                                        SpinWeighted<DataVector, 1>>,
    "failed testing is_spin_weighted_of_same_type");

namespace {
template <typename SpinWeightedType, typename CompatibleType>
void test_spinweights() {
  MAKE_GENERATOR(gen);
  UniformCustomDistribution<tt::get_fundamental_type_t<SpinWeightedType>>
      spin_weighted_dist{
          static_cast<tt::get_fundamental_type_t<SpinWeightedType>>(
              1.0),  // avoid divide by 0
          static_cast<tt::get_fundamental_type_t<SpinWeightedType>>(100.0)};

  UniformCustomDistribution<tt::get_fundamental_type_t<CompatibleType>>
      compatible_dist{
          static_cast<tt::get_fundamental_type_t<CompatibleType>>(
              1.0),  // avoid divide by 0
          static_cast<tt::get_fundamental_type_t<CompatibleType>>(100.0)};

  UniformCustomDistribution<size_t> size_dist{5, 10};
  const size_t size = size_dist(gen);

  const auto spin_weight_0 =
      make_with_random_values<SpinWeighted<SpinWeightedType, 0>>(
          make_not_null(&gen), make_not_null(&spin_weighted_dist), size);
  const auto spin_weight_1 =
      make_with_random_values<SpinWeighted<SpinWeightedType, 1>>(
          make_not_null(&gen), make_not_null(&spin_weighted_dist), size);
  const auto spin_weight_m2 =
      make_with_random_values<SpinWeighted<SpinWeightedType, -2>>(
          make_not_null(&gen), make_not_null(&spin_weighted_dist), size);
  const auto no_spin_weight = make_with_random_values<SpinWeightedType>(
      make_not_null(&gen), make_not_null(&spin_weighted_dist), size);

  const auto compatible_spin_weight_0 =
      make_with_random_values<SpinWeighted<CompatibleType, 0>>(
          make_not_null(&gen), make_not_null(&compatible_dist), size);
  const auto compatible_spin_weight_1 =
      make_with_random_values<SpinWeighted<CompatibleType, 1>>(
          make_not_null(&gen), make_not_null(&compatible_dist), size);
  const auto compatible_spin_weight_m2 =
      make_with_random_values<SpinWeighted<CompatibleType, -2>>(
          make_not_null(&gen), make_not_null(&compatible_dist), size);
  const auto compatible_no_spin_weight =
      make_with_random_values<CompatibleType>(
          make_not_null(&gen), make_not_null(&compatible_dist), size);

  SpinWeighted<SpinWeightedType, 1> rvalue_assigned_spin_weight_1{
      spin_weight_1 + compatible_spin_weight_1};
  CHECK(rvalue_assigned_spin_weight_1.data() ==
        spin_weight_1.data() + compatible_spin_weight_1.data());
  rvalue_assigned_spin_weight_1 = spin_weight_1 - compatible_spin_weight_1;
  CHECK(rvalue_assigned_spin_weight_1.data() ==
        spin_weight_1.data() - compatible_spin_weight_1.data());

  SpinWeighted<SpinWeightedType, -2> lvalue_assigned_spin_weight_m2{
      spin_weight_m2};
  CHECK(lvalue_assigned_spin_weight_m2.data() == spin_weight_m2.data());
  lvalue_assigned_spin_weight_m2 = compatible_spin_weight_m2;
  CHECK(lvalue_assigned_spin_weight_m2.data() ==
        compatible_spin_weight_m2.data());

  // check compile-time spin values
  static_assert(decltype(spin_weight_0)::spin == 0,
                "assert failed for the spin of a spin-weight 0");
  static_assert(decltype(spin_weight_1)::spin == 1,
                "assert failed for the spin of a spin-weight 1");
  static_assert(decltype(compatible_spin_weight_0 / spin_weight_m2)::spin == 2,
                "assert failed for the spin of a spin-weight ratio.");
  static_assert(decltype(compatible_spin_weight_1 * spin_weight_1)::spin == 2,
                "assert failed for the spin of a spin-weight product.");

  // check that valid spin combinations work
  CHECK(spin_weight_0 + spin_weight_0 ==
        SpinWeighted<SpinWeightedType, 0>{spin_weight_0.data() +
                                          spin_weight_0.data()});
  CHECK(spin_weight_0 - no_spin_weight ==
        SpinWeighted<decltype(std::declval<SpinWeightedType>() -
                              std::declval<SpinWeightedType>()),
                     0>{spin_weight_0.data() - no_spin_weight});
  CHECK(spin_weight_1 * spin_weight_m2 ==
        SpinWeighted<SpinWeightedType, -1>{spin_weight_1.data() *
                                           spin_weight_m2.data()});
  CHECK(
      compatible_spin_weight_1 / spin_weight_m2 ==
      SpinWeighted<decltype(std::declval<CompatibleType>() /
                            std::declval<SpinWeightedType>()),
                   3>{compatible_spin_weight_1.data() / spin_weight_m2.data()});

  // check that plain data types act as spin 0
  CHECK(
      spin_weight_0 + no_spin_weight ==
      SpinWeighted<SpinWeightedType, 0>{spin_weight_0.data() + no_spin_weight});
  CHECK(compatible_no_spin_weight - spin_weight_0 ==
        SpinWeighted<decltype(std::declval<CompatibleType>() -
                              std::declval<SpinWeightedType>()),
                     0>{compatible_no_spin_weight - spin_weight_0.data()});
  CHECK(
      spin_weight_1 * no_spin_weight ==
      SpinWeighted<SpinWeightedType, 1>{spin_weight_1.data() * no_spin_weight});
  CHECK(no_spin_weight / spin_weight_m2 ==
        SpinWeighted<decltype(std::declval<SpinWeightedType>() /
                              std::declval<SpinWeightedType>()),
                     2>{no_spin_weight / spin_weight_m2.data()});
}

using SpinWeightedTypePairs =
    tmpl::list<tmpl::list<std::complex<double>, double>,
               tmpl::list<ComplexDataVector, double>,
               tmpl::list<ComplexDataVector, std::complex<double>>>;

SPECTRE_TEST_CASE("Unit.DataStructures.SpinWeighted",
                  "[DataStructures][Unit]") {
  tmpl::for_each<SpinWeightedTypePairs>([](auto x) noexcept {
    using type_pair = typename decltype(x)::type;
    test_spinweights<tmpl::front<type_pair>, tmpl::back<type_pair>>();
  });
}

/// \cond HIDDEN_SYMBOLS
// A macro which will static_assert fail when LHSTYPE OP RHSTYPE succeeds during
// SFINAE. Used to make sure we can't violate spin addition rules.
// clang-tidy: wants parens around macro argument, but that breaks macro
#define CHECK_TYPE_OPERATION_FAIL(TAG, OP, LHSTYPE, RHSTYPE)             \
  template <typename T1, typename T2, typename = cpp17::void_t<>>        \
  struct TAG : std::true_type {};                                        \
  template <typename T1, typename T2>                                    \
  struct TAG<                                                            \
      T1, T2,                                                            \
      cpp17::void_t<decltype(std::declval<T1>() OP std::declval<T2>())>> \
      : std::false_type {};                                              \
  static_assert(TAG<LHSTYPE, RHSTYPE>::value, /*NOLINT*/                 \
                "static_assert failed, " #LHSTYPE #OP #RHSTYPE " had a type")

using SpinZero = SpinWeighted<double, 0>;
using SpinOne = SpinWeighted<double, 1>;
using SpinTwo = SpinWeighted<double, 2>;

CHECK_TYPE_OPERATION_FAIL(spin_check_1, +, SpinZero, SpinOne);
CHECK_TYPE_OPERATION_FAIL(spin_check_2, +, SpinOne, SpinTwo);
CHECK_TYPE_OPERATION_FAIL(spin_check_3, +,
                          decltype(std::declval<SpinZero>() *
                                   std::declval<SpinTwo>()),
                          SpinOne);
/// \endcond
}  // namespace
