// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <array>
#include <cstddef>

#include "DataStructures/Tags/TempTensor.hpp"
#include "DataStructures/Tensor/TypeAliases.hpp"
#include "Evolution/Systems/GeneralizedHarmonic/System.hpp"
#include "NumericalAlgorithms/Spectral/ParityFromSymmetry.hpp"
#include "Utilities/TMPL.hpp"

namespace Spectral {
namespace {

void test_basic_functionality() {
  const Scalar<double> scalar{1};
  const auto [scalar_list, scalar_even, scalar_odd] =
      compute_parity_list<Scalar<double>>();
  const std::array<size_t, scalar.size() + 1> scalar_expected_list{1, 0};
  CHECK(scalar_list == scalar_expected_list);
  CHECK(scalar_even == 1);
  CHECK(scalar_odd == 0);

  const tnsr::a<double, 3> tensor_a{1};
  const auto [tensor_a_list, a_even, a_odd] =
      compute_parity_list<tnsr::a<double, 3>>();
  const std::array<size_t, tensor_a.size() + 1> tensor_a_expected_list{1, 1, 2,
                                                                       0, 0};
  CHECK(tensor_a_list == tensor_a_expected_list);
  CHECK(a_even == 3);
  CHECK(a_odd == 1);

  const tnsr::aI<double, 2> tensor_ai{1};
  const auto [tensor_ai_list, ai_even, ai_odd] =
      compute_parity_list<tnsr::aI<double, 2>>();
  const std::array<size_t, tensor_ai.size() + 1> tensor_ai_expected_list{
      0, 1, 1, 1, 1, 1, 1};
  CHECK(tensor_ai_list == tensor_ai_expected_list);
  CHECK(ai_even == 3);
  CHECK(ai_odd == 3);

  // Test edge cases
  using empty_tags = tmpl::list<>;
  const auto [empty_list, empty_even, empty_odd] =
      compute_parity_list<empty_tags>();
  const std::array<size_t, 1> empty_expected{0};
  CHECK(empty_list == empty_expected);
  CHECK(empty_even == 0);
  CHECK(empty_odd == 0);

  using single_tag = tmpl::list<::Tags::TempScalar<0>>;
  const auto [single_list, single_even, single_odd] =
      compute_parity_list<single_tag>();
  const std::array<size_t, 2> single_expected{1, 0};
  CHECK(single_list == single_expected);
  CHECK(single_even == 1);
  CHECK(single_odd == 0);
}

void test_expected_properties() {
  const tnsr::ia<double, 2> tensor_ia{1};
  const auto [ia_list, ia_even, ia_odd] =
      compute_parity_list<tnsr::ia<double, 2>>();

  // Total components = even_count + odd_count
  CHECK(ia_even + ia_odd == tensor_ia.size());

  // Sum of run lengths equals total components
  size_t total_from_runs = 0;
  for (const auto& count : ia_list) {
    total_from_runs += count;
  }
  CHECK(total_from_runs == tensor_ia.size());

  // Non-zero entries should be contiguous (no gaps)
  bool first_index = true;
  bool found_zero = false;
  for (const auto& count : ia_list) {
    if (first_index) {
      first_index = false;
    } else {
      if (count == 0) {
        found_zero = true;
      } else if (found_zero) {
        FAIL("Found non-zero entry after zero in run-length array");
      }
    }
  }
}

template <typename VarsTags>
void check_vars_scalar_constexpr(Variables<VarsTags> /*meta*/) {
  constexpr auto res = compute_parity_list<VarsTags>();
  static_assert(std::get<0>(res) == std::array<size_t, 2>{1, 0});
  static_assert(std::get<1>(res) == 1 and std::get<2>(res) == 0);
}

template <typename DataType, typename SymmList, typename IndexList>
void check_scalar_constexpr(Tensor<DataType, SymmList, IndexList> /*meta*/) {
  constexpr auto res =
      compute_parity_list<Tensor<DataType, SymmList, IndexList>>();
  static_assert(std::get<0>(res) == std::array<size_t, 2>{1, 0});
  static_assert(std::get<1>(res) == 1 and std::get<2>(res) == 0);
}

void test_constexpr() {
  // This test is trivial because it works, but it exists because when the
  // compute_party_list is used in practice, the object's value is not known
  // at compiletime, so it cannot be passed if we want the
  // evaluation to be done at compiletime--it has to be purely template
  // evaluation. As all the tests in this file are done with objects that can be
  // fully determined at compiletime, this passing as a parameter to a function
  // makes it actually test it's constexpr-ness
  using test_tags = tmpl::list<::Tags::TempScalar<0>>;
  const Variables<test_tags> scalar_vars(1);
  check_vars_scalar_constexpr(scalar_vars);

  const Scalar<DataVector> scalar(1_st);
  check_scalar_constexpr(scalar);
}

void test_gh_system() {
  using gh_tags = gh::System<3>::gradients_tags;
  const auto [gh_list, gh_even, gh_odd] = compute_parity_list<gh_tags>();

  const std::array<size_t,
                   Variables<gh_tags>::number_of_independent_components + 1>
      expected_list{1, 1, 3, 2, 4, 1, 3, 2, 3, 1, 3, 3, 2, 1, 2, 1, 3,
                    2, 1, 3, 2, 1, 2, 1, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
  CHECK(gh_list == expected_list);
  CHECK(gh_even == 31);
  CHECK(gh_odd == 19);
  CHECK(gh_even + gh_odd ==
        Variables<gh_tags>::number_of_independent_components);
}
}  // namespace

SPECTRE_TEST_CASE("Unit.Numerical.Spectral.ParityFromSymmetry",
                  "[NumericalAlgorithms][Spectral][Unit]") {
  test_basic_functionality();
  test_expected_properties();
  test_constexpr();
  test_gh_system();
}
}  // namespace Spectral
