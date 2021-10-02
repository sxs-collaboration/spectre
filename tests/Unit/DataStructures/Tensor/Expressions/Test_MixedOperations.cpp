// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <climits>
#include <cstddef>
#include <iterator>
#include <numeric>
#include <type_traits>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tags/TempTensor.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "DataStructures/Variables.hpp"
#include "Framework/TestHelpers.hpp"
#include "Helpers/DataStructures/MakeWithRandomValues.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/MakeWithValue.hpp"
#include "Utilities/TMPL.hpp"

namespace {
// Computes \f$L_{a} = R_{ab} S^{b} + G_{a} - H_{ba}{}^{b} T\f$
template <typename R_t, typename S_t, typename G_t, typename H_t, typename T_t,
          typename DataType>
G_t compute_expected_result1(const R_t& R, const S_t& S, const G_t& G,
                             const H_t& H, const T_t& T,
                             const DataType& used_for_size) {
  using result_tensor_type = G_t;
  result_tensor_type expected_result{};
  const size_t dim = tmpl::front<typename R_t::index_list>::dim;
  for (size_t a = 0; a < dim; a++) {
    DataType expected_Rab_SB_product =
        make_with_value<DataType>(used_for_size, 0.0);
    DataType expected_HbaB_contracted_value =
        make_with_value<DataType>(used_for_size, 0.0);
    for (size_t b = 0; b < dim; b++) {
      expected_Rab_SB_product += R.get(a, b) * S.get(b);
      expected_HbaB_contracted_value += H.get(b, a, b);
    }
    expected_result.get(a) = expected_Rab_SB_product + G.get(a) -
                             (expected_HbaB_contracted_value * T.get());
  }

  return expected_result;
}

// Computes the lapse from the inverse spatial metric and spacetime metric
//
// The lapse is calulated using the following equation:
// \f$\alpha = \sqrt{\gamma^{ij} g_{jt} g_{it} - g_{tt}}\f$
template <typename DataType>
Scalar<DataType> compute_expected_result2(
    const tnsr::II<DataType, 3, Frame::Inertial>& spatial_metric,
    const tnsr::aa<DataType, 3, Frame::Inertial>& spacetime_metric,
    const DataType& used_for_size) {
  DataType expected_gamma_g_g_product =
      make_with_value<DataType>(used_for_size, 0.0);
  for (size_t i = 0; i < 3; i++) {
    for (size_t j = 0; j < 3; j++) {
      expected_gamma_g_g_product += spatial_metric.get(i, j) *
                                    spacetime_metric.get(j + 1, 0) *
                                    spacetime_metric.get(i + 1, 0);
    }
  }

  Scalar<DataType> expected_result{
      sqrt(expected_gamma_g_g_product - spacetime_metric.get(0, 0))};

  return expected_result;
}

// Computes the spacetime derivative of the spacetime metric
//
// \f$\partial_c g_{ab}\f$ is computed using the following two equations:
// (1) \f$\partial_t g_{ab} = -\alpha \Pi_{ab} + \beta^i \Phi_{iab}\f$
// (2) \f$\partial_i g_{ab} = \Phi_{iab}\f$
template <typename DataType>
tnsr::abb<DataType, 3, Frame::Inertial> compute_expected_result3(
    const Scalar<DataType>& alpha,
    const tnsr::I<DataType, 3, Frame::Inertial>& beta,
    const tnsr::aa<DataType, 3, Frame::Inertial>& pi,
    const tnsr::iaa<DataType, 3, Frame::Inertial>& phi,
    const DataType& used_for_size) {
  tnsr::abb<DataType, 3, Frame::Inertial> expected_result{};

  for (size_t a = 0; a < 4; a++) {
    for (size_t b = 0; b < 4; b++) {
      DataType expected_beta_phi_product =
          make_with_value<DataType>(used_for_size, 0.0);
      for (size_t i = 0; i < 3; i++) {
        expected_beta_phi_product += beta.get(i) * phi.get(i, a, b);
        expected_result.get(i + 1, a, b) = phi.get(i, a, b);
      }
      expected_result.get(0, a, b) =
          -alpha.get() * pi.get(a, b) + expected_beta_phi_product;
    }
  }

  return expected_result;
}

// Includes an expression with addition, subtraction, an inner product, an outer
// product, a contraction, and a scalar
template <typename DataType, typename Generator>
void test_case1(const DataType& used_for_size,
                const gsl::not_null<Generator*> generator) {
  std::uniform_real_distribution<> distribution(0.1, 1.0);

  const auto R = make_with_random_values<tnsr::ab<DataType, 3, Frame::Grid>>(
      generator, make_not_null(&distribution), used_for_size);

  const auto S = make_with_random_values<tnsr::A<DataType, 3, Frame::Grid>>(
      generator, make_not_null(&distribution), used_for_size);

  const auto G = make_with_random_values<tnsr::a<DataType, 3, Frame::Grid>>(
      generator, make_not_null(&distribution), used_for_size);

  const auto H = make_with_random_values<tnsr::abC<DataType, 3, Frame::Grid>>(
      generator, make_not_null(&distribution), used_for_size);

  const auto T = make_with_random_values<Scalar<DataType>>(
      generator, make_not_null(&distribution), used_for_size);

  using result_tensor_type = tnsr::a<DataType, 3, Frame::Grid>;
  result_tensor_type expected_result_tensor =
      compute_expected_result1(R, S, G, H, T, used_for_size);
  // \f$L_{a} = R_{ab} S^{b} + G_{a} - H_{ba}{}^{b} T\f$
  result_tensor_type actual_result_tensor_returned =
      TensorExpressions::evaluate<ti_a>(R(ti_a, ti_b) * S(ti_B) + G(ti_a) -
                                        H(ti_b, ti_a, ti_B) * T());
  result_tensor_type actual_result_tensor_filled{};
  TensorExpressions::evaluate<ti_a>(
      make_not_null(&actual_result_tensor_filled),
      R(ti_a, ti_b) * S(ti_B) + G(ti_a) - H(ti_b, ti_a, ti_B) * T());

  for (size_t a = 0; a < 4; a++) {
    CHECK_ITERABLE_APPROX(actual_result_tensor_returned.get(a),
                          expected_result_tensor.get(a));
    CHECK_ITERABLE_APPROX(actual_result_tensor_filled.get(a),
                          expected_result_tensor.get(a));
  }

  // Test with TempTensor for LHS tensor
  if constexpr (not std::is_same_v<DataType, double>) {
    Variables<tmpl::list<::Tags::TempTensor<1, result_tensor_type>>>
        actual_result_tensor_temp_var{used_for_size.size()};
    result_tensor_type& actual_result_tensor_temp =
        get<::Tags::TempTensor<1, result_tensor_type>>(
            actual_result_tensor_temp_var);
    ::TensorExpressions::evaluate<ti_a>(
        make_not_null(&actual_result_tensor_temp),
        R(ti_a, ti_b) * S(ti_B) + G(ti_a) - H(ti_b, ti_a, ti_B) * T());

    for (size_t a = 0; a < 4; a++) {
      CHECK_ITERABLE_APPROX(actual_result_tensor_temp.get(a),
                            expected_result_tensor.get(a));
    }
  }
}

// Includes an expression with subtraction, inner products, an outer product,
// a square root, a scalar, generic spatial indices used for spacetime indices,
// and concrete time indices used for spacetime indices
//
// Note: This is an expanded calculation of the lapse from the shift and
// spacetime metric, where the shift is calculated from the inverse spatial
// metric and spacetime metric.
template <typename DataType, typename Generator>
void test_case2(const DataType& used_for_size,
                const gsl::not_null<Generator*> generator) {
  // Use a higher distribution for sptial metric than spacetime metric to ensure
  // we do not take the square root of a negative number
  std::uniform_real_distribution<> spatial_metric_distribution(3.0, 4.0);
  std::uniform_real_distribution<> spacetime_metric_distribution(1.0, 2.0);

  const auto spatial_metric =
      make_with_random_values<tnsr::II<DataType, 3, Frame::Inertial>>(
          generator, make_not_null(&spatial_metric_distribution),
          used_for_size);

  const auto spacetime_metric =
      make_with_random_values<tnsr::aa<DataType, 3, Frame::Inertial>>(
          generator, make_not_null(&spacetime_metric_distribution),
          used_for_size);

  const Scalar<DataType> expected_result_tensor =
      compute_expected_result2(spatial_metric, spacetime_metric, used_for_size);
  // \f$\alpha = \sqrt{\gamma^{ij} g_{jt} g_{it} - g_{tt}}\f$
  const Scalar<DataType> actual_result_tensor_returned =
      TensorExpressions::evaluate(
          sqrt(spatial_metric(ti_I, ti_J) * spacetime_metric(ti_j, ti_t) *
                   spacetime_metric(ti_i, ti_t) -
               spacetime_metric(ti_t, ti_t)));
  Scalar<DataType> actual_result_tensor_filled{};
  TensorExpressions::evaluate(
      make_not_null(&actual_result_tensor_filled),
      sqrt(spatial_metric(ti_I, ti_J) * spacetime_metric(ti_j, ti_t) *
               spacetime_metric(ti_i, ti_t) -
           spacetime_metric(ti_t, ti_t)));

  CHECK_ITERABLE_APPROX(actual_result_tensor_returned.get(),
                        expected_result_tensor.get());

  // Test with TempTensor for LHS tensor
  if constexpr (not std::is_same_v<DataType, double>) {
    Variables<tmpl::list<::Tags::TempTensor<1, Tensor<DataType>>>>
        actual_result_tensor_temp_var{used_for_size.size()};
    Scalar<DataType>& actual_result_tensor_temp =
        get<::Tags::TempTensor<1, Tensor<DataType>>>(
            actual_result_tensor_temp_var);
    ::TensorExpressions::evaluate(
        make_not_null(&actual_result_tensor_temp),
        sqrt(spatial_metric(ti_I, ti_J) * spacetime_metric(ti_j, ti_t) *
                 spacetime_metric(ti_i, ti_t) -
             spacetime_metric(ti_t, ti_t)));

    CHECK_ITERABLE_APPROX(actual_result_tensor_temp.get(),
                          expected_result_tensor.get());
  }
}

// Includes an expression with addition, subtraction, an inner product, outer
// products, a scalar, and two calls to `evaluate` that fill the time and
// spatial components of the result tensor, respectively
//
// Note: This is the calculation of the spacetime derivative of the spacetime
// metric, \f$\partial_c g_{ab}\f$
template <typename DataType, typename Generator>
void test_case3(const DataType& used_for_size,
                const gsl::not_null<Generator*> generator) {
  std::uniform_real_distribution<> distribution(0.1, 1.0);

  const auto alpha = make_with_random_values<Scalar<DataType>>(
      generator, make_not_null(&distribution), used_for_size);

  const auto beta =
      make_with_random_values<tnsr::I<DataType, 3, Frame::Inertial>>(
          generator, make_not_null(&distribution), used_for_size);

  const auto pi =
      make_with_random_values<tnsr::aa<DataType, 3, Frame::Inertial>>(
          generator, make_not_null(&distribution), used_for_size);

  const auto phi =
      make_with_random_values<tnsr::iaa<DataType, 3, Frame::Inertial>>(
          generator, make_not_null(&distribution), used_for_size);

  using result_tensor_type = tnsr::abb<DataType, 3, Frame::Inertial>;
  result_tensor_type expected_result_tensor =
      compute_expected_result3(alpha, beta, pi, phi, used_for_size);
  result_tensor_type actual_result_tensor_filled{};
  // \f$\partial_t g_{ab} = -\alpha \Pi_{ab} + \beta^i \Phi_{iab}\f$
  TensorExpressions::evaluate<ti_t, ti_a, ti_b>(
      make_not_null(&actual_result_tensor_filled),
      -1.0 * alpha() * pi(ti_a, ti_b) + beta(ti_I) * phi(ti_i, ti_a, ti_b));
  // \f$\partial_i g_{ab} = \Phi_{iab}\f$
  TensorExpressions::evaluate<ti_i, ti_a, ti_b>(
      make_not_null(&actual_result_tensor_filled), phi(ti_i, ti_a, ti_b));

  for (size_t c = 0; c < 4; c++) {
    for (size_t a = 0; a < 4; a++) {
      for (size_t b = 0; b < 4; b++) {
        CHECK_ITERABLE_APPROX(actual_result_tensor_filled.get(c, a, b),
                              expected_result_tensor.get(c, a, b));
      }
    }
  }

  // Test with TempTensor for LHS tensor
  if constexpr (not std::is_same_v<DataType, double>) {
    Variables<tmpl::list<::Tags::TempTensor<1, result_tensor_type>>>
        actual_result_tensor_temp_var{used_for_size.size()};
    result_tensor_type& actual_result_tensor_temp =
        get<::Tags::TempTensor<1, result_tensor_type>>(
            actual_result_tensor_temp_var);
    TensorExpressions::evaluate<ti_t, ti_a, ti_b>(
        make_not_null(&actual_result_tensor_temp),
        -1.0 * alpha() * pi(ti_a, ti_b) + beta(ti_I) * phi(ti_i, ti_a, ti_b));
    TensorExpressions::evaluate<ti_i, ti_a, ti_b>(
        make_not_null(&actual_result_tensor_temp), phi(ti_i, ti_a, ti_b));

    for (size_t c = 0; c < 4; c++) {
      for (size_t a = 0; a < 4; a++) {
        for (size_t b = 0; b < 4; b++) {
          CHECK_ITERABLE_APPROX(actual_result_tensor_temp.get(c, a, b),
                                expected_result_tensor.get(c, a, b));
        }
      }
    }
  }
}

template <typename DataType, typename Generator>
void test_mixed_operations(const DataType& used_for_size,
                           const gsl::not_null<Generator*> generator) {
  test_case1(used_for_size, generator);
  test_case2(used_for_size, generator);
  test_case3(used_for_size, generator);
}
}  // namespace

SPECTRE_TEST_CASE("Unit.DataStructures.Tensor.Expression.MixedOperations",
                  "[DataStructures][Unit]") {
  MAKE_GENERATOR(generator);

  test_mixed_operations(std::numeric_limits<double>::signaling_NaN(),
                        make_not_null(&generator));
  test_mixed_operations(
      DataVector(5, std::numeric_limits<double>::signaling_NaN()),
      make_not_null(&generator));
}
