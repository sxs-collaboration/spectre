// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <climits>
#include <complex>
#include <cstddef>
#include <random>
#include <type_traits>

#include "DataStructures/ComplexDataVector.hpp"
#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tags/TempTensor.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "DataStructures/Variables.hpp"
#include "DataStructures/VectorImpl.hpp"
#include "Framework/TestHelpers.hpp"
#include "Helpers/DataStructures/MakeWithRandomValues.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/MakeWithValue.hpp"
#include "Utilities/TMPL.hpp"

namespace {
// Computes \f$L_{a} = R_{ab} S^{b} + G_{a} - H_{ba}{}^{b} T\f$
template <typename R_t, typename S_t, typename G_t, typename H_t, typename T_t,
          typename DataType>
G_t compute_expected_mixed_arithmetic_ops(const R_t& R, const S_t& S,
                                          const G_t& G, const H_t& H,
                                          const T_t& T,
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
Scalar<DataType> compute_expected_rhs_spacetime_index_subsets(
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
tnsr::abb<DataType, 3, Frame::Inertial>
compute_expected_lhs_spacetime_index_subsets(
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

// Aliases used for test_large_equation()
template <typename DataType, size_t Dim>
using result_tensor_type = tnsr::aa<DataType, Dim>;
template <typename DataType, size_t Dim>
using spacetime_deriv_gauge_function_type = tnsr::ab<DataType, Dim>;
template <typename DataType>
using pi_two_normals_type = Scalar<DataType>;
template <typename DataType, size_t Dim>
using pi_type = tnsr::aa<DataType, Dim>;
template <typename DataType>
using gamma0_type = Scalar<DataType>;
template <typename DataType, size_t Dim>
using normal_spacetime_one_form_type = tnsr::a<DataType, Dim>;
template <typename DataType, size_t Dim>
using gauge_constraint_type = tnsr::a<DataType, Dim>;
template <typename DataType, size_t Dim>
using spacetime_metric_type = tnsr::aa<DataType, Dim>;
template <typename DataType>
using normal_dot_gauge_constraint_type = Scalar<DataType>;
template <typename DataType, size_t Dim>
using christoffel_second_kind_type = tnsr::Abb<DataType, Dim>;
template <typename DataType, size_t Dim>
using gauge_function_type = tnsr::a<DataType, Dim>;
template <typename DataType, size_t Dim>
using pi_2_up_type = tnsr::aB<DataType, Dim>;
template <typename DataType, size_t Dim>
using phi_1_up_type = tnsr::Iaa<DataType, Dim>;
template <typename DataType, size_t Dim>
using phi_3_up_type = tnsr::iaB<DataType, Dim>;
template <typename DataType, size_t Dim>
using christoffel_first_kind_3_up_type = tnsr::abC<DataType, Dim>;
template <typename DataType, size_t Dim>
using pi_one_normal_type = tnsr::a<DataType, Dim>;
template <typename DataType, size_t Dim>
using inverse_spatial_metric_type = tnsr::II<DataType, Dim>;
template <typename DataType, size_t Dim>
using d_phi_type = tnsr::ijaa<DataType, Dim>;
template <typename DataType>
using lapse_type = Scalar<DataType>;
template <typename DataType>
using gamma1gamma2_type = Scalar<DataType>;
template <typename DataType, size_t Dim>
using shift_dot_three_index_constraint_type = tnsr::aa<DataType, Dim>;
template <typename DataType, size_t Dim>
using shift_type = tnsr::I<DataType, Dim>;
template <typename DataType, size_t Dim>
using d_pi_type = tnsr::iaa<DataType, Dim>;

// Computes the Generalized Harmonic equation for the time derivative,
// \f$\partial_t \Pi_{ab} \f$ (eq 36 of \cite Lindblom2005qh). This is taken
// from the implementation of `gh::TimeDerivative`.
template <typename DataType, size_t Dim>
result_tensor_type<DataType, Dim> compute_expected_large_equation(
    const spacetime_deriv_gauge_function_type<DataType, Dim>&
        spacetime_deriv_gauge_function,
    const pi_two_normals_type<DataType>& pi_two_normals,
    const pi_type<DataType, Dim>& pi, const gamma0_type<DataType>& gamma0,
    const normal_spacetime_one_form_type<DataType, Dim>&
        normal_spacetime_one_form,
    const gauge_constraint_type<DataType, Dim>& gauge_constraint,
    const spacetime_metric_type<DataType, Dim>& spacetime_metric,
    const normal_dot_gauge_constraint_type<DataType>&
        normal_dot_gauge_constraint,
    const christoffel_second_kind_type<DataType, Dim>& christoffel_second_kind,
    const gauge_function_type<DataType, Dim>& gauge_function,
    const pi_2_up_type<DataType, Dim>& pi_2_up,
    const phi_1_up_type<DataType, Dim>& phi_1_up,
    const phi_3_up_type<DataType, Dim>& phi_3_up,
    const christoffel_first_kind_3_up_type<DataType, Dim>&
        christoffel_first_kind_3_up,
    const pi_one_normal_type<DataType, Dim>& pi_one_normal,
    const inverse_spatial_metric_type<DataType, Dim>& inverse_spatial_metric,
    const d_phi_type<DataType, Dim>& d_phi, const lapse_type<DataType>& lapse,
    const gamma1gamma2_type<DataType>& gamma1gamma2,
    const shift_dot_three_index_constraint_type<DataType, Dim>&
        shift_dot_three_index_constraint,
    const shift_type<DataType, Dim>& shift,
    const d_pi_type<DataType, Dim>& d_pi) {
  result_tensor_type<DataType, Dim> expected_result{};

  for (size_t mu = 0; mu < Dim + 1; ++mu) {
    for (size_t nu = mu; nu < Dim + 1; ++nu) {
      expected_result.get(mu, nu) =
          -spacetime_deriv_gauge_function.get(mu, nu) -
          spacetime_deriv_gauge_function.get(nu, mu) -
          0.5 * get(pi_two_normals) * pi.get(mu, nu) +
          get(gamma0) *
              (normal_spacetime_one_form.get(mu) * gauge_constraint.get(nu) +
               normal_spacetime_one_form.get(nu) * gauge_constraint.get(mu)) -
          get(gamma0) * spacetime_metric.get(mu, nu) *
              get(normal_dot_gauge_constraint);

      for (size_t delta = 0; delta < Dim + 1; ++delta) {
        expected_result.get(mu, nu) +=
            2.0 * christoffel_second_kind.get(delta, mu, nu) *
                gauge_function.get(delta) -
            2.0 * pi.get(mu, delta) * pi_2_up.get(nu, delta);

        for (size_t n = 0; n < Dim; ++n) {
          expected_result.get(mu, nu) +=
              2.0 * phi_1_up.get(n, mu, delta) * phi_3_up.get(n, nu, delta);
        }

        for (size_t alpha = 0; alpha < Dim + 1; ++alpha) {
          expected_result.get(mu, nu) -=
              2.0 * christoffel_first_kind_3_up.get(mu, alpha, delta) *
              christoffel_first_kind_3_up.get(nu, delta, alpha);
        }
      }

      for (size_t m = 0; m < Dim; ++m) {
        expected_result.get(mu, nu) -=
            pi_one_normal.get(m + 1) * phi_1_up.get(m, mu, nu);

        for (size_t n = 0; n < Dim; ++n) {
          expected_result.get(mu, nu) -=
              inverse_spatial_metric.get(m, n) * d_phi.get(m, n, mu, nu);
        }
      }

      expected_result.get(mu, nu) *= get(lapse);

      expected_result.get(mu, nu) +=
          get(gamma1gamma2) * shift_dot_three_index_constraint.get(mu, nu);

      for (size_t m = 0; m < Dim; ++m) {
        // DualFrame term
        expected_result.get(mu, nu) += shift.get(m) * d_pi.get(m, mu, nu);
      }
    }
  }

  return expected_result;
}

// Includes an expression with addition, subtraction, an inner product, an outer
// product, a contraction, and a scalar
template <typename Generator, typename DataType>
void test_mixed_arithmetic_ops(const gsl::not_null<Generator*> generator,
                               const DataType& used_for_size) {
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
      compute_expected_mixed_arithmetic_ops(R, S, G, H, T, used_for_size);
  // \f$L_{a} = R_{ab} S^{b} + G_{a} - H_{ba}{}^{b} T\f$
  // [use_evaluate_to_return_result]
  result_tensor_type actual_result_tensor_returned = tenex::evaluate<ti::a>(
      R(ti::a, ti::b) * S(ti::B) + G(ti::a) - H(ti::b, ti::a, ti::B) * T());
  // [use_evaluate_to_return_result]

  // [use_evaluate_with_result_as_arg]
  result_tensor_type actual_result_tensor_filled{};
  tenex::evaluate<ti::a>(
      make_not_null(&actual_result_tensor_filled),
      R(ti::a, ti::b) * S(ti::B) + G(ti::a) - H(ti::b, ti::a, ti::B) * T());
  // [use_evaluate_with_result_as_arg]

  for (size_t a = 0; a < 4; a++) {
    CHECK_ITERABLE_APPROX(actual_result_tensor_returned.get(a),
                          expected_result_tensor.get(a));
    CHECK_ITERABLE_APPROX(actual_result_tensor_filled.get(a),
                          expected_result_tensor.get(a));
  }

  // Test with TempTensor for LHS tensor
  if constexpr (is_derived_of_vector_impl_v<DataType>) {
    Variables<tmpl::list<::Tags::TempTensor<1, result_tensor_type>>>
        actual_result_tensor_temp_var{used_for_size.size()};
    result_tensor_type& actual_result_tensor_temp =
        get<::Tags::TempTensor<1, result_tensor_type>>(
            actual_result_tensor_temp_var);
    ::tenex::evaluate<ti::a>(
        make_not_null(&actual_result_tensor_temp),
        R(ti::a, ti::b) * S(ti::B) + G(ti::a) - H(ti::b, ti::a, ti::B) * T());

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
template <typename Generator, typename DataType>
void test_rhs_spacetime_index_subsets(const gsl::not_null<Generator*> generator,
                                      const DataType& used_for_size) {
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
      compute_expected_rhs_spacetime_index_subsets(
          spatial_metric, spacetime_metric, used_for_size);
  // \f$\alpha = \sqrt{\gamma^{ij} g_{jt} g_{it} - g_{tt}}\f$
  const Scalar<DataType> actual_result_tensor_returned = tenex::evaluate(
      sqrt(spatial_metric(ti::I, ti::J) * spacetime_metric(ti::j, ti::t) *
               spacetime_metric(ti::i, ti::t) -
           spacetime_metric(ti::t, ti::t)));
  Scalar<DataType> actual_result_tensor_filled{};
  tenex::evaluate(
      make_not_null(&actual_result_tensor_filled),
      sqrt(spatial_metric(ti::I, ti::J) * spacetime_metric(ti::j, ti::t) *
               spacetime_metric(ti::i, ti::t) -
           spacetime_metric(ti::t, ti::t)));

  CHECK_ITERABLE_APPROX(actual_result_tensor_returned.get(),
                        expected_result_tensor.get());
  CHECK_ITERABLE_APPROX(actual_result_tensor_filled.get(),
                        expected_result_tensor.get());

  // Test with TempTensor for LHS tensor
  if constexpr (is_derived_of_vector_impl_v<DataType>) {
    Variables<tmpl::list<::Tags::TempTensor<1, Tensor<DataType>>>>
        actual_result_tensor_temp_var{used_for_size.size()};
    Scalar<DataType>& actual_result_tensor_temp =
        get<::Tags::TempTensor<1, Tensor<DataType>>>(
            actual_result_tensor_temp_var);
    ::tenex::evaluate(
        make_not_null(&actual_result_tensor_temp),
        sqrt(spatial_metric(ti::I, ti::J) * spacetime_metric(ti::j, ti::t) *
                 spacetime_metric(ti::i, ti::t) -
             spacetime_metric(ti::t, ti::t)));

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
template <typename Generator, typename DataType>
void test_lhs_spacetime_index_subsets(const gsl::not_null<Generator*> generator,
                                      const DataType& used_for_size) {
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
      compute_expected_lhs_spacetime_index_subsets(alpha, beta, pi, phi,
                                                   used_for_size);
  result_tensor_type actual_result_tensor_filled{};
  // \f$\partial_t g_{ab} = -\alpha \Pi_{ab} + \beta^i \Phi_{iab}\f$
  tenex::evaluate<ti::t, ti::a, ti::b>(
      make_not_null(&actual_result_tensor_filled),
      -1.0 * alpha() * pi(ti::a, ti::b) +
          beta(ti::I) * phi(ti::i, ti::a, ti::b));
  // \f$\partial_i g_{ab} = \Phi_{iab}\f$
  tenex::evaluate<ti::i, ti::a, ti::b>(
      make_not_null(&actual_result_tensor_filled), phi(ti::i, ti::a, ti::b));

  for (size_t c = 0; c < 4; c++) {
    for (size_t a = 0; a < 4; a++) {
      for (size_t b = 0; b < 4; b++) {
        CHECK_ITERABLE_APPROX(actual_result_tensor_filled.get(c, a, b),
                              expected_result_tensor.get(c, a, b));
      }
    }
  }

  // Test with TempTensor for LHS tensor
  if constexpr (is_derived_of_vector_impl_v<DataType>) {
    Variables<tmpl::list<::Tags::TempTensor<1, result_tensor_type>>>
        actual_result_tensor_temp_var{used_for_size.size()};
    result_tensor_type& actual_result_tensor_temp =
        get<::Tags::TempTensor<1, result_tensor_type>>(
            actual_result_tensor_temp_var);
    tenex::evaluate<ti::t, ti::a, ti::b>(
        make_not_null(&actual_result_tensor_temp),
        -1.0 * alpha() * pi(ti::a, ti::b) +
            beta(ti::I) * phi(ti::i, ti::a, ti::b));
    tenex::evaluate<ti::i, ti::a, ti::b>(
        make_not_null(&actual_result_tensor_temp), phi(ti::i, ti::a, ti::b));

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

// This test case is the Generalized Harmonic equation for the time derivative,
// \f$\partial_t \Pi_{ab} \f$ (eq 36 of \cite Lindblom2005qh).
//
// This test case is distinct in that it is a large equation and tests calls to
// `tenex::update`.
template <typename Generator, typename DataType>
void test_large_equation(const gsl::not_null<Generator*> generator,
                         const DataType& used_for_size) {
  constexpr size_t Dim = 3;

  using result_tensor_type = result_tensor_type<DataType, Dim>;
  using spacetime_deriv_gauge_function_type =
      spacetime_deriv_gauge_function_type<DataType, Dim>;
  using pi_two_normals_type = pi_two_normals_type<DataType>;
  using pi_type = pi_type<DataType, Dim>;
  using gamma0_type = gamma0_type<DataType>;
  using normal_spacetime_one_form_type =
      normal_spacetime_one_form_type<DataType, Dim>;
  using gauge_constraint_type = gauge_constraint_type<DataType, Dim>;
  using spacetime_metric_type = spacetime_metric_type<DataType, Dim>;
  using normal_dot_gauge_constraint_type =
      normal_dot_gauge_constraint_type<DataType>;
  using christoffel_second_kind_type =
      christoffel_second_kind_type<DataType, Dim>;
  using gauge_function_type = gauge_function_type<DataType, Dim>;
  using pi_2_up_type = pi_2_up_type<DataType, Dim>;
  using phi_1_up_type = phi_1_up_type<DataType, Dim>;
  using phi_3_up_type = phi_3_up_type<DataType, Dim>;
  using christoffel_first_kind_3_up_type =
      christoffel_first_kind_3_up_type<DataType, Dim>;
  using pi_one_normal_type = pi_one_normal_type<DataType, Dim>;
  using inverse_spatial_metric_type =
      inverse_spatial_metric_type<DataType, Dim>;
  using d_phi_type = d_phi_type<DataType, Dim>;
  using lapse_type = lapse_type<DataType>;
  using gamma1gamma2_type = gamma1gamma2_type<DataType>;
  using shift_dot_three_index_constraint_type =
      shift_dot_three_index_constraint_type<DataType, Dim>;
  using shift_type = shift_type<DataType, Dim>;
  using d_pi_type = d_pi_type<DataType, Dim>;

  std::uniform_real_distribution<> distribution(0.1, 1.0);

  // Not inputs to the eq, but needed to make some variables
  // that are computed by raising something in order for symmetric
  // assumptions to hold
  const auto inverse_spacetime_metric =
      make_with_random_values<tnsr::AA<DataType, Dim>>(generator, distribution,
                                                       used_for_size);
  const auto phi = make_with_random_values<tnsr::iaa<DataType, Dim>>(
      generator, distribution, used_for_size);

  // RHS: spacetime_deriv_gauge_function
  const spacetime_deriv_gauge_function_type spacetime_deriv_gauge_function =
      make_with_random_values<spacetime_deriv_gauge_function_type>(
          generator, distribution, used_for_size);

  // RHS: pi_two_normals
  const pi_two_normals_type pi_two_normals =
      make_with_random_values<pi_two_normals_type>(generator, distribution,
                                                   used_for_size);

  // RHS: pi
  const pi_type pi =
      make_with_random_values<pi_type>(generator, distribution, used_for_size);

  // RHS: gamma0
  const gamma0_type gamma0 = make_with_random_values<gamma0_type>(
      generator, distribution, used_for_size);

  // RHS: normal_spacetime_one_form
  const normal_spacetime_one_form_type normal_spacetime_one_form =
      make_with_random_values<normal_spacetime_one_form_type>(
          generator, distribution, used_for_size);

  // RHS: gauge_constraint
  const gauge_constraint_type gauge_constraint =
      make_with_random_values<gauge_constraint_type>(generator, distribution,
                                                     used_for_size);

  // RHS: spacetime_metric
  const spacetime_metric_type spacetime_metric =
      make_with_random_values<spacetime_metric_type>(generator, distribution,
                                                     used_for_size);

  // RHS: normal_dot_gauge_constraint
  const normal_dot_gauge_constraint_type normal_dot_gauge_constraint =
      make_with_random_values<normal_dot_gauge_constraint_type>(
          generator, distribution, used_for_size);

  // RHS: christoffel_second_kind
  const christoffel_second_kind_type christoffel_second_kind =
      make_with_random_values<christoffel_second_kind_type>(
          generator, distribution, used_for_size);

  // RHS: gauge_function
  const gauge_function_type gauge_function =
      make_with_random_values<gauge_function_type>(generator, distribution,
                                                   used_for_size);

  // RHS: pi_2_up
  pi_2_up_type pi_2_up(used_for_size);
  for (size_t a = 0; a < Dim + 1; a++) {
    for (size_t b = 0; b < Dim + 1; b++) {
      pi_2_up.get(a, b) = 0.0;
      for (size_t c = 0; c < Dim + 1; c++) {
        pi_2_up.get(a, b) += inverse_spacetime_metric.get(c, b) * pi.get(a, c);
      }
    }
  }

  // RHS: phi_3_up
  phi_3_up_type phi_3_up(used_for_size);
  for (size_t i = 0; i < Dim; i++) {
    for (size_t a = 0; a < Dim + 1; a++) {
      for (size_t b = 0; b < Dim + 1; b++) {
        phi_3_up.get(i, a, b) = 0.0;
        for (size_t c = 0; c < Dim + 1; c++) {
          phi_3_up.get(i, a, b) +=
              inverse_spacetime_metric.get(c, b) * phi.get(i, a, c);
        }
      }
    }
  }

  // RHS: christoffel_first_kind_3_up
  const christoffel_first_kind_3_up_type christoffel_first_kind_3_up =
      make_with_random_values<christoffel_first_kind_3_up_type>(
          generator, distribution, used_for_size);

  // RHS: pi_one_normal
  const pi_one_normal_type pi_one_normal =
      make_with_random_values<pi_one_normal_type>(generator, distribution,
                                                  used_for_size);

  // RHS: inverse_spatial_metric
  inverse_spatial_metric_type inverse_spatial_metric(used_for_size);
  for (size_t i = 0; i < Dim; i++) {
    for (size_t j = 0; j < Dim; j++) {
      inverse_spatial_metric.get(i, j) =
          inverse_spacetime_metric.get(i + 1, j + 1);
    }
  }

  // RHS: phi_1_up
  // Note: arg out of order down here bc needs inverse_spatial_metric
  phi_1_up_type phi_1_up(used_for_size);
  for (size_t i = 0; i < Dim; i++) {
    for (size_t a = 0; a < Dim + 1; a++) {
      for (size_t b = a; b < Dim + 1; b++) {
        phi_1_up.get(i, a, b) = 0.0;
        for (size_t j = 0; j < Dim; j++) {
          phi_1_up.get(i, a, b) +=
              inverse_spatial_metric.get(i, j) * phi.get(j, a, b);
        }
      }
    }
  }

  // RHS: d_phi
  const d_phi_type d_phi = make_with_random_values<d_phi_type>(
      generator, distribution, used_for_size);

  // RHS: lapse
  const lapse_type lapse = make_with_random_values<lapse_type>(
      generator, distribution, used_for_size);

  // RHS: gamma1gamma2
  const gamma1gamma2_type gamma1gamma2 =
      make_with_random_values<gamma1gamma2_type>(generator, distribution,
                                                 used_for_size);

  // RHS: shift_dot_three_index_constraint
  const shift_dot_three_index_constraint_type shift_dot_three_index_constraint =
      make_with_random_values<shift_dot_three_index_constraint_type>(
          generator, distribution, used_for_size);

  // RHS: shift
  const shift_type shift = make_with_random_values<shift_type>(
      generator, distribution, used_for_size);

  // RHS: d_pi
  const d_pi_type d_pi = make_with_random_values<d_pi_type>(
      generator, distribution, used_for_size);

  // LHS: dt_pi
  const result_tensor_type expected_result_tensor =
      compute_expected_large_equation(
          spacetime_deriv_gauge_function, pi_two_normals, pi, gamma0,
          normal_spacetime_one_form, gauge_constraint, spacetime_metric,
          normal_dot_gauge_constraint, christoffel_second_kind, gauge_function,
          pi_2_up, phi_1_up, phi_3_up, christoffel_first_kind_3_up,
          pi_one_normal, inverse_spatial_metric, d_phi, lapse, gamma1gamma2,
          shift_dot_three_index_constraint, shift, d_pi);

  // LHS: dt_pi
  // [use_update]
  result_tensor_type actual_result_tensor_filled{};
  tenex::evaluate<ti::a, ti::b>(
      make_not_null(&actual_result_tensor_filled),
      -spacetime_deriv_gauge_function(ti::a, ti::b) -
          spacetime_deriv_gauge_function(ti::b, ti::a) -
          0.5 * pi_two_normals() * pi(ti::a, ti::b) +
          gamma0() *
              (normal_spacetime_one_form(ti::a) * gauge_constraint(ti::b) +
               normal_spacetime_one_form(ti::b) * gauge_constraint(ti::a)) -
          gamma0() * spacetime_metric(ti::a, ti::b) *
              normal_dot_gauge_constraint() +
          2.0 * christoffel_second_kind(ti::C, ti::a, ti::b) *
              gauge_function(ti::c) -
          2.0 * pi(ti::a, ti::c) * pi_2_up(ti::b, ti::C));

  tenex::update<ti::a, ti::b>(
      make_not_null(&actual_result_tensor_filled),
      actual_result_tensor_filled(ti::a, ti::b) +
          2.0 * phi_1_up(ti::I, ti::a, ti::c) * phi_3_up(ti::i, ti::b, ti::C));

  tenex::update<ti::a, ti::b>(
      make_not_null(&actual_result_tensor_filled),
      actual_result_tensor_filled(ti::a, ti::b) -
          2.0 * christoffel_first_kind_3_up(ti::a, ti::d, ti::C) *
              christoffel_first_kind_3_up(ti::b, ti::c, ti::D));

  tenex::update<ti::a, ti::b>(
      make_not_null(&actual_result_tensor_filled),
      actual_result_tensor_filled(ti::a, ti::b) -
          pi_one_normal(ti::j) * phi_1_up(ti::J, ti::a, ti::b));

  tenex::update<ti::a, ti::b>(make_not_null(&actual_result_tensor_filled),
                              actual_result_tensor_filled(ti::a, ti::b) -
                                  inverse_spatial_metric(ti::J, ti::K) *
                                      d_phi(ti::j, ti::k, ti::a, ti::b));

  tenex::update<ti::a, ti::b>(
      make_not_null(&actual_result_tensor_filled),
      actual_result_tensor_filled(ti::a, ti::b) * lapse() +
          gamma1gamma2() * shift_dot_three_index_constraint(ti::a, ti::b) +
          shift(ti::J) * d_pi(ti::j, ti::a, ti::b));
  // [use_update]

  Approx approx = Approx::custom().epsilon(1e-12).scale(1.0);
  CHECK_ITERABLE_CUSTOM_APPROX(actual_result_tensor_filled,
                               expected_result_tensor, approx);

  // Test with TempTensor for LHS tensor
  if constexpr (is_derived_of_vector_impl_v<DataType>) {
    Variables<tmpl::list<
        ::Tags::TempTensor<0, result_tensor_type>,
        ::Tags::TempTensor<1, spacetime_deriv_gauge_function_type>,
        ::Tags::TempTensor<2, pi_two_normals_type>,
        ::Tags::TempTensor<3, pi_type>, ::Tags::TempTensor<4, gamma0_type>,
        ::Tags::TempTensor<5, normal_spacetime_one_form_type>,
        ::Tags::TempTensor<6, gauge_constraint_type>,
        ::Tags::TempTensor<7, spacetime_metric_type>,
        ::Tags::TempTensor<8, normal_dot_gauge_constraint_type>,
        ::Tags::TempTensor<9, christoffel_second_kind_type>,
        ::Tags::TempTensor<10, gauge_function_type>,
        ::Tags::TempTensor<11, pi_2_up_type>,
        ::Tags::TempTensor<12, phi_1_up_type>,
        ::Tags::TempTensor<13, phi_3_up_type>,
        ::Tags::TempTensor<14, christoffel_first_kind_3_up_type>,
        ::Tags::TempTensor<15, pi_one_normal_type>,
        ::Tags::TempTensor<16, inverse_spatial_metric_type>,
        ::Tags::TempTensor<17, d_phi_type>, ::Tags::TempTensor<18, lapse_type>,
        ::Tags::TempTensor<19, gamma1gamma2_type>,
        ::Tags::TempTensor<20, shift_dot_three_index_constraint_type>,
        ::Tags::TempTensor<21, shift_type>, ::Tags::TempTensor<22, d_pi_type>>>
        vars{used_for_size.size()};

    // RHS: spacetime_deriv_gauge_function
    spacetime_deriv_gauge_function_type& spacetime_deriv_gauge_function_temp =
        get<::Tags::TempTensor<1, spacetime_deriv_gauge_function_type>>(vars);
    spacetime_deriv_gauge_function_temp = spacetime_deriv_gauge_function;

    // RHS: pi_two_normals
    pi_two_normals_type& pi_two_normals_temp =
        get<::Tags::TempTensor<2, pi_two_normals_type>>(vars);
    pi_two_normals_temp = pi_two_normals;

    // RHS: pi
    pi_type& pi_temp = get<::Tags::TempTensor<3, pi_type>>(vars);
    pi_temp = pi;

    // RHS: gamma0
    gamma0_type& gamma0_temp = get<::Tags::TempTensor<4, gamma0_type>>(vars);
    gamma0_temp = gamma0;

    // RHS: normal_spacetime_one_form
    normal_spacetime_one_form_type& normal_spacetime_one_form_temp =
        get<::Tags::TempTensor<5, normal_spacetime_one_form_type>>(vars);
    normal_spacetime_one_form_temp = normal_spacetime_one_form;

    // RHS: gauge_constraint
    gauge_constraint_type& gauge_constraint_temp =
        get<::Tags::TempTensor<6, gauge_constraint_type>>(vars);
    gauge_constraint_temp = gauge_constraint;

    // RHS: spacetime_metric
    spacetime_metric_type& spacetime_metric_temp =
        get<::Tags::TempTensor<7, spacetime_metric_type>>(vars);
    spacetime_metric_temp = spacetime_metric;

    // RHS: normal_dot_gauge_constraint
    normal_dot_gauge_constraint_type& normal_dot_gauge_constraint_temp =
        get<::Tags::TempTensor<8, normal_dot_gauge_constraint_type>>(vars);
    normal_dot_gauge_constraint_temp = normal_dot_gauge_constraint;

    // RHS: christoffel_second_kind
    christoffel_second_kind_type& christoffel_second_kind_temp =
        get<::Tags::TempTensor<9, christoffel_second_kind_type>>(vars);
    christoffel_second_kind_temp = christoffel_second_kind;

    // RHS: gauge_function
    gauge_function_type& gauge_function_temp =
        get<::Tags::TempTensor<10, gauge_function_type>>(vars);
    gauge_function_temp = gauge_function;

    // RHS: pi_2_up
    pi_2_up_type& pi_2_up_temp =
        get<::Tags::TempTensor<11, pi_2_up_type>>(vars);
    pi_2_up_temp = pi_2_up;

    // RHS: phi_1_up
    phi_1_up_type& phi_1_up_temp =
        get<::Tags::TempTensor<12, phi_1_up_type>>(vars);
    phi_1_up_temp = phi_1_up;

    // RHS: phi_3_up
    phi_3_up_type& phi_3_up_temp =
        get<::Tags::TempTensor<13, phi_3_up_type>>(vars);
    phi_3_up_temp = phi_3_up;

    // RHS: christoffel_first_kind_3_up
    christoffel_first_kind_3_up_type& christoffel_first_kind_3_up_temp =
        get<::Tags::TempTensor<14, christoffel_first_kind_3_up_type>>(vars);
    christoffel_first_kind_3_up_temp = christoffel_first_kind_3_up;

    // RHS: pi_one_normal
    pi_one_normal_type& pi_one_normal_temp =
        get<::Tags::TempTensor<15, pi_one_normal_type>>(vars);
    pi_one_normal_temp = pi_one_normal;

    // RHS: inverse_spatial_metric
    inverse_spatial_metric_type& inverse_spatial_metric_temp =
        get<::Tags::TempTensor<16, inverse_spatial_metric_type>>(vars);
    inverse_spatial_metric_temp = inverse_spatial_metric;

    // RHS: d_phi
    d_phi_type& d_phi_temp = get<::Tags::TempTensor<17, d_phi_type>>(vars);
    d_phi_temp = d_phi;

    // RHS: lapse
    lapse_type& lapse_temp = get<::Tags::TempTensor<18, lapse_type>>(vars);
    lapse_temp = lapse;

    // RHS: gamma1gamma2
    gamma1gamma2_type& gamma1gamma2_temp =
        get<::Tags::TempTensor<19, gamma1gamma2_type>>(vars);
    gamma1gamma2_temp = gamma1gamma2;

    // RHS: shift_dot_three_index_constraint
    shift_dot_three_index_constraint_type&
        shift_dot_three_index_constraint_temp =
            get<::Tags::TempTensor<20, shift_dot_three_index_constraint_type>>(
                vars);
    shift_dot_three_index_constraint_temp = shift_dot_three_index_constraint;

    // RHS: shift
    shift_type& shift_temp = get<::Tags::TempTensor<21, shift_type>>(vars);
    shift_temp = shift;

    // RHS: d_pi
    d_pi_type& d_pi_temp = get<::Tags::TempTensor<22, d_pi_type>>(vars);
    d_pi_temp = d_pi;

    // LHS: dt_pi
    result_tensor_type& actual_result_tensor_temp =
        get<::Tags::TempTensor<0, result_tensor_type>>(vars);

    tenex::evaluate<ti::a, ti::b>(
        make_not_null(&actual_result_tensor_temp),
        -spacetime_deriv_gauge_function_temp(ti::a, ti::b) -
            spacetime_deriv_gauge_function_temp(ti::b, ti::a) -
            0.5 * pi_two_normals_temp() * pi_temp(ti::a, ti::b) +
            gamma0_temp() * (normal_spacetime_one_form_temp(ti::a) *
                                 gauge_constraint_temp(ti::b) +
                             normal_spacetime_one_form_temp(ti::b) *
                                 gauge_constraint_temp(ti::a)) -
            gamma0_temp() * spacetime_metric_temp(ti::a, ti::b) *
                normal_dot_gauge_constraint_temp() +
            2.0 * christoffel_second_kind_temp(ti::C, ti::a, ti::b) *
                gauge_function_temp(ti::c) -
            2.0 * pi_temp(ti::a, ti::c) * pi_2_up_temp(ti::b, ti::C) +
            2.0 * phi_3_up_temp(ti::i, ti::a, ti::C) *
                phi_1_up_temp(ti::I, ti::b, ti::c) -
            2.0 * christoffel_first_kind_3_up_temp(ti::a, ti::d, ti::C) *
                christoffel_first_kind_3_up_temp(ti::b, ti::c, ti::D));

    tenex::update<ti::a, ti::b>(
        make_not_null(&actual_result_tensor_temp),
        actual_result_tensor_temp(ti::a, ti::b) -
            pi_one_normal_temp(ti::j) * phi_1_up_temp(ti::J, ti::a, ti::b));

    tenex::update<ti::a, ti::b>(make_not_null(&actual_result_tensor_temp),
                                actual_result_tensor_temp(ti::a, ti::b) -
                                    inverse_spatial_metric_temp(ti::J, ti::K) *
                                        d_phi_temp(ti::j, ti::k, ti::a, ti::b));

    tenex::update<ti::a, ti::b>(
        make_not_null(&actual_result_tensor_temp),
        actual_result_tensor_temp(ti::a, ti::b) * lapse_temp() +
            gamma1gamma2_temp() *
                shift_dot_three_index_constraint_temp(ti::a, ti::b) +
            shift_temp(ti::J) * d_pi_temp(ti::j, ti::a, ti::b));

    CHECK_ITERABLE_CUSTOM_APPROX(actual_result_tensor_temp,
                                 expected_result_tensor, approx);
  }
}

// Tests the assignment of a RHS `double` to a LHS tensor. Includes testing
// multiple calls to `evaluate` that fill the time and spatial components of the
// result tensor, respectively.
template <typename DataType>
void test_assign_double(const DataType& used_for_size) {
  // [assign_double_to_index_subsets]
  tnsr::iab<DataType, 3, Frame::Inertial> L_tensor(used_for_size);
  const gsl::not_null<tnsr::iab<DataType, 3, Frame::Inertial>*> L =
      make_not_null(&L_tensor);

  // \f$L_{itt} = 8.2\f$
  tenex::evaluate<ti::i, ti::t, ti::t>(L, 8.2);
  // \f$L_{itj} = 2.2\f$
  tenex::evaluate<ti::i, ti::t, ti::j>(L, 2.2);
  // \f$L_{ijt} = -1.9\f$
  tenex::evaluate<ti::i, ti::j, ti::t>(L, -1.9);
  // \f$L_{ijk} = -0.5\f$
  tenex::evaluate<ti::i, ti::j, ti::k>(L, -0.5);
  // [assign_double_to_index_subsets]

  for (size_t i = 0; i < 3; i++) {
    for (size_t a = 0; a < 4; a++) {
      for (size_t b = 0; b < 4; b++) {
        DataType expected_result =
            make_with_value<DataType>(used_for_size, 0.0);

        if (a == 0 and b == 0) {
          expected_result = 8.2;
        } else if (a == 0) {
          expected_result = 2.2;
        } else if (b == 0) {
          expected_result = -1.9;
        } else {
          expected_result = -0.5;
        }

        CHECK(L->get(i, a, b) == expected_result);
      }
    }
  }

  // Test with TempTensor for LHS tensor
  if constexpr (is_derived_of_vector_impl_v<DataType>) {
    Variables<tmpl::list<
        ::Tags::TempTensor<1, tnsr::iab<DataType, 3, Frame::Inertial>>>>
        L_temp_var{used_for_size.size()};
    tnsr::iab<DataType, 3, Frame::Inertial>& L_temp_tensor =
        get<::Tags::TempTensor<1, tnsr::iab<DataType, 3, Frame::Inertial>>>(
            L_temp_var);
    const gsl::not_null<tnsr::iab<DataType, 3, Frame::Inertial>*> L_temp =
        make_not_null(&L_temp_tensor);

    // \f$L_{itt} = 8.2\f$
    tenex::evaluate<ti::i, ti::t, ti::t>(L_temp, 8.2);
    // \f$L_{itj} = 2.2\f$
    tenex::evaluate<ti::i, ti::t, ti::j>(L_temp, 2.2);
    // \f$L_{ijt} = -1.9\f$
    tenex::evaluate<ti::i, ti::j, ti::t>(L_temp, -1.9);
    // \f$L_{ijk} = -0.5\f$
    tenex::evaluate<ti::i, ti::j, ti::k>(L_temp, -0.5);

    for (size_t i = 0; i < 3; i++) {
      for (size_t a = 0; a < 4; a++) {
        for (size_t b = 0; b < 4; b++) {
          DataType expected_result =
              make_with_value<DataType>(used_for_size, 0.0);

          if (a == 0 and b == 0) {
            expected_result = 8.2;
          } else if (a == 0) {
            expected_result = 2.2;
          } else if (b == 0) {
            expected_result = -1.9;
          } else {
            expected_result = -0.5;
          }

          CHECK(L_temp->get(i, a, b) == expected_result);
        }
      }
    }
  }
}

// Test cases include equations with a mixture of arithmetic operations or more
// that one assignment of the LHS tensor (more than one call to
// `tenex::evaluate` or `tenex::update`)
template <typename Generator, typename DataType>
void test_mixed_operations(const gsl::not_null<Generator*> generator,
                           const DataType& used_for_size) {
  test_mixed_arithmetic_ops(generator, used_for_size);
  test_rhs_spacetime_index_subsets(generator, used_for_size);
  test_lhs_spacetime_index_subsets(generator, used_for_size);
  test_large_equation(generator, used_for_size);
  test_assign_double(used_for_size);
}
}  // namespace

SPECTRE_TEST_CASE("Unit.DataStructures.Tensor.Expression.MixedOperations",
                  "[DataStructures][Unit]") {
  MAKE_GENERATOR(generator);

  test_mixed_operations(make_not_null(&generator),
                        std::numeric_limits<double>::signaling_NaN());
  test_mixed_operations(
      make_not_null(&generator),
      std::complex<double>(std::numeric_limits<double>::signaling_NaN(),
                           std::numeric_limits<double>::signaling_NaN()));
  test_mixed_operations(
      make_not_null(&generator),
      DataVector(5, std::numeric_limits<double>::signaling_NaN()));
  test_mixed_operations(
      make_not_null(&generator),
      ComplexDataVector(5, std::numeric_limits<double>::signaling_NaN()));
}
