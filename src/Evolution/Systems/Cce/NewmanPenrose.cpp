// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Evolution/Systems/Cce/NewmanPenrose.hpp"

#include "DataStructures/ComplexDataVector.hpp"
#include "DataStructures/SpinWeighted.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Evolution/Systems/Cce/Tags.hpp"
#include "NumericalAlgorithms/SpinWeightedSphericalHarmonics/SwshDerivatives.hpp"
#include "NumericalAlgorithms/SpinWeightedSphericalHarmonics/SwshInterpolation.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/TMPL.hpp"

namespace Cce {
namespace {
void newman_penrose_alpha_impl(
    const gsl::not_null<SpinWeighted<ComplexDataVector, -1>*> np_alpha,
    const SpinWeighted<ComplexDataVector, 2>& bondi_j,
    const SpinWeighted<ComplexDataVector, 3>& eth_j,
    const SpinWeighted<ComplexDataVector, 1>& ethbar_j,
    const SpinWeighted<ComplexDataVector, 0>& bondi_k,
    const SpinWeighted<ComplexDataVector, 0>& bondi_r,
    const SpinWeighted<ComplexDataVector, 1>& bondi_q,
    const SpinWeighted<ComplexDataVector, 1>& eth_beta,
    const SpinWeighted<ComplexDataVector, 0>& one_minus_y) {

  // Intentionally using expressions with auto. This code is only used for
  // observing, and should be profiled if used more often
  const auto one_plus_k = 1. + bondi_k;
  const auto sqrt_one_plus_k = sqrt(one_plus_k);
  const auto q_plus_two_eth_beta = bondi_q + 2. * eth_beta;

  *np_alpha =
      one_minus_y / (32. * bondi_r)
      * (
          1. / sqrt_one_plus_k *
          (
            (square(conj(bondi_j)) * eth_j) / (bondi_k * one_plus_k)
            +
            1. / bondi_k *
            ( bondi_j * conj(eth_j) + conj(bondi_j) * ethbar_j
              - conj(ethbar_j))
            + ( 2. * conj(bondi_j) * q_plus_two_eth_beta
                - 3. * conj(ethbar_j) )
          )
          - 2. * sqrt_one_plus_k * conj(q_plus_two_eth_beta)
        );
}

void newman_penrose_beta_impl(
    const gsl::not_null<SpinWeighted<ComplexDataVector, +1>*> np_beta,
    const SpinWeighted<ComplexDataVector, 2>& bondi_j,
    const SpinWeighted<ComplexDataVector, 3>& eth_j,
    const SpinWeighted<ComplexDataVector, 1>& ethbar_j,
    const SpinWeighted<ComplexDataVector, 0>& bondi_k,
    const SpinWeighted<ComplexDataVector, 0>& bondi_r,
    const SpinWeighted<ComplexDataVector, 1>& bondi_q,
    const SpinWeighted<ComplexDataVector, 1>& eth_beta,
    const SpinWeighted<ComplexDataVector, 0>& one_minus_y) {

  // Intentionally using expressions with auto. This code is only used for
  // observing, and should be profiled if used more often
  const auto one_plus_k = 1. + bondi_k;
  const auto sqrt_one_plus_k = sqrt(one_plus_k);
  const auto q_plus_two_eth_beta = bondi_q + 2. * eth_beta;

  *np_beta =
      one_minus_y / (32. * bondi_r)
      * (
          1. / sqrt_one_plus_k *
          (
            ( - square(bondi_j) * conj(eth_j) / (bondi_k * one_plus_k)
              +
              1. / bondi_k *
              ( - bondi_j * conj(ethbar_j) - conj(bondi_j) * eth_j
                + ethbar_j)
              + ( 2. * bondi_j * conj(q_plus_two_eth_beta)
                  - 3. * ethbar_j )
            )
          )
          - 2. * sqrt_one_plus_k * q_plus_two_eth_beta
       );
}

void newman_penrose_gamma_impl(
    const gsl::not_null<SpinWeighted<ComplexDataVector, 0>*> np_gamma,
    const SpinWeighted<ComplexDataVector, 2>& bondi_j,
    const SpinWeighted<ComplexDataVector, 2>& dy_j,
    const SpinWeighted<ComplexDataVector, 3>& eth_j,
    const SpinWeighted<ComplexDataVector, 1>& ethbar_j,
    const SpinWeighted<ComplexDataVector, 0>& bondi_k,
    const SpinWeighted<ComplexDataVector, 2>& bondi_h,
    const SpinWeighted<ComplexDataVector, 0>& bondi_r,
    const SpinWeighted<ComplexDataVector, 1>& bondi_u,
    const SpinWeighted<ComplexDataVector, 2>& eth_u,
    const SpinWeighted<ComplexDataVector, 0>& ethbar_u,
    const SpinWeighted<ComplexDataVector, 0>& bondi_w,
    const SpinWeighted<ComplexDataVector, 0>& dy_w,
    const SpinWeighted<ComplexDataVector, 0>& exp_2_beta,
    const SpinWeighted<ComplexDataVector, 0>& one_minus_y) {

  // Intentionally using expressions with auto. This code is only used for
  // observing, and should be profiled if used more often
  const auto one_plus_k = 1. + bondi_k;

  *np_gamma =
    1. / ( sqrt(32.) * exp_2_beta )
      * ( 0.5 / one_plus_k
          * ( one_minus_y * ( 0.5 * one_minus_y / bondi_r + bondi_w )
              * ( conj(bondi_j) * dy_j - bondi_j * conj(dy_j) )
              + ( 2. * conj(bondi_h) * bondi_j - 2. * bondi_h * conj(bondi_j)
                  + bondi_u * ( bondi_j * conj(eth_j)
                                - conj(bondi_j) * ethbar_j)
                  + conj(bondi_u) * ( bondi_j * conj(ethbar_j)
                                      - conj(bondi_j) * eth_j )
                )
             )
          + 2. * one_minus_y * dy_w
          + ( 2. * bondi_w + bondi_j * conj(eth_u) - conj(bondi_j) * eth_u
              + bondi_k * ( ethbar_u - conj(ethbar_u) ) )
    );
}

void newman_penrose_epsilon_impl(
    const gsl::not_null<SpinWeighted<ComplexDataVector, 0>*> np_epsilon,
    const SpinWeighted<ComplexDataVector, 2>& bondi_j,
    const SpinWeighted<ComplexDataVector, 2>& dy_j,
    const SpinWeighted<ComplexDataVector, 0>& bondi_k,
    const SpinWeighted<ComplexDataVector, 0>& bondi_r,
    const SpinWeighted<ComplexDataVector, 0>& dy_beta,
    const SpinWeighted<ComplexDataVector, 0>& one_minus_y) {

  *np_epsilon =
      square(one_minus_y) / ( sqrt(8.) * bondi_r)
      * ( dy_beta +
          ( bondi_j * conj(dy_j) - conj(bondi_j) * dy_j )
          * 0.125 / (1. + bondi_k)
      );
}

void newman_penrose_tau_impl(
    const gsl::not_null<SpinWeighted<ComplexDataVector, +1>*> np_tau,
    const SpinWeighted<ComplexDataVector, 2>& bondi_j,
    const SpinWeighted<ComplexDataVector, 0>& bondi_k,
    const SpinWeighted<ComplexDataVector, 0>& bondi_r,
    const SpinWeighted<ComplexDataVector, 1>& bondi_q,
    const SpinWeighted<ComplexDataVector, 1>& eth_beta,
    const SpinWeighted<ComplexDataVector, 0>& one_minus_y) {

  // Intentionally using expressions with auto. This code is only used for
  // observing, and should be profiled if used more often
  const auto one_plus_k = 1. + bondi_k;
  const auto sqrt_one_plus_k = sqrt(one_plus_k);
  const auto two_eth_beta_minus_q = 2. * eth_beta - bondi_q;

  *np_tau =
      0.125 * one_minus_y / bondi_r *
      ( sqrt_one_plus_k * two_eth_beta_minus_q
        - bondi_j * conj(two_eth_beta_minus_q) / sqrt_one_plus_k
      );
}

void newman_penrose_sigma_impl(
    const gsl::not_null<SpinWeighted<ComplexDataVector, +2>*> np_sigma,
    const SpinWeighted<ComplexDataVector, 2>& bondi_j,
    const SpinWeighted<ComplexDataVector, 2>& dy_j,
    const SpinWeighted<ComplexDataVector, 0>& bondi_k,
    const SpinWeighted<ComplexDataVector, 0>& bondi_r,
    const SpinWeighted<ComplexDataVector, 0>& one_minus_y) {

  // Intentionally using expressions with auto. This code is only used for
  // observing, and should be profiled if used more often
  const auto one_plus_k = 1. + bondi_k;

  *np_sigma =
      square(one_minus_y) / ( sqrt(128.) * bondi_k * bondi_r ) *
      ( square(bondi_j) * conj(dy_j) / one_plus_k
        - one_plus_k * dy_j
      );
}

void newman_penrose_rho_impl(
    const gsl::not_null<SpinWeighted<ComplexDataVector, 0>*> np_rho,
    const SpinWeighted<ComplexDataVector, 0>& bondi_r,
    const SpinWeighted<ComplexDataVector, 0>& one_minus_y) {

  *np_rho = - one_minus_y / ( sqrt(8.) * bondi_r );
}

void newman_penrose_pi_impl(
    const gsl::not_null<SpinWeighted<ComplexDataVector, -1>*> np_pi,
    const SpinWeighted<ComplexDataVector, 2>& bondi_j,
    const SpinWeighted<ComplexDataVector, 0>& bondi_k,
    const SpinWeighted<ComplexDataVector, 0>& bondi_r,
    const SpinWeighted<ComplexDataVector, 1>& bondi_q,
    const SpinWeighted<ComplexDataVector, 1>& eth_beta,
    const SpinWeighted<ComplexDataVector, 0>& one_minus_y) {

  // Intentionally using expressions with auto. This code is only used for
  // observing, and should be profiled if used more often
  const auto one_plus_k = 1. + bondi_k;
  const auto sqrt_one_plus_k = sqrt(one_plus_k);
  const auto q_plus_two_eth_beta = bondi_q + 2. * eth_beta;

  *np_pi =
      0.125 * one_minus_y / bondi_r *
      ( conj(bondi_j) * q_plus_two_eth_beta / sqrt_one_plus_k
        - sqrt_one_plus_k * conj(q_plus_two_eth_beta)
      );
}

void newman_penrose_nu_impl(
    const gsl::not_null<SpinWeighted<ComplexDataVector, -1>*> np_nu,
    const SpinWeighted<ComplexDataVector, 2>& bondi_j,
    const SpinWeighted<ComplexDataVector, 0>& bondi_k,
    const SpinWeighted<ComplexDataVector, 1>& eth_w,
    const SpinWeighted<ComplexDataVector, 0>& exp_2_beta) {

  // Intentionally using expressions with auto. This code is only used for
  // observing, and should be profiled if used more often
  const auto one_plus_k = 1. + bondi_k;
  const auto sqrt_one_plus_k = sqrt(one_plus_k);

  *np_nu = 0.5 / exp_2_beta *
           ( conj(bondi_j) * eth_w / sqrt_one_plus_k
             - sqrt_one_plus_k * conj(eth_w) );
}

void newman_penrose_mu_impl(
    const gsl::not_null<SpinWeighted<ComplexDataVector, 0>*> np_mu,
    const SpinWeighted<ComplexDataVector, 0>& bondi_r,
    const SpinWeighted<ComplexDataVector, 0>& bondi_w,
    const SpinWeighted<ComplexDataVector, 0>& ethbar_u,
    const SpinWeighted<ComplexDataVector, 0>& exp_2_beta,
    const SpinWeighted<ComplexDataVector, 0>& one_minus_y) {

  *np_mu = 1. / ( sqrt(8.) * exp_2_beta ) *
           ( conj(ethbar_u) + ethbar_u - one_minus_y / bondi_r - 2. * bondi_w );
}

void newman_penrose_lambda_impl(
    const gsl::not_null<SpinWeighted<ComplexDataVector, -2>*> np_lambda,
    const SpinWeighted<ComplexDataVector, 2>& bondi_j,
    const SpinWeighted<ComplexDataVector, 2>& dy_j,
    const SpinWeighted<ComplexDataVector, 3>& eth_j,
    const SpinWeighted<ComplexDataVector, 1>& ethbar_j,
    const SpinWeighted<ComplexDataVector, 0>& bondi_k,
    const SpinWeighted<ComplexDataVector, 2>& bondi_h,
    const SpinWeighted<ComplexDataVector, 0>& bondi_r,
    const SpinWeighted<ComplexDataVector, 1>& bondi_u,
    const SpinWeighted<ComplexDataVector, 2>& eth_u,
    const SpinWeighted<ComplexDataVector, 0>& ethbar_u,
    const SpinWeighted<ComplexDataVector, 0>& bondi_w,
    const SpinWeighted<ComplexDataVector, 0>& exp_2_beta,
    const SpinWeighted<ComplexDataVector, 0>& one_minus_y) {

  // Intentionally using expressions with auto. This code is only used for
  // observing, and should be profiled if used more often
  const auto one_plus_k = 1. + bondi_k;

  const auto inner1 =
      0.5 * one_minus_y / one_plus_k *
      ( ( square(conj(bondi_j)) * dy_j - conj(dy_j) ) / bondi_k
        - (2. + bondi_k) * conj(dy_j) );

  const auto inner2 = 2. * bondi_h + bondi_u * ethbar_j + conj(bondi_u) * eth_j;

  *np_lambda =
      1. / ( sqrt(32.) * exp_2_beta ) *
      ( ( one_minus_y / bondi_r + 2. * bondi_w ) * inner1
        + 2. * one_plus_k * conj(eth_u)
        + ( conj(inner2) + 2. * conj(bondi_j) * ( ethbar_u - conj(ethbar_u) ) )
        + conj(inner2) / bondi_k
        - square(conj(bondi_j)) * ( inner2 + 2. * bondi_k * eth_u ) /
          ( bondi_k * one_plus_k ) );
}

void weyl_psi0_impl(
    const gsl::not_null<SpinWeighted<ComplexDataVector, 2>*> psi_0,
    const SpinWeighted<ComplexDataVector, 2>& bondi_j,
    const SpinWeighted<ComplexDataVector, 2>& dy_j,
    const SpinWeighted<ComplexDataVector, 2>& dy_dy_j,
    const SpinWeighted<ComplexDataVector, 0>& bondi_k,
    const SpinWeighted<ComplexDataVector, 0>& bondi_r,
    const SpinWeighted<ComplexDataVector, 0>& one_minus_y) {
  *psi_0 = pow<4>(one_minus_y) * 0.0625 / (square(bondi_r) * bondi_k) *
           (0.125 * one_minus_y *
                (dy_j * conj(dy_j) -
                 0.25 * square(bondi_j * conj(dy_j) + conj(bondi_j) * dy_j) /
                     square(bondi_k)) *
                ((1.0 + bondi_k) * dy_j -
                 square(bondi_j) * conj(dy_j) / (1.0 + bondi_k)) -
            0.5 * (1.0 + bondi_k) * dy_dy_j +
            0.5 * square(bondi_j) * conj(dy_dy_j) / (1.0 + bondi_k) +
            (-0.25 * bondi_j *
                 (square(conj(bondi_j)) * square(dy_j) +
                  square(bondi_j) * square(conj(dy_j))) +
             0.5 * bondi_j * (1.0 + square(bondi_k)) * dy_j * conj(dy_j)) /
                square(bondi_k));
}

void weyl_psi1_impl(
    const gsl::not_null<SpinWeighted<ComplexDataVector, 1>*> psi_1,
    const SpinWeighted<ComplexDataVector, 2>& bondi_j,
    const SpinWeighted<ComplexDataVector, 2>& dy_j,
    const SpinWeighted<ComplexDataVector, 0>& bondi_k,
    const SpinWeighted<ComplexDataVector, 1>& bondi_q,
    const SpinWeighted<ComplexDataVector, 1>& dy_q,
    const SpinWeighted<ComplexDataVector, 0>& bondi_r,
    const SpinWeighted<ComplexDataVector, 1>& eth_r_divided_by_r,
    const SpinWeighted<ComplexDataVector, 0>& dy_beta,
    const SpinWeighted<ComplexDataVector, 1>& eth_beta,
    const SpinWeighted<ComplexDataVector, 1>& eth_dy_beta,
    const SpinWeighted<ComplexDataVector, 0>& one_minus_y) {

  const double prefac = 1./sqrt(128.); // compile time const, but
                                       // sqrt is not constexpr yet
  const auto one_plus_k = 1. + bondi_k;
  const auto eth_beta_plus_half_q = eth_beta + 0.5 * bondi_q;
  const auto conj_j_times_dy_j = conj(bondi_j) * dy_j;

  const auto inner_expr =
    bondi_j
    * (-2. * conj(dy_q)
       + conj(dy_j) * (2. * eth_beta_plus_half_q
                       + bondi_j * conj(eth_beta_plus_half_q)))
    + one_plus_k
    * (eth_beta_plus_half_q * (conj_j_times_dy_j
                               - conj(conj_j_times_dy_j))
       + 2. * (dy_q + bondi_j * conj(dy_q))
       - one_plus_k * (2. * dy_q + dy_j * conj(eth_beta_plus_half_q)));

  *psi_1 = prefac * square(one_minus_y) / (square(bondi_r) * sqrt(one_plus_k))
    * (bondi_j * conj(eth_beta_plus_half_q)
       - one_plus_k * eth_beta_plus_half_q
       + one_minus_y
       * (eth_dy_beta * one_plus_k
          - bondi_j * conj(eth_dy_beta)
          + dy_beta * (one_plus_k * eth_r_divided_by_r
                       - bondi_j * conj(eth_r_divided_by_r))
          + 0.25 * inner_expr / bondi_k));
}
}  // namespace


void newman_penrose_alpha(
    const gsl::not_null<Scalar<SpinWeighted<ComplexDataVector, -1>>*> np_alpha,
    const Scalar<SpinWeighted<ComplexDataVector, 2>>& bondi_j,
    const Scalar<SpinWeighted<ComplexDataVector, 3>>& eth_j,
    const Scalar<SpinWeighted<ComplexDataVector, 1>>& ethbar_j,
    const Scalar<SpinWeighted<ComplexDataVector, 0>>& bondi_k,
    const Scalar<SpinWeighted<ComplexDataVector, 0>>& bondi_r,
    const Scalar<SpinWeighted<ComplexDataVector, 1>>& bondi_q,
    const Scalar<SpinWeighted<ComplexDataVector, 1>>& eth_beta,
    const Scalar<SpinWeighted<ComplexDataVector, 0>>& one_minus_y) {
  newman_penrose_alpha_impl(
      make_not_null(&get(*np_alpha)), get(bondi_j), get(eth_j),
      get(ethbar_j), get(bondi_k), get(bondi_r), get(bondi_q),
      get(eth_beta), get(one_minus_y));
}

void newman_penrose_beta(
    const gsl::not_null<Scalar<SpinWeighted<ComplexDataVector, +1>>*> np_beta,
    const Scalar<SpinWeighted<ComplexDataVector, 2>>& bondi_j,
    const Scalar<SpinWeighted<ComplexDataVector, 3>>& eth_j,
    const Scalar<SpinWeighted<ComplexDataVector, 1>>& ethbar_j,
    const Scalar<SpinWeighted<ComplexDataVector, 0>>& bondi_k,
    const Scalar<SpinWeighted<ComplexDataVector, 0>>& bondi_r,
    const Scalar<SpinWeighted<ComplexDataVector, 1>>& bondi_q,
    const Scalar<SpinWeighted<ComplexDataVector, 1>>& eth_beta,
    const Scalar<SpinWeighted<ComplexDataVector, 0>>& one_minus_y) {
  newman_penrose_beta_impl(
      make_not_null(&get(*np_beta)), get(bondi_j), get(eth_j),
      get(ethbar_j), get(bondi_k), get(bondi_r), get(bondi_q),
      get(eth_beta), get(one_minus_y));
}

void newman_penrose_gamma(
    const gsl::not_null<Scalar<SpinWeighted<ComplexDataVector, 0>>*> np_gamma,
    const Scalar<SpinWeighted<ComplexDataVector, 2>>& bondi_j,
    const Scalar<SpinWeighted<ComplexDataVector, 2>>& dy_j,
    const Scalar<SpinWeighted<ComplexDataVector, 3>>& eth_j,
    const Scalar<SpinWeighted<ComplexDataVector, 1>>& ethbar_j,
    const Scalar<SpinWeighted<ComplexDataVector, 0>>& bondi_k,
    const Scalar<SpinWeighted<ComplexDataVector, 2>>& bondi_h,
    const Scalar<SpinWeighted<ComplexDataVector, 0>>& bondi_r,
    const Scalar<SpinWeighted<ComplexDataVector, 1>>& bondi_u,
    const Scalar<SpinWeighted<ComplexDataVector, 2>>& eth_u,
    const Scalar<SpinWeighted<ComplexDataVector, 0>>& ethbar_u,
    const Scalar<SpinWeighted<ComplexDataVector, 0>>& bondi_w,
    const Scalar<SpinWeighted<ComplexDataVector, 0>>& dy_w,
    const Scalar<SpinWeighted<ComplexDataVector, 0>>& exp_2_beta,
    const Scalar<SpinWeighted<ComplexDataVector, 0>>& one_minus_y) {
  newman_penrose_gamma_impl(
      make_not_null(&get(*np_gamma)), get(bondi_j), get(dy_j),
      get(eth_j), get(ethbar_j), get(bondi_k), get(bondi_h),
      get(bondi_r), get(bondi_u), get(eth_u), get(ethbar_u),
      get(bondi_w), get(dy_w), get(exp_2_beta), get(one_minus_y));
}

void newman_penrose_epsilon(
    const gsl::not_null<Scalar<SpinWeighted<ComplexDataVector, 0>>*> np_epsilon,
    const Scalar<SpinWeighted<ComplexDataVector, 2>>& bondi_j,
    const Scalar<SpinWeighted<ComplexDataVector, 2>>& dy_j,
    const Scalar<SpinWeighted<ComplexDataVector, 0>>& bondi_k,
    const Scalar<SpinWeighted<ComplexDataVector, 0>>& bondi_r,
    const Scalar<SpinWeighted<ComplexDataVector, 0>>& dy_beta,
    const Scalar<SpinWeighted<ComplexDataVector, 0>>& one_minus_y) {
  newman_penrose_epsilon_impl(
      make_not_null(&get(*np_epsilon)), get(bondi_j), get(dy_j),
      get(bondi_k), get(bondi_r), get(dy_beta), get(one_minus_y));
}

// There is no newmpan_penrose_kappa because in our conventions, it's 0.

void newman_penrose_tau(
    const gsl::not_null<Scalar<SpinWeighted<ComplexDataVector, +1>>*> np_tau,
    const Scalar<SpinWeighted<ComplexDataVector, 2>>& bondi_j,
    const Scalar<SpinWeighted<ComplexDataVector, 0>>& bondi_k,
    const Scalar<SpinWeighted<ComplexDataVector, 0>>& bondi_r,
    const Scalar<SpinWeighted<ComplexDataVector, 1>>& bondi_q,
    const Scalar<SpinWeighted<ComplexDataVector, 1>>& eth_beta,
    const Scalar<SpinWeighted<ComplexDataVector, 0>>& one_minus_y) {
  newman_penrose_tau_impl(
      make_not_null(&get(*np_tau)), get(bondi_j), get(bondi_k),
      get(bondi_r), get(bondi_q), get(eth_beta), get(one_minus_y));
}

void newman_penrose_sigma(
    const gsl::not_null<Scalar<SpinWeighted<ComplexDataVector, +2>>*> np_sigma,
    const Scalar<SpinWeighted<ComplexDataVector, 2>>& bondi_j,
    const Scalar<SpinWeighted<ComplexDataVector, 2>>& dy_j,
    const Scalar<SpinWeighted<ComplexDataVector, 0>>& bondi_k,
    const Scalar<SpinWeighted<ComplexDataVector, 0>>& bondi_r,
    const Scalar<SpinWeighted<ComplexDataVector, 0>>& one_minus_y) {
  newman_penrose_sigma_impl(
      make_not_null(&get(*np_sigma)), get(bondi_j), get(dy_j),
      get(bondi_k), get(bondi_r), get(one_minus_y));
}

void newman_penrose_rho(
    const gsl::not_null<Scalar<SpinWeighted<ComplexDataVector, 0>>*> np_rho,
    const Scalar<SpinWeighted<ComplexDataVector, 0>>& bondi_r,
    const Scalar<SpinWeighted<ComplexDataVector, 0>>& one_minus_y) {
  newman_penrose_rho_impl(
      make_not_null(&get(*np_rho)), get(bondi_r), get(one_minus_y));
}

void newman_penrose_pi(
    const gsl::not_null<Scalar<SpinWeighted<ComplexDataVector, -1>>*> np_pi,
    const Scalar<SpinWeighted<ComplexDataVector, 2>>& bondi_j,
    const Scalar<SpinWeighted<ComplexDataVector, 0>>& bondi_k,
    const Scalar<SpinWeighted<ComplexDataVector, 0>>& bondi_r,
    const Scalar<SpinWeighted<ComplexDataVector, 1>>& bondi_q,
    const Scalar<SpinWeighted<ComplexDataVector, 1>>& eth_beta,
    const Scalar<SpinWeighted<ComplexDataVector, 0>>& one_minus_y) {
  newman_penrose_pi_impl(
      make_not_null(&get(*np_pi)), get(bondi_j), get(bondi_k),
      get(bondi_r), get(bondi_q), get(eth_beta), get(one_minus_y));
}

void newman_penrose_nu(
    const gsl::not_null<Scalar<SpinWeighted<ComplexDataVector, -1>>*> np_nu,
    const Scalar<SpinWeighted<ComplexDataVector, 2>>& bondi_j,
    const Scalar<SpinWeighted<ComplexDataVector, 0>>& bondi_k,
    const Scalar<SpinWeighted<ComplexDataVector, 1>>& eth_w,
    const Scalar<SpinWeighted<ComplexDataVector, 0>>& exp_2_beta) {
  newman_penrose_nu_impl(
      make_not_null(&get(*np_nu)), get(bondi_j), get(bondi_k),
      get(eth_w), get(exp_2_beta));
}

void newman_penrose_mu(
    const gsl::not_null<Scalar<SpinWeighted<ComplexDataVector, 0>>*> np_mu,
    const Scalar<SpinWeighted<ComplexDataVector, 0>>& bondi_r,
    const Scalar<SpinWeighted<ComplexDataVector, 0>>& bondi_w,
    const Scalar<SpinWeighted<ComplexDataVector, 0>>& ethbar_u,
    const Scalar<SpinWeighted<ComplexDataVector, 0>>& exp_2_beta,
    const Scalar<SpinWeighted<ComplexDataVector, 0>>& one_minus_y) {
  newman_penrose_mu_impl(
      make_not_null(&get(*np_mu)), get(bondi_r), get(bondi_w),
      get(ethbar_u), get(exp_2_beta), get(one_minus_y));
}


void newman_penrose_lambda(
    const gsl::not_null<Scalar<SpinWeighted<ComplexDataVector, -2>>*> np_lambda,
    const Scalar<SpinWeighted<ComplexDataVector, 2>>& bondi_j,
    const Scalar<SpinWeighted<ComplexDataVector, 2>>& dy_j,
    const Scalar<SpinWeighted<ComplexDataVector, 3>>& eth_j,
    const Scalar<SpinWeighted<ComplexDataVector, 1>>& ethbar_j,
    const Scalar<SpinWeighted<ComplexDataVector, 0>>& bondi_k,
    const Scalar<SpinWeighted<ComplexDataVector, 2>>& bondi_h,
    const Scalar<SpinWeighted<ComplexDataVector, 0>>& bondi_r,
    const Scalar<SpinWeighted<ComplexDataVector, 1>>& bondi_u,
    const Scalar<SpinWeighted<ComplexDataVector, 2>>& eth_u,
    const Scalar<SpinWeighted<ComplexDataVector, 0>>& ethbar_u,
    const Scalar<SpinWeighted<ComplexDataVector, 0>>& bondi_w,
    const Scalar<SpinWeighted<ComplexDataVector, 0>>& exp_2_beta,
    const Scalar<SpinWeighted<ComplexDataVector, 0>>& one_minus_y) {
  newman_penrose_lambda_impl(
      make_not_null(&get(*np_lambda)), get(bondi_j), get(dy_j),
      get(eth_j), get(ethbar_j), get(bondi_k), get(bondi_h),
      get(bondi_r), get(bondi_u), get(eth_u), get(ethbar_u),
      get(bondi_w), get(exp_2_beta), get(one_minus_y));
}

void VolumeWeyl<Tags::Psi0>::apply(
    const gsl::not_null<Scalar<SpinWeighted<ComplexDataVector, 2>>*> psi_0,
    const Scalar<SpinWeighted<ComplexDataVector, 2>>& bondi_j,
    const Scalar<SpinWeighted<ComplexDataVector, 2>>& dy_j,
    const Scalar<SpinWeighted<ComplexDataVector, 2>>& dy_dy_j,
    const Scalar<SpinWeighted<ComplexDataVector, 0>>& bondi_k,
    const Scalar<SpinWeighted<ComplexDataVector, 0>>& bondi_r,
    const Scalar<SpinWeighted<ComplexDataVector, 0>>& one_minus_y) {
  weyl_psi0_impl(make_not_null(&get(*psi_0)), get(bondi_j), get(dy_j),
                 get(dy_dy_j), get(bondi_k), get(bondi_r), get(one_minus_y));
}

void VolumeWeyl<Tags::Psi1>::apply(
    const gsl::not_null<Scalar<SpinWeighted<ComplexDataVector, 1>>*> psi_1,
    const Scalar<SpinWeighted<ComplexDataVector, 2>>& bondi_j,
    const Scalar<SpinWeighted<ComplexDataVector, 2>>& dy_j,
    const Scalar<SpinWeighted<ComplexDataVector, 0>>& bondi_k,
    const Scalar<SpinWeighted<ComplexDataVector, 1>>& bondi_q,
    const Scalar<SpinWeighted<ComplexDataVector, 1>>& dy_q,
    const Scalar<SpinWeighted<ComplexDataVector, 0>>& bondi_r,
    const Scalar<SpinWeighted<ComplexDataVector, 1>>& eth_r_divided_by_r,
    const Scalar<SpinWeighted<ComplexDataVector, 0>>& dy_beta,
    const Scalar<SpinWeighted<ComplexDataVector, 1>>& eth_beta,
    const Scalar<SpinWeighted<ComplexDataVector, 1>>& eth_dy_beta,
    const Scalar<SpinWeighted<ComplexDataVector, 0>>& one_minus_y) {
  weyl_psi1_impl(make_not_null(&get(*psi_1)), get(bondi_j), get(dy_j),
                 get(bondi_k), get(bondi_q), get(dy_q),
                 get(bondi_r), get(eth_r_divided_by_r), get(dy_beta),
                 get(eth_beta), get(eth_dy_beta), get(one_minus_y));
}

void TransformBondiJToCauchyCoords::apply(
    gsl::not_null<Scalar<SpinWeighted<ComplexDataVector, 2>>*>
        cauchy_view_volume_j,
    const Scalar<SpinWeighted<ComplexDataVector, 2>>& gauge_cauchy_c,
    const Scalar<SpinWeighted<ComplexDataVector, 2>>& volume_j,
    const Scalar<SpinWeighted<ComplexDataVector, 0>>& gauge_cauchy_d,
    const Scalar<SpinWeighted<ComplexDataVector, 0>>& omega_cauchy,
    const Spectral::Swsh::SwshInterpolator& interpolator,
    const size_t l_max) {
  const size_t number_of_angular_points =
      Spectral::Swsh::number_of_swsh_collocation_points(l_max);
  const size_t number_of_radial_points =
      get(volume_j).size() / number_of_angular_points;

  SpinWeighted<ComplexDataVector, 2> target_angular_view;
  const SpinWeighted<ComplexDataVector, 2> source_angular_view;
  // Iterate for each spherical shell
  for (size_t i = 0; i < number_of_radial_points; ++i) {
    target_angular_view.set_data_ref(
        get(*cauchy_view_volume_j).data().data() +
            i * Spectral::Swsh::number_of_swsh_collocation_points(l_max),
        Spectral::Swsh::number_of_swsh_collocation_points(l_max));

    make_const_view(make_not_null(&source_angular_view), get(volume_j),
                    i * number_of_angular_points, number_of_angular_points);
    interpolator.interpolate(make_not_null(&target_angular_view),
                             source_angular_view);
    target_angular_view.data() =
        target_angular_view.data() * conj(square(get(gauge_cauchy_d).data())) +
        conj(target_angular_view.data()) * square(get(gauge_cauchy_c).data()) +
        2.0 * get(gauge_cauchy_c).data() * conj(get(gauge_cauchy_d).data()) *
            sqrt(1.0 +
                 target_angular_view.data() * conj(target_angular_view.data()));
    target_angular_view.data() *= 0.25 / square(get(omega_cauchy).data());
  }
}

void VolumeWeyl<Tags::Psi0Match>::apply(
    gsl::not_null<Scalar<SpinWeighted<ComplexDataVector, 2>>*> psi_0,
    const Scalar<SpinWeighted<ComplexDataVector, 2>>& bondi_j_cauchy,
    const Scalar<SpinWeighted<ComplexDataVector, 2>>& dy_j_cauchy,
    const Scalar<SpinWeighted<ComplexDataVector, 2>>& dy_dy_j_cauchy,
    const Scalar<SpinWeighted<ComplexDataVector, 0>>& bondi_r_cauchy,
    const Scalar<SpinWeighted<ComplexDataVector, 0>>& one_minus_y,
    const size_t l_max) {
  const size_t number_of_angular_points =
      Spectral::Swsh::number_of_swsh_collocation_points(l_max);
  const size_t number_of_radial_points =
      get(bondi_j_cauchy).size() / number_of_angular_points;

  // Get bondi_k in the Cauchy coordinates
  SpinWeighted<ComplexDataVector, 0> bondi_k_cauchy;
  bondi_k_cauchy.data() =
      sqrt(1.0 + get(bondi_j_cauchy).data() * conj(get(bondi_j_cauchy).data()));

  const SpinWeighted<ComplexDataVector, 2> bondi_j_cauchy_view;
  const SpinWeighted<ComplexDataVector, 2> dy_j_cauchy_view;
  const SpinWeighted<ComplexDataVector, 2> dy_dy_j_cauchy_view;
  const SpinWeighted<ComplexDataVector, 0> bondi_k_cauchy_view;
  const SpinWeighted<ComplexDataVector, 0> one_minus_y_view;

  SpinWeighted<ComplexDataVector, 2> psi0_view;

  // Iterate for each spherical shell
  for (size_t i = 0; i < number_of_radial_points; ++i) {
    // Note that bondi_r_cauchy, bondi_j_cauchy, dy_j_cauchy, dy_dy_j_cauchy,
    // one_minus_y and bondi_k_cauchy are available only as surface quantities
    make_const_view(make_not_null(&bondi_j_cauchy_view), get(bondi_j_cauchy),
                    i * number_of_angular_points, number_of_angular_points);
    make_const_view(make_not_null(&dy_j_cauchy_view), get(dy_j_cauchy),
                    i * number_of_angular_points, number_of_angular_points);
    make_const_view(make_not_null(&dy_dy_j_cauchy_view), get(dy_dy_j_cauchy),
                    i * number_of_angular_points, number_of_angular_points);
    make_const_view(make_not_null(&bondi_k_cauchy_view), bondi_k_cauchy,
                    i * number_of_angular_points, number_of_angular_points);
    make_const_view(make_not_null(&one_minus_y_view), get(one_minus_y),
                    i * number_of_angular_points, number_of_angular_points);

    psi0_view.set_data_ref(
        get(*psi_0).data().data() +
            i * Spectral::Swsh::number_of_swsh_collocation_points(l_max),
        Spectral::Swsh::number_of_swsh_collocation_points(l_max));

    weyl_psi0_impl(make_not_null(&psi0_view), bondi_j_cauchy_view,
                   dy_j_cauchy_view, dy_dy_j_cauchy_view, bondi_k_cauchy_view,
                   get(bondi_r_cauchy), one_minus_y_view);
  }
}

void InnerBoundaryWeyl::apply(
    gsl::not_null<Scalar<SpinWeighted<ComplexDataVector, 2>>*> psi_0_boundary,
    gsl::not_null<Scalar<SpinWeighted<ComplexDataVector, 2>>*>
        dlambda_psi_0_boundary,
    const Scalar<SpinWeighted<ComplexDataVector, 2>>& psi_0,
    const Scalar<SpinWeighted<ComplexDataVector, 2>>& dy_psi_0,
    const Scalar<SpinWeighted<ComplexDataVector, 0>>& one_minus_y,
    const Scalar<SpinWeighted<ComplexDataVector, 0>>& bondi_r_cauchy,
    const Scalar<SpinWeighted<ComplexDataVector, 0>>& bondi_beta_cauchy,
    const size_t l_max) {
  const size_t number_of_angular_points =
      Spectral::Swsh::number_of_swsh_collocation_points(l_max);

  const SpinWeighted<ComplexDataVector, 0> one_minus_y_boundary;
  const SpinWeighted<ComplexDataVector, 0> bondi_beta_cauchy_boundary;
  const SpinWeighted<ComplexDataVector, 2> psi_0_boundary_view;
  const SpinWeighted<ComplexDataVector, 2> dy_psi_0_boundary_view;

  // Take the boundary data
  make_const_view(make_not_null(&psi_0_boundary_view), get(psi_0), 0,
                  number_of_angular_points);
  make_const_view(make_not_null(&dy_psi_0_boundary_view), get(dy_psi_0), 0,
                  number_of_angular_points);
  make_const_view(make_not_null(&one_minus_y_boundary), get(one_minus_y), 0,
                  number_of_angular_points);
  make_const_view(make_not_null(&bondi_beta_cauchy_boundary),
                  get(bondi_beta_cauchy), 0, number_of_angular_points);

  get(*psi_0_boundary) = psi_0_boundary_view;
  get(*dlambda_psi_0_boundary) = dy_psi_0_boundary_view.data() *
                              square(one_minus_y_boundary.data()) /
                              (2.0 * get(bondi_r_cauchy).data()) *
                              exp(-2.0 * bondi_beta_cauchy_boundary.data());
}
}  // namespace Cce
