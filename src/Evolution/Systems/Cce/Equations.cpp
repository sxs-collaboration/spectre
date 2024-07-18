// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Evolution/Systems/Cce/Equations.hpp"

#include <complex>

#include "DataStructures/ComplexDataVector.hpp"
#include "DataStructures/DataVector.hpp"
#include "DataStructures/SpinWeighted.hpp"
#include "DataStructures/Tags/TempTensor.hpp"
#include "Utilities/ConstantExpressions.hpp"

namespace Cce {

namespace detail {
void klein_gordon_rhs_npsi1(
    const gsl::not_null<SpinWeighted<ComplexDataVector, 0>*> result,
    const SpinWeighted<ComplexDataVector, 1>& eth_beta,
    const SpinWeighted<ComplexDataVector, 1>& eth_kg_psi,
    const SpinWeighted<ComplexDataVector, 0>& eth_ethbar_kg_psi,
    const SpinWeighted<ComplexDataVector, 0>& bondi_k) {
  (*result).data() = (2. * real(eth_beta.data() * conj(eth_kg_psi).data()) +
                      eth_ethbar_kg_psi.data()) *
                     bondi_k.data();
}

void klein_gordon_rhs_npsi2(
    const gsl::not_null<SpinWeighted<ComplexDataVector, 0>*> result,
    const SpinWeighted<ComplexDataVector, 2>& j,
    const SpinWeighted<ComplexDataVector, 2>& eth_eth_kg_psi,
    const SpinWeighted<ComplexDataVector, 1>& eth_beta,
    const SpinWeighted<ComplexDataVector, 1>& eth_kg_psi,
    const SpinWeighted<ComplexDataVector, 1>& ethbar_j) {
  (*result).data() = real(conj(j).data() * eth_eth_kg_psi.data());
  (*result).data() +=
      2 * real(conj(eth_beta).data() * conj(eth_kg_psi).data() * j.data());
  (*result).data() += real(ethbar_j.data() * conj(eth_kg_psi).data());
}

void klein_gordon_rhs_npsi3(
    const gsl::not_null<SpinWeighted<ComplexDataVector, 0>*> result,
    const SpinWeighted<ComplexDataVector, 1>& eth_k,
    const SpinWeighted<ComplexDataVector, 1>& eth_kg_psi) {
  (*result).data() = real(eth_k.data() * conj(eth_kg_psi).data());
}

void klein_gordon_rhs_npsi4_divided_by_one_minues_y_squared(
    const gsl::not_null<SpinWeighted<ComplexDataVector, 0>*> result,
    const SpinWeighted<ComplexDataVector, 1>& eth_kg_psi,
    const SpinWeighted<ComplexDataVector, 1>& dy_bondi_u,
    const SpinWeighted<ComplexDataVector, 0>& ethbar_u,
    const SpinWeighted<ComplexDataVector, 1>& bondi_u,
    const SpinWeighted<ComplexDataVector, 1>& eth_dy_kg_psi,
    const SpinWeighted<ComplexDataVector, 0>& dy_kg_psi,
    const SpinWeighted<ComplexDataVector, 0>& bondi_r,
    const SpinWeighted<ComplexDataVector, 1>& eth_r_divided_by_r) {
  auto dr_u = 0.5 / bondi_r * dy_bondi_u;
  auto dr_psi = 0.5 / bondi_r * dy_kg_psi;
  auto eth_dr_psi =
      0.5 / bondi_r * (eth_dy_kg_psi + eth_r_divided_by_r * dy_kg_psi);

  auto res = conj(eth_kg_psi) * dr_u + conj(ethbar_u) * dr_psi +
             2.0 * conj(bondi_u) * eth_dr_psi;
  (*result).data() = 2.0 * real(res.data());
}

void klein_gordon_rhs_tau(
    const gsl::not_null<SpinWeighted<ComplexDataVector, 0>*> result,
    const SpinWeighted<ComplexDataVector, 0>& one_minus_y,
    const SpinWeighted<ComplexDataVector, 0>& dy_w,
    const SpinWeighted<ComplexDataVector, 0>& dy_kg_psi,
    const SpinWeighted<ComplexDataVector, 0>& dy_dy_kg_psi,
    const SpinWeighted<ComplexDataVector, 0>& bondi_w,
    const SpinWeighted<ComplexDataVector, 0>& bondi_r) {
  *result = one_minus_y * dy_w * dy_kg_psi +
            0.5 * square(one_minus_y) / bondi_r * dy_dy_kg_psi +
            one_minus_y * bondi_w * dy_dy_kg_psi + bondi_w * dy_kg_psi;
  (*result).data() *= 0.5;
}
}  // namespace detail

// suppresses doxygen problems with these functions

void ComputeBondiIntegrand<Tags::Integrand<Tags::BondiBeta>>::apply_impl(
    const gsl::not_null<SpinWeighted<ComplexDataVector, 0>*> integrand_for_beta,
    const SpinWeighted<ComplexDataVector, 2>& dy_j,
    const SpinWeighted<ComplexDataVector, 2>& j,
    const SpinWeighted<ComplexDataVector, 0>& one_minus_y) {
  *integrand_for_beta =
      0.125 * one_minus_y *
      (dy_j * conj(dy_j) -
       0.25 * square(j * conj(dy_j) + conj(j) * dy_j) / (1.0 + j * conj(j)));
}

void ComputeBondiIntegrand<Tags::PoleOfIntegrand<Tags::BondiQ>>::apply_impl(
    const gsl::not_null<SpinWeighted<ComplexDataVector, 1>*>
        pole_of_integrand_for_q,
    const SpinWeighted<ComplexDataVector, 1>& eth_beta) {
  *pole_of_integrand_for_q = -4.0 * eth_beta;
}

void ComputeBondiIntegrand<Tags::RegularIntegrand<Tags::BondiQ>>::apply_impl(
    const gsl::not_null<SpinWeighted<ComplexDataVector, 1>*>
        regular_integrand_for_q,
    const gsl::not_null<SpinWeighted<ComplexDataVector, 1>*> script_aq,
    const SpinWeighted<ComplexDataVector, 0>& dy_beta,
    const SpinWeighted<ComplexDataVector, 2>& dy_j,
    const SpinWeighted<ComplexDataVector, 2>& j,
    const SpinWeighted<ComplexDataVector, 1>& eth_dy_beta,
    const SpinWeighted<ComplexDataVector, 1>& eth_j_jbar,
    const SpinWeighted<ComplexDataVector, 1>& eth_jbar_dy_j,
    const SpinWeighted<ComplexDataVector, 1>& ethbar_dy_j,
    const SpinWeighted<ComplexDataVector, 1>& ethbar_j,
    const SpinWeighted<ComplexDataVector, 1>& eth_r_divided_by_r,
    const SpinWeighted<ComplexDataVector, 0>& k) {
  *script_aq =
      0.25 * (j * conj(ethbar_dy_j) - eth_jbar_dy_j - conj(ethbar_j) * dy_j +
              0.5 * eth_j_jbar * (conj(j) * dy_j + j * conj(dy_j)) /
                  (1.0 + j * conj(j)) -
              (conj(j) * dy_j - j * conj(dy_j)) * eth_r_divided_by_r);

  *regular_integrand_for_q =
      -2.0 * (*script_aq + j * conj(*script_aq) / k - eth_dy_beta +
              0.5 * ethbar_dy_j / k - dy_beta * eth_r_divided_by_r +
              0.5 * dy_j * conj(eth_r_divided_by_r) / k);
}

void ComputeBondiIntegrand<Tags::Integrand<Tags::BondiU>>::apply_impl(
    const gsl::not_null<SpinWeighted<ComplexDataVector, 1>*>
        regular_integrand_for_u,
    const SpinWeighted<ComplexDataVector, 0>& exp_2_beta,
    const SpinWeighted<ComplexDataVector, 2>& j,
    const SpinWeighted<ComplexDataVector, 1>& q,
    const SpinWeighted<ComplexDataVector, 0>& k,
    const SpinWeighted<ComplexDataVector, 0>& r) {
  *regular_integrand_for_u = 0.5 * exp_2_beta / r * (k * q - j * conj(q));
}

void ComputeBondiIntegrand<Tags::PoleOfIntegrand<Tags::BondiW>>::apply_impl(
    const gsl::not_null<SpinWeighted<ComplexDataVector, 0>*>
        pole_of_integrand_for_w,
    const SpinWeighted<ComplexDataVector, 0>& ethbar_u) {
  *pole_of_integrand_for_w = ethbar_u + conj(ethbar_u);
}

void ComputeBondiIntegrand<Tags::RegularIntegrand<Tags::BondiW>>::apply_impl(
    const gsl::not_null<SpinWeighted<ComplexDataVector, 0>*>
        regular_integrand_for_w,
    const gsl::not_null<SpinWeighted<ComplexDataVector, 0>*> script_av,
    const SpinWeighted<ComplexDataVector, 1>& dy_u,
    const SpinWeighted<ComplexDataVector, 0>& exp_2_beta,
    const SpinWeighted<ComplexDataVector, 2>& j,
    const SpinWeighted<ComplexDataVector, 1>& q,
    const SpinWeighted<ComplexDataVector, 1>& eth_beta,
    const SpinWeighted<ComplexDataVector, 2>& eth_eth_beta,
    const SpinWeighted<ComplexDataVector, 0>& eth_ethbar_beta,
    const SpinWeighted<ComplexDataVector, 2>& eth_ethbar_j,
    const SpinWeighted<ComplexDataVector, 0>& eth_ethbar_j_jbar,
    const SpinWeighted<ComplexDataVector, 1>& eth_j_jbar,
    const SpinWeighted<ComplexDataVector, 0>& ethbar_dy_u,
    const SpinWeighted<ComplexDataVector, 0>& ethbar_ethbar_j,
    const SpinWeighted<ComplexDataVector, 1>& ethbar_j,
    const SpinWeighted<ComplexDataVector, 1>& eth_r_divided_by_r,
    const SpinWeighted<ComplexDataVector, 0>& k,
    const SpinWeighted<ComplexDataVector, 0>& r) {
  // this computation is split over two lines because GCC-10 on release mode
  // optimizes the long expression templates in such a way to cause segfaults.
  *script_av =
      eth_beta * conj(ethbar_j) + 0.5 * ethbar_ethbar_j +
      j * square(conj(eth_beta)) + j * conj(eth_eth_beta) +
      0.125 * eth_j_jbar * conj(eth_j_jbar) / (k * (1.0 + j * conj(j))) +
      0.5 *
          (1.0 - 0.25 * eth_ethbar_j_jbar - eth_j_jbar * conj(eth_beta) -
           0.5 * conj(ethbar_j) * ethbar_j - 0.5 * conj(j) * eth_ethbar_j) /
      k;
  *script_av += k * (0.5 - eth_ethbar_beta - eth_beta * conj(eth_beta) -
                     0.25 * q * conj(q)) +
                0.25 * j * square(conj(q));

  *regular_integrand_for_w =
      0.5 * (0.5 * (ethbar_dy_u + conj(ethbar_dy_u) +
                    conj(dy_u) * eth_r_divided_by_r +
                    dy_u * conj(eth_r_divided_by_r)) -
             1.0 / r + 0.5 * exp_2_beta * (*script_av + conj(*script_av)) / r);
}

void ComputeBondiIntegrand<Tags::PoleOfIntegrand<Tags::BondiH>>::apply_impl(
    const gsl::not_null<SpinWeighted<ComplexDataVector, 2>*>
        pole_of_integrand_for_h,
    const SpinWeighted<ComplexDataVector, 2>& j,
    const SpinWeighted<ComplexDataVector, 1>& u,
    const SpinWeighted<ComplexDataVector, 0>& w,
    const SpinWeighted<ComplexDataVector, 2>& eth_u,
    const SpinWeighted<ComplexDataVector, 1>& ethbar_j,
    const SpinWeighted<ComplexDataVector, -2>& ethbar_jbar_u,
    const SpinWeighted<ComplexDataVector, 0>& ethbar_u,
    const SpinWeighted<ComplexDataVector, 0>& k) {
  *pole_of_integrand_for_h = -0.5 * conj(ethbar_jbar_u) - j * conj(ethbar_u) -
                             0.5 * j * ethbar_u - k * eth_u -
                             0.5 * u * ethbar_j + 2.0 * j * w;
}

void ComputeBondiIntegrand<Tags::RegularIntegrand<Tags::BondiH>>::apply_impl(
    const gsl::not_null<SpinWeighted<ComplexDataVector, 2>*>
        regular_integrand_for_h,
    const gsl::not_null<SpinWeighted<ComplexDataVector, 0>*> script_aj,
    const gsl::not_null<SpinWeighted<ComplexDataVector, 0>*> script_bj,
    const gsl::not_null<SpinWeighted<ComplexDataVector, 2>*> script_cj,
    const SpinWeighted<ComplexDataVector, 2>& dy_dy_j,
    const SpinWeighted<ComplexDataVector, 2>& dy_j,
    const SpinWeighted<ComplexDataVector, 0>& dy_w,
    const SpinWeighted<ComplexDataVector, 0>& exp_2_beta,
    const SpinWeighted<ComplexDataVector, 2>& j,
    const SpinWeighted<ComplexDataVector, 1>& q,
    const SpinWeighted<ComplexDataVector, 1>& u,
    const SpinWeighted<ComplexDataVector, 0>& w,
    const SpinWeighted<ComplexDataVector, 1>& eth_beta,
    const SpinWeighted<ComplexDataVector, 2>& eth_eth_beta,
    const SpinWeighted<ComplexDataVector, 0>& eth_ethbar_beta,
    const SpinWeighted<ComplexDataVector, 2>& eth_ethbar_j,
    const SpinWeighted<ComplexDataVector, 0>& eth_ethbar_j_jbar,
    const SpinWeighted<ComplexDataVector, 1>& eth_j_jbar,
    const SpinWeighted<ComplexDataVector, 2>& eth_q,
    const SpinWeighted<ComplexDataVector, 2>& eth_u,
    const SpinWeighted<ComplexDataVector, 2>& eth_ubar_dy_j,
    const SpinWeighted<ComplexDataVector, 1>& ethbar_dy_j,
    const SpinWeighted<ComplexDataVector, 0>& ethbar_ethbar_j,
    const SpinWeighted<ComplexDataVector, 1>& ethbar_j,
    const SpinWeighted<ComplexDataVector, -1>& ethbar_jbar_dy_j,
    const SpinWeighted<ComplexDataVector, -2>& ethbar_jbar_q_minus_2_eth_beta,
    const SpinWeighted<ComplexDataVector, 0>& ethbar_q,
    const SpinWeighted<ComplexDataVector, 0>& ethbar_u,
    const SpinWeighted<ComplexDataVector, 0>& du_r_divided_by_r,
    const SpinWeighted<ComplexDataVector, 1>& eth_r_divided_by_r,
    const SpinWeighted<ComplexDataVector, 0>& k,
    const SpinWeighted<ComplexDataVector, 0>& one_minus_y,
    const SpinWeighted<ComplexDataVector, 0>& r) {
  *script_aj =
      0.25 *
      (conj(ethbar_ethbar_j) -
       0.25 * (4.0 + eth_ethbar_j_jbar - j * conj(eth_ethbar_j)) /
           (k * (1.0 + j * conj(j))) +
       (3.0 - eth_ethbar_beta -
        conj(j) * eth_ethbar_j * (1.0 - 0.25 / (1.0 + j * conj(j)))) /
           k +
       conj(ethbar_j) * (2.0 * eth_beta +
                         0.5 *
                             (j * conj(eth_j_jbar) -
                              ethbar_j * (2.0 * (1.0 + j * conj(j)) - 1.0)) /
                             (k * (1.0 + j * conj(j))) -
                         q));

  // this computation is split over multiple lines because GCC-10 on release
  // mode optimizes the long expression templates in such a way to cause
  // segfaults.
  *script_bj =
      0.25 * (2.0 * dy_w -
              conj(j) * eth_u * (conj(j) * dy_j + j * conj(dy_j)) / k +
              1.0 / r + u * ethbar_j * conj(dy_j) -
              0.5 * u * conj(eth_j_jbar) * (conj(j) * dy_j + j * conj(dy_j)) /
                  (1.0 + j * conj(j)) +
              conj(u) * (conj(ethbar_jbar_dy_j) - j * conj(ethbar_dy_j)));
  *script_bj +=
      square(one_minus_y) * 0.125 *
      (0.25 * square(conj(j) * dy_j + j * conj(dy_j)) / (1.0 + j * conj(j)) -
       dy_j * conj(dy_j)) /
      r;
  *script_bj +=
      one_minus_y * 0.25 *
      (du_r_divided_by_r * dy_j *
           (conj(j) * (conj(j) * dy_j + j * conj(dy_j)) / (1.0 + j * conj(j)) -
            2.0 * conj(dy_j)) -
       w * (dy_j * conj(dy_j) - 0.25 *
                                    square((conj(j) * dy_j + j * conj(dy_j))) /
                                    (1.0 + j * conj(j))));

  *script_cj = 0.5 * ethbar_j * k * (eth_beta - 0.5 * q);

  *regular_integrand_for_h =
      j * (*script_bj + conj(*script_bj)) -
      0.5 * (eth_ubar_dy_j + u * ethbar_dy_j +
             u * dy_j * conj(eth_r_divided_by_r)) +
      0.5 * exp_2_beta / r *
          (*script_cj + square(j) / (1.0 + j * conj(j)) * conj(*script_cj) -
           j * (*script_aj + conj(*script_aj)) + eth_eth_beta - 0.5 * eth_q +
           0.25 * (conj(ethbar_jbar_q_minus_2_eth_beta) - j * conj(ethbar_q)) /
               k +
           square(eth_beta - 0.5 * q)) -
      dy_j * 0.5 *
          (conj(j) * eth_u / k - j * k * conj(eth_u) +
           0.5 * (1.0 + j * conj(j)) * (conj(ethbar_u) - ethbar_u) +
           conj(u) * eth_r_divided_by_r - w) +
      conj(dy_j) * (0.5 * j * eth_u * (j * conj(j) / k) -
                    0.25 * square(j) * (ethbar_u - conj(ethbar_u))) +
      one_minus_y *
          (0.5 * (dy_dy_j * (w + 2.0 * du_r_divided_by_r) - dy_j / r) +
           0.5 * dy_j * (dy_w + 1.0 / r)) +
      square(one_minus_y) * 0.25 * dy_dy_j / r;
}

void ComputeBondiIntegrand<Tags::LinearFactor<Tags::BondiH>>::apply_impl(
    const gsl::not_null<SpinWeighted<ComplexDataVector, 0>*>
        linear_factor_for_h,
    const gsl::not_null<SpinWeighted<ComplexDataVector, 2>*> script_djbar,
    const SpinWeighted<ComplexDataVector, 2>& dy_j,
    const SpinWeighted<ComplexDataVector, 2>& j,
    const SpinWeighted<ComplexDataVector, 0>& one_minus_y) {
  *script_djbar = 0.25 * one_minus_y *
                  (-2.0 * dy_j +
                   j * (conj(j) * dy_j + j * conj(dy_j)) / (1.0 + j * conj(j)));
  *linear_factor_for_h = 1.0 + j * conj(*script_djbar);
}

void ComputeBondiIntegrand<Tags::LinearFactorForConjugate<Tags::BondiH>>::
    apply_impl(
        const gsl::not_null<SpinWeighted<ComplexDataVector, 4>*>
            linear_factor_for_conjugate_h,
        const gsl::not_null<SpinWeighted<ComplexDataVector, 2>*> script_djbar,
        const SpinWeighted<ComplexDataVector, 2>& dy_j,
        const SpinWeighted<ComplexDataVector, 2>& j,
        const SpinWeighted<ComplexDataVector, 0>& one_minus_y) {
  *script_djbar = 0.25 * one_minus_y *
                  (-2.0 * dy_j +
                   j * (conj(j) * dy_j + j * conj(dy_j)) / (1.0 + j * conj(j)));
  *linear_factor_for_conjugate_h = j * (*script_djbar);
}

void ComputeBondiIntegrand<Tags::PoleOfIntegrand<Tags::KleinGordonPi>>::
    apply_impl(gsl::not_null<SpinWeighted<ComplexDataVector, 0>*>
                   pole_of_integrand_for_kg_pi,
               const SpinWeighted<ComplexDataVector, 1>& eth_kg_psi,
               const SpinWeighted<ComplexDataVector, 1>& bondi_u) {
  *pole_of_integrand_for_kg_pi =
      -real(bondi_u.data() * conj(eth_kg_psi).data());
}

void ComputeBondiIntegrand<Tags::RegularIntegrand<Tags::KleinGordonPi>>::
    apply_impl(gsl::not_null<SpinWeighted<ComplexDataVector, 0>*>
                   regular_integrand_for_kg_pi,
               // pre_swsh_derivative_tags
               const SpinWeighted<ComplexDataVector, 0>& dy_dy_kg_psi,
               const SpinWeighted<ComplexDataVector, 0>& dy_kg_psi,
               const SpinWeighted<ComplexDataVector, 1>& dy_bondi_u,
               const SpinWeighted<ComplexDataVector, 0>& dy_w,
               // swsh_derivative_tags
               const SpinWeighted<ComplexDataVector, 1>& eth_dy_kg_psi,
               const SpinWeighted<ComplexDataVector, 2>& eth_eth_kg_psi,
               const SpinWeighted<ComplexDataVector, 1>& eth_kg_psi,
               const SpinWeighted<ComplexDataVector, 1>& ethbar_j,
               const SpinWeighted<ComplexDataVector, 0>& ethbar_u,
               const SpinWeighted<ComplexDataVector, 1>& eth_beta,
               const SpinWeighted<ComplexDataVector, 1>& eth_j_jbar,
               const SpinWeighted<ComplexDataVector, 0>& eth_ethbar_kg_psi,
               // integration_independent_tags
               const SpinWeighted<ComplexDataVector, 2>& j,
               const SpinWeighted<ComplexDataVector, 0>& exp2beta,
               const SpinWeighted<ComplexDataVector, 0>& du_r_divided_by_r,
               const SpinWeighted<ComplexDataVector, 0>& one_minus_y,
               const SpinWeighted<ComplexDataVector, 1>& eth_r_divided_by_r,
               const SpinWeighted<ComplexDataVector, 0>& bondi_r,
               const SpinWeighted<ComplexDataVector, 0>& bondi_k,
               const SpinWeighted<ComplexDataVector, 1>& bondi_u,
               const SpinWeighted<ComplexDataVector, 0>& bondi_w) {
  SpinWeighted<ComplexDataVector, 1> eth_k = 0.5 * eth_j_jbar / bondi_k;
  // `from_lhs` comes from switching \Pi to \breve{\Pi} on the left-hand side
  SpinWeighted<ComplexDataVector, 0> from_lhs =
      du_r_divided_by_r * one_minus_y * dy_dy_kg_psi;

  SpinWeighted<ComplexDataVector, 0> n_psi1;
  SpinWeighted<ComplexDataVector, 0> n_psi2;
  SpinWeighted<ComplexDataVector, 0> n_psi3;
  SpinWeighted<ComplexDataVector, 0> n_psi4;
  SpinWeighted<ComplexDataVector, 0> tau;

  detail::klein_gordon_rhs_npsi1(make_not_null(&n_psi1), eth_beta, eth_kg_psi,
                                 eth_ethbar_kg_psi, bondi_k);
  detail::klein_gordon_rhs_npsi2(make_not_null(&n_psi2), j, eth_eth_kg_psi,
                                 eth_beta, eth_kg_psi, ethbar_j);
  detail::klein_gordon_rhs_npsi3(make_not_null(&n_psi3), eth_k, eth_kg_psi);
  detail::klein_gordon_rhs_npsi4_divided_by_one_minues_y_squared(
      make_not_null(&n_psi4), eth_kg_psi, dy_bondi_u, ethbar_u, bondi_u,
      eth_dy_kg_psi, dy_kg_psi, bondi_r, eth_r_divided_by_r);
  detail::klein_gordon_rhs_tau(make_not_null(&tau), one_minus_y, dy_w,
                               dy_kg_psi, dy_dy_kg_psi, bondi_w, bondi_r);

  *regular_integrand_for_kg_pi =
      0.25 * exp2beta / bondi_r * (n_psi1 - n_psi2 + n_psi3) -
      0.5 * bondi_r * n_psi4 + tau + from_lhs;
}
}  // namespace Cce
