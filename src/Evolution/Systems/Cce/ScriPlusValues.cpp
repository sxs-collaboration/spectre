// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Evolution/Systems/Cce/ScriPlusValues.hpp"

#include <cstddef>

#include "DataStructures/DataBox/Prefixes.hpp"
#include "Evolution/Systems/Cce/Tags.hpp"
#include "NumericalAlgorithms/Spectral/SwshDerivatives.hpp"
#include "NumericalAlgorithms/Spectral/SwshFiltering.hpp"
#include "NumericalAlgorithms/Spectral/SwshInterpolation.hpp"

namespace Cce {

void CalculateScriPlusValue<Tags::News>::apply(
    const gsl::not_null<Scalar<SpinWeighted<ComplexDataVector, -2>>*> news,
    const Scalar<SpinWeighted<ComplexDataVector, 2>>& dy_du_bondi_j,
    const Scalar<SpinWeighted<ComplexDataVector, 0>>& beta,
    const Scalar<SpinWeighted<ComplexDataVector, 1>>& eth_beta,
    const Scalar<SpinWeighted<ComplexDataVector, 2>>& eth_eth_beta,
    const Scalar<SpinWeighted<ComplexDataVector, 0>>& boundary_r,
    const size_t l_max, const size_t number_of_radial_points) noexcept {
  const size_t number_of_angular_points =
      Spectral::Swsh::number_of_swsh_collocation_points(l_max);

  const SpinWeighted<ComplexDataVector, 2> dy_du_j_at_scri;
  make_const_view(make_not_null(&dy_du_j_at_scri), get(dy_du_bondi_j),
                  (number_of_radial_points - 1) * number_of_angular_points,
                  number_of_angular_points);

  const SpinWeighted<ComplexDataVector, 0> beta_at_scri;
  make_const_view(make_not_null(&beta_at_scri), get(beta),
                  (number_of_radial_points - 1) * number_of_angular_points,
                  number_of_angular_points);

  const SpinWeighted<ComplexDataVector, 1> eth_beta_at_scri;
  make_const_view(make_not_null(&eth_beta_at_scri), get(eth_beta),
                  (number_of_radial_points - 1) * number_of_angular_points,
                  number_of_angular_points);

  const SpinWeighted<ComplexDataVector, 2> eth_eth_beta_at_scri;
  make_const_view(make_not_null(&eth_eth_beta_at_scri), get(eth_eth_beta),
                  (number_of_radial_points - 1) * number_of_angular_points,
                  number_of_angular_points);

  // Note: -2 * r extra factor due to derivative l to y
  // Note also: extra factor of 2.0 for conversion to strain.
  get(*news) =
      2.0 * conj(-get(boundary_r) * exp(-2.0 * beta_at_scri) * dy_du_j_at_scri +
                 eth_eth_beta_at_scri + 2.0 * square(eth_beta_at_scri));
}

void CalculateScriPlusValue<Tags::TimeIntegral<Tags::ScriPlus<Tags::Psi4>>>::
    apply(const gsl::not_null<Scalar<SpinWeighted<ComplexDataVector, -2>>*>
              integral_of_psi_4,
          const Scalar<SpinWeighted<ComplexDataVector, 0>>& exp_2_beta,
          const Scalar<SpinWeighted<ComplexDataVector, 1>>& dy_bondi_u,
          const Scalar<SpinWeighted<ComplexDataVector, 2>>& eth_dy_bondi_u,
          const Scalar<SpinWeighted<ComplexDataVector, 2>>& dy_du_bondi_j,
          const Scalar<SpinWeighted<ComplexDataVector, 0>>& boundary_r,
          const Scalar<SpinWeighted<ComplexDataVector, 1>>& eth_r_divided_by_r,
          const size_t l_max, const size_t number_of_radial_points) noexcept {
  const size_t number_of_angular_points =
      Spectral::Swsh::number_of_swsh_collocation_points(l_max);

  const SpinWeighted<ComplexDataVector, 0> exp_2_beta_at_scri;
  make_const_view(make_not_null(&exp_2_beta_at_scri), get(exp_2_beta),
                  (number_of_radial_points - 1) * number_of_angular_points,
                  number_of_angular_points);

  const SpinWeighted<ComplexDataVector, 1> dy_u_at_scri;
  make_const_view(make_not_null(&dy_u_at_scri), get(dy_bondi_u),
                  (number_of_radial_points - 1) * number_of_angular_points,
                  number_of_angular_points);
  const SpinWeighted<ComplexDataVector, 2> eth_dy_u_at_scri;
  make_const_view(make_not_null(&eth_dy_u_at_scri), get(eth_dy_bondi_u),
                  (number_of_radial_points - 1) * number_of_angular_points,
                  number_of_angular_points);

  const SpinWeighted<ComplexDataVector, 1> eth_r_divided_by_r_view;
  make_const_view(make_not_null(&eth_r_divided_by_r_view),
                  get(eth_r_divided_by_r),
                  (number_of_radial_points - 1) * number_of_angular_points,
                  number_of_angular_points);

  const SpinWeighted<ComplexDataVector, 2> dy_du_j_at_scri;
  make_const_view(make_not_null(&dy_du_j_at_scri), get(dy_du_bondi_j),
                  (number_of_radial_points - 1) * number_of_angular_points,
                  number_of_angular_points);

  get(*integral_of_psi_4) =
      2.0 * get(boundary_r) *
      ((conj(eth_dy_u_at_scri) +
        conj(eth_r_divided_by_r_view) * conj(dy_u_at_scri)) +
       conj(dy_du_j_at_scri)) /
      exp_2_beta_at_scri;
}

void CalculateScriPlusValue<Tags::ScriPlus<Tags::Psi3>>::apply(
    const gsl::not_null<Scalar<SpinWeighted<ComplexDataVector, -1>>*> psi_3,
    const Scalar<SpinWeighted<ComplexDataVector, 0>>& exp_2_beta,
    const Scalar<SpinWeighted<ComplexDataVector, 1>>& eth_beta,
    const Scalar<SpinWeighted<ComplexDataVector, 0>>& eth_ethbar_beta,
    const Scalar<SpinWeighted<ComplexDataVector, -1>>& ethbar_eth_ethbar_beta,
    const Scalar<SpinWeighted<ComplexDataVector, 2>>& dy_du_bondi_j,
    const Scalar<SpinWeighted<ComplexDataVector, 1>>& ethbar_dy_du_bondi_j,
    const Scalar<SpinWeighted<ComplexDataVector, 0>>& boundary_r,
    const Scalar<SpinWeighted<ComplexDataVector, 1>>& eth_r_divided_by_r,
    const size_t l_max, const size_t number_of_radial_points) noexcept {
  const size_t number_of_angular_points =
      Spectral::Swsh::number_of_swsh_collocation_points(l_max);

  const SpinWeighted<ComplexDataVector, 0> exp_2_beta_at_scri;
  make_const_view(make_not_null(&exp_2_beta_at_scri), get(exp_2_beta),
                  (number_of_radial_points - 1) * number_of_angular_points,
                  number_of_angular_points);

  const SpinWeighted<ComplexDataVector, 1> eth_beta_at_scri;
  make_const_view(make_not_null(&eth_beta_at_scri), get(eth_beta),
                  (number_of_radial_points - 1) * number_of_angular_points,
                  number_of_angular_points);

  const SpinWeighted<ComplexDataVector, 0> eth_ethbar_beta_at_scri;
  make_const_view(make_not_null(&eth_ethbar_beta_at_scri), get(eth_ethbar_beta),
                  (number_of_radial_points - 1) * number_of_angular_points,
                  number_of_angular_points);

  const SpinWeighted<ComplexDataVector, -1> ethbar_eth_ethbar_beta_at_scri;
  make_const_view(make_not_null(&ethbar_eth_ethbar_beta_at_scri),
                  get(ethbar_eth_ethbar_beta),
                  (number_of_radial_points - 1) * number_of_angular_points,
                  number_of_angular_points);

  const SpinWeighted<ComplexDataVector, 1> eth_r_divided_by_r_view;
  make_const_view(make_not_null(&eth_r_divided_by_r_view),
                  get(eth_r_divided_by_r),
                  (number_of_radial_points - 1) * number_of_angular_points,
                  number_of_angular_points);

  const SpinWeighted<ComplexDataVector, 2> dy_du_j_at_scri;
  make_const_view(make_not_null(&dy_du_j_at_scri), get(dy_du_bondi_j),
                  (number_of_radial_points - 1) * number_of_angular_points,
                  number_of_angular_points);

  const SpinWeighted<ComplexDataVector, 1> ethbar_dy_du_j_at_scri;
  make_const_view(make_not_null(&ethbar_dy_du_j_at_scri),
                  get(ethbar_dy_du_bondi_j),
                  (number_of_radial_points - 1) * number_of_angular_points,
                  number_of_angular_points);

  // Attempting the consensus form; math still needs re-examining
  // extra factor of * -sqrt(2) to agree with SpEC tetrad normalization
  get(*psi_3) = 2.0 * conj(eth_beta_at_scri) +
                4.0 * conj(eth_beta_at_scri) * eth_ethbar_beta_at_scri +
                ethbar_eth_ethbar_beta_at_scri +
                get(boundary_r) *
                    (-(conj(ethbar_dy_du_j_at_scri) +
                       eth_r_divided_by_r_view * conj(dy_du_j_at_scri)) +
                     2.0 * eth_beta_at_scri * conj(dy_du_j_at_scri)) /
                    exp_2_beta_at_scri;
}

void CalculateScriPlusValue<Tags::ScriPlus<Tags::Psi2>>::apply(
    const gsl::not_null<Scalar<SpinWeighted<ComplexDataVector, 0>>*> psi_2,
    const Scalar<SpinWeighted<ComplexDataVector, 0>>& exp_2_beta,
    const Scalar<SpinWeighted<ComplexDataVector, 1>>& dy_bondi_q,
    const Scalar<SpinWeighted<ComplexDataVector, 0>>& ethbar_dy_bondi_q,
    const Scalar<SpinWeighted<ComplexDataVector, 1>>& dy_bondi_u,
    const Scalar<SpinWeighted<ComplexDataVector, 2>>& eth_dy_bondi_u,
    const Scalar<SpinWeighted<ComplexDataVector, 1>>& dy_dy_bondi_u,
    const Scalar<SpinWeighted<ComplexDataVector, 0>>& ethbar_dy_dy_bondi_u,
    const Scalar<SpinWeighted<ComplexDataVector, 0>>& dy_dy_bondi_w,
    const Scalar<SpinWeighted<ComplexDataVector, 2>>& dy_bondi_j,
    const Scalar<SpinWeighted<ComplexDataVector, 2>>& dy_du_bondi_j,
    const Scalar<SpinWeighted<ComplexDataVector, 0>>& boundary_r,
    const Scalar<SpinWeighted<ComplexDataVector, 1>>& eth_r_divided_by_r,
    const size_t l_max, const size_t number_of_radial_points) noexcept {
  const size_t number_of_angular_points =
      Spectral::Swsh::number_of_swsh_collocation_points(l_max);

  const SpinWeighted<ComplexDataVector, 2> dy_du_j_at_scri;
  make_const_view(make_not_null(&dy_du_j_at_scri), get(dy_du_bondi_j),
                  (number_of_radial_points - 1) * number_of_angular_points,
                  number_of_angular_points);

  const SpinWeighted<ComplexDataVector, 0> dy_dy_w_at_scri;
  make_const_view(make_not_null(&dy_dy_w_at_scri), get(dy_dy_bondi_w),
                  (number_of_radial_points - 1) * number_of_angular_points,
                  number_of_angular_points);

  const SpinWeighted<ComplexDataVector, 2> dy_j_at_scri;
  make_const_view(make_not_null(&dy_j_at_scri), get(dy_bondi_j),
                  (number_of_radial_points - 1) * number_of_angular_points,
                  number_of_angular_points);

  const SpinWeighted<ComplexDataVector, 1> dy_q_at_scri;
  make_const_view(make_not_null(&dy_q_at_scri), get(dy_bondi_q),
                  (number_of_radial_points - 1) * number_of_angular_points,
                  number_of_angular_points);
  const SpinWeighted<ComplexDataVector, 0> ethbar_dy_q_at_scri;
  make_const_view(make_not_null(&ethbar_dy_q_at_scri), get(ethbar_dy_bondi_q),
                  (number_of_radial_points - 1) * number_of_angular_points,
                  number_of_angular_points);

  const SpinWeighted<ComplexDataVector, 1> dy_u_at_scri;
  make_const_view(make_not_null(&dy_u_at_scri), get(dy_bondi_u),
                  (number_of_radial_points - 1) * number_of_angular_points,
                  number_of_angular_points);
  const SpinWeighted<ComplexDataVector, 2> eth_dy_u_at_scri;
  make_const_view(make_not_null(&eth_dy_u_at_scri), get(eth_dy_bondi_u),
                  (number_of_radial_points - 1) * number_of_angular_points,
                  number_of_angular_points);

  const SpinWeighted<ComplexDataVector, 1> dy_dy_u_at_scri;
  make_const_view(make_not_null(&dy_dy_u_at_scri), get(dy_dy_bondi_u),
                  (number_of_radial_points - 1) * number_of_angular_points,
                  number_of_angular_points);
  const SpinWeighted<ComplexDataVector, 0> ethbar_dy_dy_u_at_scri;
  make_const_view(make_not_null(&ethbar_dy_dy_u_at_scri),
                  get(ethbar_dy_dy_bondi_u),
                  (number_of_radial_points - 1) * number_of_angular_points,
                  number_of_angular_points);

  const SpinWeighted<ComplexDataVector, 0> exp_2_beta_at_scri;
  make_const_view(make_not_null(&exp_2_beta_at_scri), get(exp_2_beta),
                  (number_of_radial_points - 1) * number_of_angular_points,
                  number_of_angular_points);

  const SpinWeighted<ComplexDataVector, 1> eth_r_divided_by_r_view;
  make_const_view(make_not_null(&eth_r_divided_by_r_view),
                  get(eth_r_divided_by_r),
                  (number_of_radial_points - 1) * number_of_angular_points,
                  number_of_angular_points);

  get(*psi_2) =
      -0.5 * get(boundary_r) *
      (-exp_2_beta_at_scri * (conj(ethbar_dy_q_at_scri) +
                              eth_r_divided_by_r_view * conj(dy_q_at_scri)) +
       get(boundary_r) *
           (ethbar_dy_dy_u_at_scri +
            2.0 * conj(eth_r_divided_by_r_view) * dy_dy_u_at_scri +
            conj(ethbar_dy_dy_u_at_scri) +
            2.0 * eth_r_divided_by_r_view * conj(dy_dy_u_at_scri)) +
       2.0 * get(boundary_r) *
           (dy_j_at_scri *
                (conj(eth_dy_u_at_scri) +
                 conj(eth_r_divided_by_r_view) * conj(dy_u_at_scri)) +
            dy_j_at_scri * conj(dy_du_j_at_scri) - dy_dy_w_at_scri)) /
      exp_2_beta_at_scri;
}

void CalculateScriPlusValue<Tags::ScriPlus<Tags::Psi1>>::apply(
    const gsl::not_null<Scalar<SpinWeighted<ComplexDataVector, 1>>*> psi_1,
    const Scalar<SpinWeighted<ComplexDataVector, 0>>& dy_dy_bondi_beta,
    const Scalar<SpinWeighted<ComplexDataVector, 1>>& eth_dy_dy_bondi_beta,
    const Scalar<SpinWeighted<ComplexDataVector, 2>>& dy_bondi_j,
    const Scalar<SpinWeighted<ComplexDataVector, 1>>& dy_bondi_q,
    const Scalar<SpinWeighted<ComplexDataVector, 1>>& dy_dy_bondi_q,
    const Scalar<SpinWeighted<ComplexDataVector, 0>>& boundary_r,
    const Scalar<SpinWeighted<ComplexDataVector, 1>>& eth_r_divided_by_r,
    const size_t l_max, const size_t number_of_radial_points) noexcept {
  const size_t number_of_angular_points =
      Spectral::Swsh::number_of_swsh_collocation_points(l_max);

  const SpinWeighted<ComplexDataVector, 0> dy_dy_beta_at_scri;
  make_const_view(make_not_null(&dy_dy_beta_at_scri), get(dy_dy_bondi_beta),
                  (number_of_radial_points - 1) * number_of_angular_points,
                  number_of_angular_points);

  const SpinWeighted<ComplexDataVector, 1> eth_dy_dy_beta_at_scri;
  make_const_view(make_not_null(&eth_dy_dy_beta_at_scri),
                  get(eth_dy_dy_bondi_beta),
                  (number_of_radial_points - 1) * number_of_angular_points,
                  number_of_angular_points);

  const SpinWeighted<ComplexDataVector, 2> dy_j_at_scri;
  make_const_view(make_not_null(&dy_j_at_scri), get(dy_bondi_j),
                  (number_of_radial_points - 1) * number_of_angular_points,
                  number_of_angular_points);
  const SpinWeighted<ComplexDataVector, 1> dy_q_at_scri;
  make_const_view(make_not_null(&dy_q_at_scri), get(dy_bondi_q),
                  (number_of_radial_points - 1) * number_of_angular_points,
                  number_of_angular_points);

  const SpinWeighted<ComplexDataVector, 1> dy_dy_q_at_scri;
  make_const_view(make_not_null(&dy_dy_q_at_scri), get(dy_dy_bondi_q),
                  (number_of_radial_points - 1) * number_of_angular_points,
                  number_of_angular_points);

  const SpinWeighted<ComplexDataVector, 1> eth_r_divided_by_r_view;
  make_const_view(make_not_null(&eth_r_divided_by_r_view),
                  get(eth_r_divided_by_r),
                  (number_of_radial_points - 1) * number_of_angular_points,
                  number_of_angular_points);

  // extra -1/sqrt(2) factor to agree with SXS tetrad normalization
  get(*psi_1) = -0.5 * square(get(boundary_r)) *
                (6.0 * (eth_dy_dy_beta_at_scri +
                        2.0 * eth_r_divided_by_r_view * dy_dy_beta_at_scri) -
                 dy_j_at_scri * conj(dy_q_at_scri) - dy_dy_q_at_scri);
}

void CalculateScriPlusValue<Tags::ScriPlus<Tags::Psi0>>::apply(
    const gsl::not_null<Scalar<SpinWeighted<ComplexDataVector, 2>>*> psi_0,
    const Scalar<SpinWeighted<ComplexDataVector, 2>>& dy_bondi_j,
    const Scalar<SpinWeighted<ComplexDataVector, 2>>& dy_dy_dy_bondi_j,
    const Scalar<SpinWeighted<ComplexDataVector, 0>>& boundary_r,
    const size_t l_max, const size_t number_of_radial_points) noexcept {
  const size_t number_of_angular_points =
      Spectral::Swsh::number_of_swsh_collocation_points(l_max);

  const SpinWeighted<ComplexDataVector, 2> dy_dy_dy_j_at_scri;
  make_const_view(make_not_null(&dy_dy_dy_j_at_scri), get(dy_dy_dy_bondi_j),
                  (number_of_radial_points - 1) * number_of_angular_points,
                  number_of_angular_points);

  const SpinWeighted<ComplexDataVector, 2> dy_j_at_scri;
  make_const_view(make_not_null(&dy_j_at_scri), get(dy_bondi_j),
                  (number_of_radial_points - 1) * number_of_angular_points,
                  number_of_angular_points);

  // extra 1/2 factor to agree with SXS tetrad normalization
  get(*psi_0) = -pow<3>(get(boundary_r)) *
                (3.0 * conj(dy_j_at_scri) * square(dy_j_at_scri) -
                 2.0 * dy_dy_dy_j_at_scri);
}

void CalculateScriPlusValue<Tags::ScriPlus<Tags::Strain>>::apply(
    const gsl::not_null<Scalar<SpinWeighted<ComplexDataVector, -2>>*> strain,
    const Scalar<SpinWeighted<ComplexDataVector, 2>>& dy_bondi_j,
    const Scalar<SpinWeighted<ComplexDataVector, 2>>& eth_eth_retarded_time,
    const Scalar<SpinWeighted<ComplexDataVector, 0>>& boundary_r,
    const size_t l_max, const size_t number_of_radial_points) noexcept {
  const size_t number_of_angular_points =
      Spectral::Swsh::number_of_swsh_collocation_points(l_max);

  const SpinWeighted<ComplexDataVector, 2> dy_j_at_scri;
  make_const_view(make_not_null(&dy_j_at_scri), get(dy_bondi_j),
                  (number_of_radial_points - 1) * number_of_angular_points,
                  number_of_angular_points);

  // conjugate to retrieve the spin -2 quantity everyone else in the world
  // uses.
  get(*strain) =
      conj(-2.0 * get(boundary_r) * dy_j_at_scri + get(eth_eth_retarded_time));
}

void CalculateScriPlusValue<Tags::EthInertialRetardedTime>::apply(
    gsl::not_null<Scalar<SpinWeighted<ComplexDataVector, 1>>*>
        eth_inertial_time,
    const Scalar<SpinWeighted<ComplexDataVector, 0>>& inertial_time,
    const size_t l_max) noexcept {
  Spectral::Swsh::angular_derivatives<tmpl::list<Spectral::Swsh::Tags::Eth>>(
      l_max, 1, make_not_null(&get(*eth_inertial_time)), get(inertial_time));
}

void CalculateScriPlusValue<::Tags::dt<Tags::InertialRetardedTime>>::apply(
    const gsl::not_null<Scalar<DataVector>*> dt_inertial_time,
    const Scalar<SpinWeighted<ComplexDataVector, 0>>& exp_2_beta) noexcept {
  const SpinWeighted<ComplexDataVector, 0> exp_2_beta_at_scri;
  make_const_view(make_not_null(&exp_2_beta_at_scri), get(exp_2_beta),
                  get(exp_2_beta).size() - get(*dt_inertial_time).size(),
                  get(*dt_inertial_time).size());
  get(*dt_inertial_time) = real(exp_2_beta_at_scri.data());
}
}  // namespace Cce
