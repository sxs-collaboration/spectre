// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Evolution/Systems/Cce/GaugeTransformBoundaryData.hpp"

#include "DataStructures/DataBox/Prefixes.hpp"
#include "DataStructures/SpinWeighted.hpp"
#include "DataStructures/Tags.hpp"
#include "DataStructures/Variables.hpp"
#include "Evolution/Systems/Cce/Tags.hpp"
#include "NumericalAlgorithms/Spectral/SwshCollocation.hpp"
#include "NumericalAlgorithms/Spectral/SwshDerivatives.hpp"
#include "NumericalAlgorithms/Spectral/SwshInterpolation.hpp"

/// \cond

namespace Cce {

void GaugeAdjustedBoundaryValue<Tags::BondiR>::apply(
    const gsl::not_null<Scalar<SpinWeighted<ComplexDataVector, 0>>*>
        evolution_gauge_r,
    const Scalar<SpinWeighted<ComplexDataVector, 0>>& cauchy_gauge_r,
    const Scalar<SpinWeighted<ComplexDataVector, 0>>& omega,
    const Spectral::Swsh::SwshInterpolator& interpolator) noexcept {
  interpolator.interpolate(make_not_null(&get(*evolution_gauge_r)),
                           get(cauchy_gauge_r));
  get(*evolution_gauge_r) = get(*evolution_gauge_r) * get(omega);
}

void GaugeAdjustedBoundaryValue<Tags::DuRDividedByR>::apply(
    const gsl::not_null<Scalar<SpinWeighted<ComplexDataVector, 0>>*>
        evolution_gauge_du_r_divided_by_r,
    const Scalar<SpinWeighted<ComplexDataVector, 0>>&
        cauchy_gauge_du_r_divided_by_r,
    const Scalar<SpinWeighted<ComplexDataVector, 1>>& bondi_u_at_scri,
    const Scalar<SpinWeighted<ComplexDataVector, 0>>& evolution_gauge_r,
    const Scalar<SpinWeighted<ComplexDataVector, 0>>& omega,
    const Scalar<SpinWeighted<ComplexDataVector, 0>>& du_omega,
    const Spectral::Swsh::SwshInterpolator& interpolator,
    const size_t l_max) noexcept {
  interpolator.interpolate(
      make_not_null(&get(*evolution_gauge_du_r_divided_by_r)),
      get(cauchy_gauge_du_r_divided_by_r));

  // Allocation
  const SpinWeighted<ComplexDataVector, 0> r_buffer =
      get(evolution_gauge_r) / get(omega);
  const auto eth_r =
      Spectral::Swsh::angular_derivative<Spectral::Swsh::Tags::Eth>(l_max, 1,
                                                                    r_buffer);
  get(*evolution_gauge_du_r_divided_by_r) +=
      0.5 *
          (get(bondi_u_at_scri) * conj(eth_r) +
           conj(get(bondi_u_at_scri)) * eth_r) /
          r_buffer +
      get(du_omega) / get(omega);
}

void GaugeAdjustedBoundaryValue<Tags::BondiJ>::apply(
    const gsl::not_null<Scalar<SpinWeighted<ComplexDataVector, 2>>*>
        evolution_gauge_j,
    const Scalar<SpinWeighted<ComplexDataVector, 2>>& cauchy_gauge_j,
    const Scalar<SpinWeighted<ComplexDataVector, 2>>& gauge_c,
    const Scalar<SpinWeighted<ComplexDataVector, 0>>& gauge_d,
    const Scalar<SpinWeighted<ComplexDataVector, 0>>& omega,
    const Spectral::Swsh::SwshInterpolator& interpolator) noexcept {
  interpolator.interpolate(make_not_null(&get(*evolution_gauge_j)),
                           get(cauchy_gauge_j));

  get(*evolution_gauge_j).data() =
      0.25 *
      (square(conj(get(gauge_d).data())) * get(*evolution_gauge_j).data() +
       square(get(gauge_c).data()) * conj(get(*evolution_gauge_j).data()) +
       2.0 * get(gauge_c).data() * conj(get(gauge_d).data()) *
           sqrt(1.0 + get(*evolution_gauge_j).data() *
                          conj(get(*evolution_gauge_j).data()))) /
      square(get(omega).data());
}

void GaugeAdjustedBoundaryValue<Tags::Dr<Tags::BondiJ>>::apply(
    const gsl::not_null<Scalar<SpinWeighted<ComplexDataVector, 2>>*>
        evolution_gauge_dr_j,
    const Scalar<SpinWeighted<ComplexDataVector, 2>>& cauchy_gauge_dr_j,
    const Scalar<SpinWeighted<ComplexDataVector, 2>>& cauchy_gauge_j,
    const Scalar<SpinWeighted<ComplexDataVector, 2>>& gauge_c,
    const Scalar<SpinWeighted<ComplexDataVector, 0>>& gauge_d,
    const Scalar<SpinWeighted<ComplexDataVector, 0>>& omega,
    const Spectral::Swsh::SwshInterpolator& interpolator,
    const size_t l_max) noexcept {
  interpolator.interpolate(make_not_null(&get(*evolution_gauge_dr_j)),
                           get(cauchy_gauge_dr_j));

  SpinWeighted<ComplexDataVector, 2> interpolated_j{
      Spectral::Swsh::number_of_swsh_collocation_points(l_max)};
  interpolator.interpolate(make_not_null(&interpolated_j), get(cauchy_gauge_j));
  get(*evolution_gauge_dr_j).data() =
      (0.25 * square(conj(get(gauge_d).data())) *
           get(*evolution_gauge_dr_j).data() +
       0.25 * square(get(gauge_c).data()) *
           conj(get(*evolution_gauge_dr_j).data()) +
       0.25 * get(gauge_c).data() * conj(get(gauge_d).data()) *
           (get(*evolution_gauge_dr_j).data() * conj(interpolated_j.data()) +
            conj(get(*evolution_gauge_dr_j).data()) * interpolated_j.data()) /
           sqrt(1.0 + interpolated_j.data() * conj(interpolated_j.data()))) /
      pow<3>(get(omega).data());
}

void GaugeAdjustedBoundaryValue<Tags::BondiBeta>::apply(
    const gsl::not_null<Scalar<SpinWeighted<ComplexDataVector, 0>>*>
        evolution_gauge_beta,
    const Scalar<SpinWeighted<ComplexDataVector, 0>>& cauchy_gauge_beta,
    const Scalar<SpinWeighted<ComplexDataVector, 0>>& omega,
    const Spectral::Swsh::SwshInterpolator& interpolator) noexcept {
  interpolator.interpolate(make_not_null(&get(*evolution_gauge_beta)),
                           get(cauchy_gauge_beta));
  get(*evolution_gauge_beta).data() -= 0.5 * log(get(omega).data());
}

void GaugeAdjustedBoundaryValue<Tags::BondiQ>::apply_impl(
    const gsl::not_null<SpinWeighted<ComplexDataVector, 1>*> evolution_gauge_q,
    const SpinWeighted<ComplexDataVector, 1>& cauchy_gauge_dr_u,
    const SpinWeighted<ComplexDataVector, 2>& volume_j,
    const SpinWeighted<ComplexDataVector, 2>& volume_dy_j,
    const SpinWeighted<ComplexDataVector, 0>& evolution_gauge_r,
    const SpinWeighted<ComplexDataVector, 0>& evolution_gauge_beta,
    const SpinWeighted<ComplexDataVector, 2>& gauge_c,
    const SpinWeighted<ComplexDataVector, 0>& gauge_d,
    const SpinWeighted<ComplexDataVector, 0>& omega,
    const SpinWeighted<ComplexDataVector, 1>& eth_omega,
    const Spectral::Swsh::SwshInterpolator& interpolator,
    const size_t l_max) noexcept {
  interpolator.interpolate(evolution_gauge_q, cauchy_gauge_dr_u);

  const SpinWeighted<ComplexDataVector, 2> evolution_gauge_j;
  make_const_view(make_not_null(&evolution_gauge_j), volume_j, 0,
                  Spectral::Swsh::number_of_swsh_collocation_points(l_max));

  const SpinWeighted<ComplexDataVector, 2> evolution_gauge_dy_j;
  make_const_view(make_not_null(&evolution_gauge_dy_j), volume_dy_j, 0,
                  Spectral::Swsh::number_of_swsh_collocation_points(l_max));

  // Allocation
  // optimization note: this allocation can be eliminated in favor of more
  // make_const_view s at the cost of more tightly constraining the order of
  // operations between this gauge transformation function and the volume
  // computations
  SpinWeighted<ComplexDataVector, 0> evolution_gauge_k;
  evolution_gauge_k.data() =
      sqrt(1.0 + evolution_gauge_j.data() * conj(evolution_gauge_j.data()));

  // we reuse the storage for the `evolution_gauge_q`. After the interpolation
  // it is \partial_r U(\hat x), and after the below assignment it is
  // \partial_{\hat r} \hat U.
  // Note also that by necessity we use derivatives with respect to y, which are
  // related to derivatives with respect to r by
  // 2.0 / r \partial_y = \partial_r
  *evolution_gauge_q =
      0.5 / pow<3>(omega) *
          (conj(gauge_d) * *evolution_gauge_q -
           gauge_c * conj(*evolution_gauge_q)) -
      ((eth_omega * evolution_gauge_k - conj(eth_omega) * evolution_gauge_j) *
           (-1.0 + (evolution_gauge_dy_j * conj(evolution_gauge_dy_j) -
                    0.25 *
                        square(evolution_gauge_j * conj(evolution_gauge_dy_j) +
                               evolution_gauge_dy_j * conj(evolution_gauge_j)) /
                        square(evolution_gauge_k))) -
       2.0 * (-conj(eth_omega) * evolution_gauge_dy_j +
              0.5 * eth_omega / evolution_gauge_k *
                  (evolution_gauge_j * conj(evolution_gauge_dy_j) +
                   conj(evolution_gauge_j) * evolution_gauge_dy_j))) *
          exp(2.0 * evolution_gauge_beta) / (square(evolution_gauge_r) * omega);

  *evolution_gauge_q = square(evolution_gauge_r) *
                       exp(-2.0 * evolution_gauge_beta) *
                       (evolution_gauge_j * conj(*evolution_gauge_q) +
                        evolution_gauge_k * *evolution_gauge_q);
}

void GaugeAdjustedBoundaryValue<Tags::BondiU>::apply(
    const gsl::not_null<Scalar<SpinWeighted<ComplexDataVector, 1>>*>
        evolution_gauge_u,
    const Scalar<SpinWeighted<ComplexDataVector, 1>>& cauchy_gauge_u,
    const Scalar<SpinWeighted<ComplexDataVector, 2>>& volume_j,
    const Scalar<SpinWeighted<ComplexDataVector, 0>>& evolution_gauge_r,
    const Scalar<SpinWeighted<ComplexDataVector, 0>>& evolution_gauge_beta,
    const Scalar<SpinWeighted<ComplexDataVector, 2>>& gauge_c,
    const Scalar<SpinWeighted<ComplexDataVector, 0>>& gauge_d,
    const Scalar<SpinWeighted<ComplexDataVector, 0>>& omega,
    const Scalar<SpinWeighted<ComplexDataVector, 1>>& eth_omega,
    const Spectral::Swsh::SwshInterpolator& interpolator,
    const size_t l_max) noexcept {
  interpolator.interpolate(make_not_null(&get(*evolution_gauge_u)),
                           get(cauchy_gauge_u));

  const SpinWeighted<ComplexDataVector, 2> evolution_gauge_j;
  make_const_view(make_not_null(&evolution_gauge_j), get(volume_j), 0,
                  Spectral::Swsh::number_of_swsh_collocation_points(l_max));

  get(*evolution_gauge_u) =
      0.5 / square(get(omega)) *
          (-get(gauge_c) * conj(get(*evolution_gauge_u)) +
           conj(get(gauge_d)) * get(*evolution_gauge_u)) +
      exp(2.0 * get(evolution_gauge_beta)) /
          (get(evolution_gauge_r) * get(omega)) *
          (conj(get(eth_omega)) * evolution_gauge_j -
           get(eth_omega) *
               sqrt(1.0 + evolution_gauge_j * conj(evolution_gauge_j)));
}

void GaugeAdjustedBoundaryValue<Tags::BondiW>::apply_impl(
    const gsl::not_null<SpinWeighted<ComplexDataVector, 0>*> evolution_gauge_w,
    const SpinWeighted<ComplexDataVector, 0>& cauchy_gauge_w,
    const SpinWeighted<ComplexDataVector, 2>& volume_j,
    const SpinWeighted<ComplexDataVector, 1>& evolution_gauge_u,
    const SpinWeighted<ComplexDataVector, 0>& evolution_gauge_beta,
    const SpinWeighted<ComplexDataVector, 1>& evolution_gauge_u_at_scri,
    const SpinWeighted<ComplexDataVector, 0>& evolution_gauge_r,
    const SpinWeighted<ComplexDataVector, 0>& omega,
    const SpinWeighted<ComplexDataVector, 0>& du_omega,
    const SpinWeighted<ComplexDataVector, 1>& eth_omega,
    const Spectral::Swsh::SwshInterpolator& interpolator,
    const size_t l_max) noexcept {
  interpolator.interpolate(evolution_gauge_w, cauchy_gauge_w);

  const SpinWeighted<ComplexDataVector, 2> evolution_gauge_j;
  make_const_view(make_not_null(&evolution_gauge_j), volume_j, 0,
                  Spectral::Swsh::number_of_swsh_collocation_points(l_max));

  // note that at this point in the computation `evolution_gauge_u` is \mathcal
  // U from the documentation, so
  // `evolution_gauge_u - evolution_gauge_u_at_scri`
  // is \mathcal U - \mathcal U^{(0)} = \hat U
  *evolution_gauge_w +=
      (omega - 1.0) / evolution_gauge_r - 2.0 * du_omega / omega -
      (conj(eth_omega) * (evolution_gauge_u - evolution_gauge_u_at_scri) +
       eth_omega * conj(evolution_gauge_u - evolution_gauge_u_at_scri)) /
          omega +
      0.5 * exp(2.0 * evolution_gauge_beta) /
          (square(omega) * evolution_gauge_r) *
          (square(conj(eth_omega)) * evolution_gauge_j +
           square(eth_omega) * conj(evolution_gauge_j) -
           2.0 * eth_omega * conj(eth_omega) *
               sqrt(1.0 + evolution_gauge_j * conj(evolution_gauge_j)));
}

void GaugeAdjustedBoundaryValue<Tags::BondiH>::apply_impl(
    const gsl::not_null<SpinWeighted<ComplexDataVector, 2>*> evolution_gauge_h,
    const SpinWeighted<ComplexDataVector, 2>& volume_j,
    const SpinWeighted<ComplexDataVector, 2>& cauchy_gauge_du_j,
    const SpinWeighted<ComplexDataVector, 2>& volume_dy_j,
    const SpinWeighted<ComplexDataVector, 1>& evolution_gauge_u_at_scri,
    const SpinWeighted<ComplexDataVector, 0>& evolution_gauge_r,
    const SpinWeighted<ComplexDataVector, 2>& gauge_c,
    const SpinWeighted<ComplexDataVector, 0>& gauge_d,
    const SpinWeighted<ComplexDataVector, 0>& omega,
    const SpinWeighted<ComplexDataVector, 0>& du_omega,
    const SpinWeighted<ComplexDataVector, 1>& eth_omega,
    const SpinWeighted<ComplexDataVector, 0>& evolution_gauge_du_r_divided_by_r,
    const Spectral::Swsh::SwshInterpolator& interpolator,
    const size_t l_max) noexcept {
  // optimization note: this has several spin-weighted derivatives, they can
  // be aggregated

  const SpinWeighted<ComplexDataVector, 2> evolution_gauge_j;
  make_const_view(make_not_null(&evolution_gauge_j), volume_j, 0,
                  Spectral::Swsh::number_of_swsh_collocation_points(l_max));

  const SpinWeighted<ComplexDataVector, 2> evolution_gauge_dy_j;
  make_const_view(make_not_null(&evolution_gauge_dy_j), volume_dy_j, 0,
                  Spectral::Swsh::number_of_swsh_collocation_points(l_max));

  // optimization note: this allocation could potentially be moved to the
  // DataBox and this function would then have an additional pointer argument
  // for the buffer.
  Variables<
      tmpl::list<::Tags::SpinWeighted<::Tags::TempScalar<0, ComplexDataVector>,
                                      std::integral_constant<int, 2>>,
                 ::Tags::SpinWeighted<::Tags::TempScalar<1, ComplexDataVector>,
                                      std::integral_constant<int, 0>>,
                 ::Tags::SpinWeighted<::Tags::TempScalar<2, ComplexDataVector>,
                                      std::integral_constant<int, 0>>,
                 ::Tags::SpinWeighted<::Tags::TempScalar<3, ComplexDataVector>,
                                      std::integral_constant<int, 1>>,
                 ::Tags::SpinWeighted<::Tags::TempScalar<4, ComplexDataVector>,
                                      std::integral_constant<int, 2>>,
                 ::Tags::SpinWeighted<::Tags::TempScalar<5, ComplexDataVector>,
                                      std::integral_constant<int, 0>>,
                 ::Tags::SpinWeighted<::Tags::TempScalar<6, ComplexDataVector>,
                                      std::integral_constant<int, 2>>,
                 ::Tags::SpinWeighted<::Tags::TempScalar<7, ComplexDataVector>,
                                      std::integral_constant<int, 1>>,
                 ::Tags::SpinWeighted<::Tags::TempScalar<8, ComplexDataVector>,
                                      std::integral_constant<int, 2>>,
                 ::Tags::SpinWeighted<::Tags::TempScalar<9, ComplexDataVector>,
                                      std::integral_constant<int, 0>>,
                 ::Tags::SpinWeighted<::Tags::TempScalar<10, ComplexDataVector>,
                                      std::integral_constant<int, 1>>>>
      computation_buffers{
          Spectral::Swsh::number_of_swsh_collocation_points(l_max)};

  auto& interpolated_j =
      get(get<::Tags::SpinWeighted<::Tags::TempScalar<0, ComplexDataVector>,
                                   std::integral_constant<int, 2>>>(
          computation_buffers));
  interpolated_j =
      0.25 * (square(gauge_d) * evolution_gauge_j +
              square(gauge_c) * conj(evolution_gauge_j) -
              2.0 * gauge_c * gauge_d *
                  sqrt(1.0 + evolution_gauge_j * conj(evolution_gauge_j)));

  auto& interpolated_k =
      get(get<::Tags::SpinWeighted<::Tags::TempScalar<1, ComplexDataVector>,
                                   std::integral_constant<int, 0>>>(
          computation_buffers));
  interpolated_k = sqrt(1.0 + interpolated_j * conj(interpolated_j));

  auto& evolution_gauge_k =
      get(get<::Tags::SpinWeighted<::Tags::TempScalar<2, ComplexDataVector>,
                                   std::integral_constant<int, 0>>>(
          computation_buffers));
  evolution_gauge_k = sqrt(1.0 + evolution_gauge_j * conj(evolution_gauge_j));

  auto& evolution_gauge_u_at_scri_bar_j =
      get(get<::Tags::SpinWeighted<::Tags::TempScalar<3, ComplexDataVector>,
                                   std::integral_constant<int, 1>>>(
          computation_buffers));
  evolution_gauge_u_at_scri_bar_j =
      conj(evolution_gauge_u_at_scri) * evolution_gauge_j;

  auto& eth_evolution_gauge_u_at_scri_bar_j =
      get(get<::Tags::SpinWeighted<::Tags::TempScalar<6, ComplexDataVector>,
                                   std::integral_constant<int, 2>>>(
          computation_buffers));
  auto& eth_evolution_gauge_r =
      get(get<::Tags::SpinWeighted<::Tags::TempScalar<7, ComplexDataVector>,
                                   std::integral_constant<int, 1>>>(
          computation_buffers));
  auto& eth_evolution_gauge_u_at_scri =
      get(get<::Tags::SpinWeighted<::Tags::TempScalar<8, ComplexDataVector>,
                                   std::integral_constant<int, 2>>>(
          computation_buffers));
  auto& ethbar_evolution_gauge_u_at_scri =
      get(get<::Tags::SpinWeighted<::Tags::TempScalar<9, ComplexDataVector>,
                                   std::integral_constant<int, 0>>>(
          computation_buffers));
  auto& ethbar_evolution_gauge_j =
      get(get<::Tags::SpinWeighted<::Tags::TempScalar<10, ComplexDataVector>,
                                   std::integral_constant<int, 1>>>(
          computation_buffers));

  Spectral::Swsh::angular_derivatives<
      tmpl::list<Spectral::Swsh::Tags::Eth, Spectral::Swsh::Tags::Eth,
                 Spectral::Swsh::Tags::Eth, Spectral::Swsh::Tags::Ethbar,
                 Spectral::Swsh::Tags::Ethbar>>(
      l_max, 1, make_not_null(&eth_evolution_gauge_u_at_scri_bar_j),
      make_not_null(&eth_evolution_gauge_r),
      make_not_null(&eth_evolution_gauge_u_at_scri),
      make_not_null(&ethbar_evolution_gauge_u_at_scri),
      make_not_null(&ethbar_evolution_gauge_j),
      evolution_gauge_u_at_scri_bar_j, evolution_gauge_r,
      evolution_gauge_u_at_scri, evolution_gauge_u_at_scri, evolution_gauge_j);

  auto& evolution_gauge_u_at_scri_bar_eth_j =
      get(get<::Tags::SpinWeighted<::Tags::TempScalar<4, ComplexDataVector>,
                                   std::integral_constant<int, 2>>>(
          computation_buffers));
  // this is \bar{\mathcal U}^{(0)} \eth J, but we have to calculate it
  // using quantities of spin 2 or less.
  // note the conversion Jacobian for the angular derivative
  evolution_gauge_u_at_scri_bar_eth_j =
      eth_evolution_gauge_u_at_scri_bar_j -
      evolution_gauge_j * conj(ethbar_evolution_gauge_u_at_scri) -
      2.0 * conj(evolution_gauge_u_at_scri) * eth_evolution_gauge_r *
          (evolution_gauge_dy_j) / evolution_gauge_r;

  auto& cauchy_du_omega =
      get(get<::Tags::SpinWeighted<::Tags::TempScalar<5, ComplexDataVector>,
                                   std::integral_constant<int, 0>>>(
          computation_buffers));
  cauchy_du_omega =
      du_omega - 0.5 * (evolution_gauge_u_at_scri * conj(eth_omega) +
                              conj(evolution_gauge_u_at_scri) * eth_omega);

  // Note that by necessity we use derivatives with respect to y, which are
  // related to derivatives with respect to r by
  // 2.0 / r \partial_y = \partial_r

  // Note also that when angular derivatives are taken numerically, they must be
  // corrected to the Bondi \eth with jacobians proportional to \eth R / R

  // `cauchy_du_omega` is \partial_u \hat \omega, determined by an angular
  // jacobian that depends on the \mathcal U^{(0)}.
  interpolator.interpolate(evolution_gauge_h, cauchy_gauge_du_j);
  *evolution_gauge_h =
      0.5 * (evolution_gauge_u_at_scri * ethbar_evolution_gauge_j -
             2.0 * evolution_gauge_u_at_scri * conj(eth_evolution_gauge_r) /
                 evolution_gauge_r * evolution_gauge_dy_j +
             evolution_gauge_u_at_scri_bar_eth_j) -
      2.0 * cauchy_du_omega / omega *
          (evolution_gauge_dy_j - evolution_gauge_j) -
      ethbar_evolution_gauge_u_at_scri * (evolution_gauge_j) +
      eth_evolution_gauge_u_at_scri * evolution_gauge_k +
      0.25 / square(omega) *
          (square(conj(gauge_d)) * *evolution_gauge_h +
           square(gauge_c) * conj(*evolution_gauge_h) +
           gauge_c * conj(gauge_d) *
               (*evolution_gauge_h * conj(interpolated_j) +
                interpolated_j * conj(*evolution_gauge_h)) /
               interpolated_k) +
      2.0 * evolution_gauge_du_r_divided_by_r * (evolution_gauge_dy_j);
}

void GaugeUpdateTimeDerivatives::apply(
    const gsl::not_null<tnsr::i<DataVector, 3>*> cartesian_cauchy_du_x,
    const gsl::not_null<Scalar<SpinWeighted<ComplexDataVector, 1>>*>
        evolution_gauge_u_at_scri,
    const gsl::not_null<Scalar<SpinWeighted<ComplexDataVector, 1>>*> volume_u,
    const gsl::not_null<Scalar<SpinWeighted<ComplexDataVector, 0>>*> du_omega,
    const tnsr::i<DataVector, 3>& cartesian_cauchy_coordinates,
    const Scalar<SpinWeighted<ComplexDataVector, 0>>& omega,
    const Scalar<SpinWeighted<ComplexDataVector, 1>>& eth_omega,
    const size_t l_max) noexcept {
  const size_t number_of_angular_points =
      Spectral::Swsh::number_of_swsh_collocation_points(l_max);
  const size_t number_of_radial_points =
      get(*volume_u).size() / number_of_angular_points;

  const ComplexDataVector u_scri_view;
  make_const_view(make_not_null(&u_scri_view), get(*volume_u).data(),
                  (number_of_radial_points - 1) * number_of_angular_points,
                  number_of_angular_points);

  get(*evolution_gauge_u_at_scri).data() = u_scri_view;

  for (size_t i = 0; i < number_of_radial_points; ++i) {
    ComplexDataVector angular_view_u{
        get(*volume_u).data().data() +
            i * Spectral::Swsh::number_of_swsh_collocation_points(l_max),
        Spectral::Swsh::number_of_swsh_collocation_points(l_max)};
    angular_view_u -= get(*evolution_gauge_u_at_scri).data();
  }

  Variables<
      tmpl::list<::Tags::SpinWeighted<::Tags::TempScalar<0, ComplexDataVector>,
                                      std::integral_constant<int, 0>>,
                 ::Tags::SpinWeighted<::Tags::TempScalar<1, ComplexDataVector>,
                                      std::integral_constant<int, 0>>,
                 ::Tags::SpinWeighted<::Tags::TempScalar<2, ComplexDataVector>,
                                      std::integral_constant<int, 0>>,
                 ::Tags::SpinWeighted<::Tags::TempScalar<3, ComplexDataVector>,
                                      std::integral_constant<int, 1>>,
                 ::Tags::SpinWeighted<::Tags::TempScalar<4, ComplexDataVector>,
                                      std::integral_constant<int, 1>>,
                 ::Tags::SpinWeighted<::Tags::TempScalar<5, ComplexDataVector>,
                                      std::integral_constant<int, 1>>,
                 ::Tags::SpinWeighted<::Tags::TempScalar<6, ComplexDataVector>,
                                      std::integral_constant<int, 2>>,
                 ::Tags::SpinWeighted<::Tags::TempScalar<7, ComplexDataVector>,
                                      std::integral_constant<int, 0>>>>
      computation_buffers{number_of_angular_points};

  auto& x =
      get(get<::Tags::SpinWeighted<::Tags::TempScalar<0, ComplexDataVector>,
                                   std::integral_constant<int, 0>>>(
          computation_buffers));
  auto& y =
      get(get<::Tags::SpinWeighted<::Tags::TempScalar<1, ComplexDataVector>,
                                   std::integral_constant<int, 0>>>(
          computation_buffers));
  auto& z =
      get(get<::Tags::SpinWeighted<::Tags::TempScalar<2, ComplexDataVector>,
                                   std::integral_constant<int, 0>>>(
          computation_buffers));
  // Switch to complex so we can take spin-weighted derivatives
  x.data() =
      std::complex<double>(1.0, 0.0) * get<0>(cartesian_cauchy_coordinates);
  y.data() =
      std::complex<double>(1.0, 0.0) * get<1>(cartesian_cauchy_coordinates);
  z.data() =
      std::complex<double>(1.0, 0.0) * get<2>(cartesian_cauchy_coordinates);

  auto& eth_x =
      get(get<::Tags::SpinWeighted<::Tags::TempScalar<3, ComplexDataVector>,
                                   std::integral_constant<int, 1>>>(
          computation_buffers));
  auto& eth_y =
      get(get<::Tags::SpinWeighted<::Tags::TempScalar<4, ComplexDataVector>,
                                   std::integral_constant<int, 1>>>(
          computation_buffers));
  auto& eth_z =
      get(get<::Tags::SpinWeighted<::Tags::TempScalar<5, ComplexDataVector>,
                                   std::integral_constant<int, 1>>>(
          computation_buffers));
  auto& eth_evolution_gauge_u_at_scri =
      get(get<::Tags::SpinWeighted<::Tags::TempScalar<6, ComplexDataVector>,
                                   std::integral_constant<int, 2>>>(
          computation_buffers));

  auto& ethbar_evolution_gauge_u_at_scri =
      get(get<::Tags::SpinWeighted<::Tags::TempScalar<7, ComplexDataVector>,
                                   std::integral_constant<int, 0>>>(
          computation_buffers));
  Spectral::Swsh::angular_derivatives<
      tmpl::list<Spectral::Swsh::Tags::Eth, Spectral::Swsh::Tags::Eth,
                 Spectral::Swsh::Tags::Eth, Spectral::Swsh::Tags::Eth,
                 Spectral::Swsh::Tags::Ethbar>>(
      l_max, 1, make_not_null(&eth_x), make_not_null(&eth_y),
      make_not_null(&eth_z), make_not_null(&eth_evolution_gauge_u_at_scri),
      make_not_null(&ethbar_evolution_gauge_u_at_scri), x, y, z,
      get(*evolution_gauge_u_at_scri), get(*evolution_gauge_u_at_scri));
  get<0>(*cartesian_cauchy_du_x) =
      real(conj(get(*evolution_gauge_u_at_scri).data()) * eth_x.data());
  get<1>(*cartesian_cauchy_du_x) =
      real(conj(get(*evolution_gauge_u_at_scri).data()) * eth_y.data());
  get<2>(*cartesian_cauchy_du_x) =
      real(conj(get(*evolution_gauge_u_at_scri).data()) * eth_z.data());

  get(*du_omega) =
      0.25 *
          (ethbar_evolution_gauge_u_at_scri +
           conj(ethbar_evolution_gauge_u_at_scri)) *
          get(omega) +
      0.5 * (get(*evolution_gauge_u_at_scri) * conj(get(eth_omega)) +
             conj(get(*evolution_gauge_u_at_scri)) * get(eth_omega));
}

namespace detail {
void gauge_update_jacobian_from_coordinates_apply_impl(
    const gsl::not_null<Scalar<SpinWeighted<ComplexDataVector, 2>>*>
        gauge_factor_spin_2,
    const gsl::not_null<Scalar<SpinWeighted<ComplexDataVector, 0>>*>
        gauge_factor_spin_0,
    const gsl::not_null<
        tnsr::i<DataVector, 2, ::Frame::Spherical<::Frame::Inertial>>*>
        angular_source_coordinates,
    const tnsr::i<DataVector, 3>& cartesian_source_coordinates,
    const size_t l_max) noexcept {
  const size_t number_of_angular_points =
      Spectral::Swsh::number_of_swsh_collocation_points(l_max);
  Variables<
      tmpl::list<::Tags::TempiJ<0, 3>,
                 ::Tags::TempiJ<0, 2, ::Frame::Spherical<::Frame::Inertial>>>>
      tensor_buffers{number_of_angular_points};
  auto& angular_derivative_cartesian_source_coordinates =
      get<::Tags::TempiJ<0, 3>>(tensor_buffers);
  auto& angular_derivative_angular_source_coordinates =
      get<::Tags::TempiJ<0, 2, ::Frame::Spherical<::Frame::Inertial>>>(
          tensor_buffers);

  Variables<
      tmpl::list<::Tags::SpinWeighted<::Tags::TempScalar<0, ComplexDataVector>,
                                      std::integral_constant<int, 0>>,
                 ::Tags::SpinWeighted<::Tags::TempScalar<1, ComplexDataVector>,
                                      std::integral_constant<int, 0>>,
                 ::Tags::SpinWeighted<::Tags::TempScalar<2, ComplexDataVector>,
                                      std::integral_constant<int, 0>>,
                 ::Tags::SpinWeighted<::Tags::TempScalar<3, ComplexDataVector>,
                                      std::integral_constant<int, 1>>,
                 ::Tags::SpinWeighted<::Tags::TempScalar<4, ComplexDataVector>,
                                      std::integral_constant<int, 1>>,
                 ::Tags::SpinWeighted<::Tags::TempScalar<5, ComplexDataVector>,
                                      std::integral_constant<int, 1>>>>
      computation_buffers{number_of_angular_points};

  auto& x =
      get(get<::Tags::SpinWeighted<::Tags::TempScalar<0, ComplexDataVector>,
                                   std::integral_constant<int, 0>>>(
          computation_buffers));
  auto& y =
      get(get<::Tags::SpinWeighted<::Tags::TempScalar<1, ComplexDataVector>,
                                   std::integral_constant<int, 0>>>(
          computation_buffers));
  auto& z =
      get(get<::Tags::SpinWeighted<::Tags::TempScalar<2, ComplexDataVector>,
                                   std::integral_constant<int, 0>>>(
          computation_buffers));
  // Switch to complex so we can take spin-weighted derivatives
  x.data() =
      std::complex<double>(1.0, 0.0) * get<0>(cartesian_source_coordinates);
  y.data() =
      std::complex<double>(1.0, 0.0) * get<1>(cartesian_source_coordinates);
  z.data() =
      std::complex<double>(1.0, 0.0) * get<2>(cartesian_source_coordinates);

  auto& eth_x =
      get(get<::Tags::SpinWeighted<::Tags::TempScalar<3, ComplexDataVector>,
                                   std::integral_constant<int, 1>>>(
          computation_buffers));
  auto& eth_y =
      get(get<::Tags::SpinWeighted<::Tags::TempScalar<4, ComplexDataVector>,
                                   std::integral_constant<int, 1>>>(
          computation_buffers));
  auto& eth_z =
      get(get<::Tags::SpinWeighted<::Tags::TempScalar<5, ComplexDataVector>,
                                   std::integral_constant<int, 1>>>(
          computation_buffers));
  Spectral::Swsh::angular_derivatives<
      tmpl::list<Spectral::Swsh::Tags::Eth, Spectral::Swsh::Tags::Eth,
                 Spectral::Swsh::Tags::Eth>>(l_max, 1, make_not_null(&eth_x),
                                             make_not_null(&eth_y),
                                             make_not_null(&eth_z), x, y, z);
  get<0, 0>(angular_derivative_cartesian_source_coordinates) =
      -real(eth_x.data());
  get<1, 0>(angular_derivative_cartesian_source_coordinates) =
      -imag(eth_x.data());
  get<0, 1>(angular_derivative_cartesian_source_coordinates) =
      -real(eth_y.data());
  get<1, 1>(angular_derivative_cartesian_source_coordinates) =
      -imag(eth_y.data());
  get<0, 2>(angular_derivative_cartesian_source_coordinates) =
      -real(eth_z.data());
  get<1, 2>(angular_derivative_cartesian_source_coordinates) =
      -imag(eth_z.data());

  for (size_t i = 0; i < 2; ++i) {
    angular_derivative_angular_source_coordinates.get(i, 0) =
        cos(get<1>(*angular_source_coordinates)) *
            cos(get<0>(*angular_source_coordinates)) *
            angular_derivative_cartesian_source_coordinates.get(i, 0) +
        cos(get<0>(*angular_source_coordinates)) *
            sin(get<1>(*angular_source_coordinates)) *
            angular_derivative_cartesian_source_coordinates.get(i, 1) -
        sin(get<0>(*angular_source_coordinates)) *
            angular_derivative_cartesian_source_coordinates.get(i, 2);
    angular_derivative_angular_source_coordinates.get(i, 1) =
        -sin(get<1>(*angular_source_coordinates)) *
            angular_derivative_cartesian_source_coordinates.get(i, 0) +
        cos(get<1>(*angular_source_coordinates)) *
            angular_derivative_cartesian_source_coordinates.get(i, 1);
  }

  // in the standard evaluation, this is \hat c
  get(*gauge_factor_spin_2).data() =
      std::complex<double>(1.0, 0.0) *
          (get<0, 0>(angular_derivative_angular_source_coordinates) -
           get<1, 1>(angular_derivative_angular_source_coordinates)) +
      std::complex<double>(0.0, 1.0) *
          (get<1, 0>(angular_derivative_angular_source_coordinates) +
           get<0, 1>(angular_derivative_angular_source_coordinates));
  // in the standard evaluation, this is \hat d
  get(*gauge_factor_spin_0).data() =
      std::complex<double>(1.0, 0.0) *
          (get<0, 0>(angular_derivative_angular_source_coordinates) +
           get<1, 1>(angular_derivative_angular_source_coordinates)) +
      std::complex<double>(0.0, 1.0) *
          (-get<1, 0>(angular_derivative_angular_source_coordinates) +
           get<0, 1>(angular_derivative_angular_source_coordinates));
}
}  // namespace detail

void GaugeUpdateOmega::apply(
    const gsl::not_null<Scalar<SpinWeighted<ComplexDataVector, 0>>*> omega,
    const gsl::not_null<Scalar<SpinWeighted<ComplexDataVector, 1>>*> eth_omega,
    const Scalar<SpinWeighted<ComplexDataVector, 2>>& gauge_c,
    const Scalar<SpinWeighted<ComplexDataVector, 0>>& gauge_d,
    const size_t l_max) noexcept {
  get(*omega) = 0.5 * sqrt(get(gauge_d).data() * conj(get(gauge_d).data()) -
                           get(gauge_c).data() * conj(get(gauge_c).data()));

  Spectral::Swsh::angular_derivatives<tmpl::list<Spectral::Swsh::Tags::Eth>>(
      l_max, 1, make_not_null(&get(*eth_omega)), get(*omega));
}

void InitializeGauge::apply(
    const gsl::not_null<
        tnsr::i<DataVector, 2, ::Frame::Spherical<::Frame::Inertial>>*>
        angular_cauchy_coordinates,
    const gsl::not_null<tnsr::i<DataVector, 3>*> cartesian_cauchy_coordinates,
    const gsl::not_null<Scalar<SpinWeighted<ComplexDataVector, 2>>*> gauge_c,
    const gsl::not_null<Scalar<SpinWeighted<ComplexDataVector, 0>>*> gauge_d,
    const gsl::not_null<Scalar<SpinWeighted<ComplexDataVector, 0>>*> omega,
    const size_t l_max) noexcept {
  const auto& collocation = Spectral::Swsh::cached_collocation_metadata<
      Spectral::Swsh::ComplexRepresentation::Interleaved>(l_max);
  for (const auto& collocation_point : collocation) {
    get<0>(*angular_cauchy_coordinates)[collocation_point.offset] =
        collocation_point.theta;
    get<1>(*angular_cauchy_coordinates)[collocation_point.offset] =
        collocation_point.phi;
  }
  get<0>(*cartesian_cauchy_coordinates) =
      sin(get<0>(*angular_cauchy_coordinates)) *
      cos(get<1>(*angular_cauchy_coordinates));
  get<1>(*cartesian_cauchy_coordinates) =
      sin(get<0>(*angular_cauchy_coordinates)) *
      sin(get<1>(*angular_cauchy_coordinates));
  get<2>(*cartesian_cauchy_coordinates) =
      cos(get<0>(*angular_cauchy_coordinates));
  get(*omega).data() = 1.0;
  get(*gauge_c).data() = 0.0;
  get(*gauge_d).data() = 2.0;
}

}  // namespace Cce

/// \endcond
