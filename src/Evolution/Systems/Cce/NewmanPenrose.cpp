// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Evolution/Systems/Cce/NewmanPenrose.hpp"

#include "DataStructures/ComplexDataVector.hpp"
#include "DataStructures/SpinWeighted.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Evolution/Systems/Cce/Tags.hpp"
#include "NumericalAlgorithms/Spectral/SwshDerivatives.hpp"
#include "NumericalAlgorithms/Spectral/SwshInterpolation.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/TMPL.hpp"

namespace Cce {
namespace {
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
}  // namespace

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
