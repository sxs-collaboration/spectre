// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "ControlSystem/ControlErrors/Size/ComovingCharSpeedDerivative.hpp"

#include <cmath>
#include <cstddef>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tags/TempTensor.hpp"
#include "DataStructures/TempBuffer.hpp"
#include "DataStructures/Tensor/IndexType.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "DataStructures/Tensor/TypeAliases.hpp"
#include "Utilities/Gsl.hpp"

namespace control_system::size {
void comoving_char_speed_derivative(
    const gsl::not_null<Scalar<DataVector>*> result, const double lambda_00,
    const double dt_lambda_00, const double horizon_00,
    const double dt_horizon_00, const double grid_frame_excision_sphere_radius,
    const tnsr::i<DataVector, 3, Frame::Distorted>& excision_rhat,
    const tnsr::i<DataVector, 3, Frame::Distorted>& excision_normal_one_form,
    const Scalar<DataVector>& excision_normal_one_form_norm,
    const tnsr::I<DataVector, 3, Frame::Distorted>&
        distorted_components_of_grid_shift,
    const tnsr::II<DataVector, 3, Frame::Distorted>&
        inverse_spatial_metric_on_excision_boundary,
    const tnsr::Ijj<DataVector, 3, Frame::Distorted>&
        spatial_christoffel_second_kind,
    const tnsr::i<DataVector, 3, Frame::Distorted>& deriv_lapse,
    const tnsr::iJ<DataVector, 3, Frame::Distorted>& deriv_of_distorted_shift,
    const InverseJacobian<DataVector, 3, Frame::Grid, Frame::Distorted>&
        inverse_jacobian_grid_to_distorted) {
  const double Y00 = 0.25 * M_2_SQRTPI;

  // Define temporary storage.
  using excision_normal_vector_tag =
      ::Tags::TempI<1, 3, Frame::Distorted, DataVector>;
  using deriv_normal_one_form_tag =
      ::Tags::Tempi<2, 3, Frame::Distorted, DataVector>;
  TempBuffer<tmpl::list<excision_normal_vector_tag, deriv_normal_one_form_tag>>
      buffer(get<0>(excision_rhat).size());
  auto& excision_normal_vector = get<excision_normal_vector_tag>(buffer);
  auto& deriv_normal_one_form = get<deriv_normal_one_form_tag>(buffer);

  // excision_rhat is a tnsr:i when it is returned from a Strahlkorper.
  // But excision_rhat is a coordinate quantity, not a physical tensor, so
  // it can also be used as a tnsr::I.  Here we create a tnsr::I called
  // excision_rhat_vector that points into excision_rhat.
  const tnsr::I<DataVector, 3, Frame::Distorted> excision_rhat_vector{};
  for (size_t i = 0; i < 3; ++i) {
    // Is there a way to do this without the const_casts?
    // Note that excision_rhat_vector must be non-const because
    // we are changing it (by calling set_data_ref).
    // And set_data_ref expects a non-const argument.
    const_cast<DataVector*>(&excision_rhat_vector.get(i))  // NOLINT
        ->set_data_ref(
            const_cast<DataVector*>(&excision_rhat.get(i)));  // NOLINT
  }

  tenex::evaluate<ti::I>(
      make_not_null(&excision_normal_vector),
      excision_normal_one_form(ti::j) *
          inverse_spatial_metric_on_excision_boundary(ti::J, ti::I));

  // Fill result temporarily with all the terms in d/dlambda00 (n_hati)
  // that are proportional to n_hati.
  //   First, fill result with s_p s_j gamma^{pk} xi^i Gamma^j_{ki},
  //   which is (almost) the last term in d/dlambda00(n_hati).
  tenex::evaluate<>(
      result, excision_normal_vector(ti::K) * excision_normal_one_form(ti::j) *
                  excision_rhat_vector(ti::I) *
                  spatial_christoffel_second_kind(ti::J, ti::k, ti::i));
  //   Second, add to result s^k s_j InvJac^j_k / r_EB, which is
  //   (almost) the second term in d/dlambda00(n_hati)
  //     Note that the for the contraction s_j InvJac^j_k, the j on the s is
  //     a distorted-frame index but the j on the InvJac is a grid-frame index.
  //     This is really weird but it happens because some of the things are
  //     not tensors. (In particular, the map itself looks like
  //     x^{i_distorted} = x^{i_grid} * stuff, which equates grid and
  //     distorted incides).
  for (size_t j = 0; j < 3; ++j) {
    for (size_t k = 0; k < 3; ++k) {
      get(*result) += excision_normal_vector.get(k) *
                      excision_normal_one_form.get(j) *
                      inverse_jacobian_grid_to_distorted.get(j, k) /
                      grid_frame_excision_sphere_radius;
    }
  }
  //   Third, scale by norm^3 so that result contains
  //   1/a (n^k n_j InvJac^j_k / r_EB + n_p n_j gamma^{pk} xi^i Gamma^j_{ki}),
  //   which is (almost) the last two terms of d/dlambda00(n_hati).
  get(*result) /= cube(get(excision_normal_one_form_norm));

  // Set deriv_normal_one_form to the first two terms of d/dlambda00 (n_hati).
  // Possible memory optimization: excision_normal_vector isn't used anymore,
  // so that storage could be used for deriv_normal_one_form.
  tenex::evaluate<ti::i>(make_not_null(&deriv_normal_one_form),
                         -Y00 * (*result)() * excision_normal_one_form(ti::i));

  // Add the first term to deriv_normal_one_form, so that deriv_normal_one_form
  // contains the entire d/dlambda00 (n_hati).
  //     Note that the for the contraction s_j InvJac^j_i, the j on the s is
  //     a distorted-frame index but the j on the InvJac is a grid-frame index.
  //     This is really weird but it happens because some of the things are
  //     not tensors. (In particular, the map itself looks like
  //     x^{i_distorted} = x^{i_grid} * stuff, which equates grid and
  //     distorted incides).
  for (size_t i = 0; i < 3; ++i) {
    for (size_t j = 0; j < 3; ++j) {
      deriv_normal_one_form.get(i) +=
          excision_normal_one_form.get(j) *
          inverse_jacobian_grid_to_distorted.get(j, i) * Y00 /
          (grid_frame_excision_sphere_radius *
           get(excision_normal_one_form_norm));
    }
  }

  // Now put the actual result, i.e. d/dlambda00 (v_c), into result.
  // Do this term by term.  Ignore the overall factor of Y00 until the
  // end.
  //   First do the dt_horizon_00 term (without the normalization)
  tenex::evaluate<>(result, -excision_normal_one_form(ti::i) *
                                excision_rhat_vector(ti::I) * dt_horizon_00 /
                                horizon_00);
  //   Next do the shift term (again without the normalization)
  tenex::update<>(result,
                  (*result)() - excision_normal_one_form(ti::i) *
                                    excision_rhat_vector(ti::J) *
                                    deriv_of_distorted_shift(ti::j, ti::I));

  // Put in the norm factor.
  get(*result) /= get(excision_normal_one_form_norm);

  // Add the dlapse term (without the Y00 factor).
  tenex::update<>(
      result, (*result)() + deriv_lapse(ti::i) * excision_rhat_vector(ti::I));

  // Put in the Y00 factor.
  get(*result) *= Y00;

  // Add the final term to result
  tenex::update<>(
      result,
      (*result)() +
          deriv_normal_one_form(ti::i) *
              (Y00 * dt_lambda_00 * excision_rhat_vector(ti::I) +
               distorted_components_of_grid_shift(ti::I) -
               excision_rhat_vector(ti::I) * (dt_horizon_00 / horizon_00) *
                   (Y00 * lambda_00 - grid_frame_excision_sphere_radius)));
}
}  // namespace control_system::size
