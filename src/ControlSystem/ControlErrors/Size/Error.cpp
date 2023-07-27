// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "ControlSystem/ControlErrors/Size/Error.hpp"

#include <algorithm>
#include <cstddef>
#include <string>

#include "ControlSystem/ControlErrors/Size/Info.hpp"
#include "ControlSystem/ControlErrors/Size/State.hpp"
#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tags/TempTensor.hpp"
#include "DataStructures/TempBuffer.hpp"
#include "DataStructures/Tensor/EagerMath/Magnitude.hpp"
#include "DataStructures/Tensor/IndexType.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "DataStructures/Tensor/TypeAliases.hpp"
#include "Domain/FunctionsOfTime/FunctionOfTime.hpp"
#include "NumericalAlgorithms/Interpolation/ZeroCrossingPredictor.hpp"
#include "NumericalAlgorithms/SphericalHarmonics/Strahlkorper.hpp"
#include "NumericalAlgorithms/SphericalHarmonics/StrahlkorperFunctions.hpp"
#include "NumericalAlgorithms/SphericalHarmonics/TagsTypeAliases.hpp"
#include "PointwiseFunctions/GeneralRelativity/Surfaces/AreaElement.hpp"
#include "PointwiseFunctions/GeneralRelativity/Surfaces/RadialDistance.hpp"
#include "PointwiseFunctions/GeneralRelativity/Surfaces/SurfaceIntegralOfScalar.hpp"
#include "Utilities/GenerateInstantiations.hpp"
#include "Utilities/Gsl.hpp"

namespace control_system::size {

template <typename Frame>
ErrorDiagnostics control_error(
    const gsl::not_null<Info*> info,
    const gsl::not_null<intrp::ZeroCrossingPredictor*> predictor_char_speed,
    const gsl::not_null<intrp::ZeroCrossingPredictor*>
        predictor_comoving_char_speed,
    const gsl::not_null<intrp::ZeroCrossingPredictor*> predictor_delta_radius,
    const double time, const double control_error_delta_r,
    const double dt_lambda_00, const Strahlkorper<Frame>& apparent_horizon,
    const Strahlkorper<Frame>& excision_boundary,
    const Scalar<DataVector>& lapse_on_excision_boundary,
    const tnsr::I<DataVector, 3, Frame>& frame_components_of_grid_shift,
    const tnsr::ii<DataVector, 3, Frame>& spatial_metric_on_excision_boundary,
    const tnsr::II<DataVector, 3, Frame>&
        inverse_spatial_metric_on_excision_boundary) {
  const double Y00 = 0.25 * M_2_SQRTPI;

  // Define various quantities on excision boundary.
  // Declare a TempBuffer to do this with a single memory allocation.
  using excision_theta_phi_tag =
      ::Tags::Tempi<0, 2, ::Frame::Spherical<Frame>, DataVector>;
  using excision_radius_tag = ::Tags::TempScalar<0, DataVector>;
  using excision_rhat_tag = ::Tags::Tempi<1, 3, Frame, DataVector>;
  using excision_normal_one_form_tag = ::Tags::Tempi<2, 3, Frame, DataVector>;
  using excision_jacobian_tag =
      ::Tags::TempTensor<0, StrahlkorperTags::aliases::Jacobian<Frame>>;
  using excision_inv_jacobian_tag =
      ::Tags::TempTensor<1, StrahlkorperTags::aliases::InvJacobian<Frame>>;
  using excision_dx_radius_tag = Tags::Tempi<3, 3, Frame, DataVector>;
  using area_element_tag = ::Tags::TempScalar<1, DataVector>;
  using distorted_normal_dot_unit_coord_vector_tag =
      ::Tags::TempScalar<3, DataVector>;
  using comoving_char_speed_tag = ::Tags::TempScalar<4, DataVector>;
  using radial_distance_tag = ::Tags::TempScalar<5, DataVector>;
  using excision_normal_one_form_norm_tag = ::Tags::TempScalar<6, DataVector>;
  using unity_tag = ::Tags::TempScalar<7, DataVector>;
  using char_speed_tag = ::Tags::TempScalar<8, DataVector>;

  TempBuffer<
      tmpl::list<excision_theta_phi_tag, excision_radius_tag, excision_rhat_tag,
                 excision_normal_one_form_tag, excision_jacobian_tag,
                 excision_inv_jacobian_tag, excision_dx_radius_tag,
                 area_element_tag, distorted_normal_dot_unit_coord_vector_tag,
                 comoving_char_speed_tag, radial_distance_tag,
                 excision_normal_one_form_norm_tag, unity_tag, char_speed_tag>>
      buffer(excision_boundary.ylm_spherepack().physical_size());
  auto& excision_radius = get<excision_radius_tag>(buffer);
  auto& excision_theta_phi = get<excision_theta_phi_tag>(buffer);
  auto& excision_rhat = get<excision_rhat_tag>(buffer);
  auto& excision_normal_one_form = get<excision_normal_one_form_tag>(buffer);
  auto& excision_jacobian = get<excision_jacobian_tag>(buffer);
  auto& excision_inv_jacobian = get<excision_inv_jacobian_tag>(buffer);
  auto& excision_dx_radius = get<excision_dx_radius_tag>(buffer);
  auto& area_element = get<area_element_tag>(buffer);
  auto& distorted_normal_dot_unit_coord_vector =
      get<distorted_normal_dot_unit_coord_vector_tag>(buffer);
  auto& comoving_char_speed = get<comoving_char_speed_tag>(buffer);
  auto& radial_distance = get<radial_distance_tag>(buffer);
  auto& excision_normal_one_form_norm =
      get<excision_normal_one_form_norm_tag>(buffer);
  auto& unity = get<unity_tag>(buffer);
  auto& characteristic_speed_on_excision_boundary = get<char_speed_tag>(buffer);

  // Compute the quantities on the excision boundary.
  StrahlkorperFunctions::theta_phi(make_not_null(&excision_theta_phi),
                                   excision_boundary);
  StrahlkorperFunctions::radius(make_not_null(&excision_radius),
                                excision_boundary);
  // rhat is x^i/r
  StrahlkorperFunctions::rhat(make_not_null(&excision_rhat),
                              excision_theta_phi);
  StrahlkorperFunctions::jacobian(make_not_null(&excision_jacobian),
                                  excision_theta_phi);
  StrahlkorperFunctions::inv_jacobian(make_not_null(&excision_inv_jacobian),
                                      excision_theta_phi);
  StrahlkorperFunctions::cartesian_derivs_of_scalar(
      make_not_null(&excision_dx_radius), excision_radius, excision_boundary,
      excision_radius, excision_inv_jacobian);
  StrahlkorperFunctions::normal_one_form(
      make_not_null(&excision_normal_one_form), excision_dx_radius,
      excision_rhat);
  magnitude(make_not_null(&excision_normal_one_form_norm),
            excision_normal_one_form,
            inverse_spatial_metric_on_excision_boundary);
  StrahlkorperGr::area_element(make_not_null(&area_element),
                               spatial_metric_on_excision_boundary,
                               excision_jacobian, excision_normal_one_form,
                               excision_radius, excision_rhat);

  // distorted_normal_dot_unit_coord_vector is nhat_i x^i/r where
  // nhat_i is the distorted-frame unit normal to the excision
  // boundary (pointing INTO the hole, i.e. out of the domain), and
  // x^i/r is the distorted-frame (or equivalently the grid frame
  // because it is invariant between these two frames because of the
  // required limiting behavior of the map we choose) Euclidean normal
  // vector from the center of the excision-boundary Strahlkorper to
  // each point on the excision-boundary Strahlkorper.
  //
  // Minus sign is because we want the normal pointing into the hole,
  // not out of the hole.
  get(distorted_normal_dot_unit_coord_vector) =
      -get<0>(excision_normal_one_form) * get<0>(excision_rhat);
  for (size_t i = 1; i < 3; ++i) {
    get(distorted_normal_dot_unit_coord_vector) -=
        excision_normal_one_form.get(i) * excision_rhat.get(i);
  }
  get(distorted_normal_dot_unit_coord_vector) /=
      get(excision_normal_one_form_norm);

  // Average value of distorted_normal_dot_unit_coord_vector on the excision
  // boundary.  Compute the average by integrating.
  get(unity) = 1.0;
  const double avg_distorted_normal_dot_unit_coord_vector =
      StrahlkorperGr::surface_integral_of_scalar(
          area_element, distorted_normal_dot_unit_coord_vector,
          excision_boundary) /
      StrahlkorperGr::surface_integral_of_scalar(area_element, unity,
                                                 excision_boundary);

  // Compute char speed on excision boundary, Eq. 87 in ArXiv:1211.6079
  get(characteristic_speed_on_excision_boundary) =
      -get(lapse_on_excision_boundary);
  for (size_t i = 0; i < 3; ++i) {
    // Plus sign here is because we want the normal pointing into the hole,
    // not out of the hole, and excision_normal_one_form points out of the hole.
    get(characteristic_speed_on_excision_boundary) +=
        frame_components_of_grid_shift.get(i) *
        excision_normal_one_form.get(i) / get(excision_normal_one_form_norm);
  }

  // Minimum char speed on the excision boundary
  const double min_char_speed =
      min(get(characteristic_speed_on_excision_boundary));

  // comoving_char_speed is the quantity v_c, Eq. 98 in ArXiv:1211.6079
  // (but implemented in a simpler way).
  get(comoving_char_speed) =
      get(characteristic_speed_on_excision_boundary) +
      control_error_delta_r * Y00 * get(distorted_normal_dot_unit_coord_vector);

  // Minimum of the comoving char speed on the excision boundary.
  const double min_comoving_char_speed = min(get(comoving_char_speed));

  // Difference between horizon and excision boundary.
  StrahlkorperGr::radial_distance(make_not_null(&radial_distance),
                                  apparent_horizon, excision_boundary);

  // Update zero-crossing predictors.
  predictor_char_speed->add(time,
                            get(characteristic_speed_on_excision_boundary));
  predictor_comoving_char_speed->add(time, get(comoving_char_speed));
  predictor_delta_radius->add(time, get(radial_distance));

  // Compute crossing times for state-change logic.
  const std::optional<double> char_speed_crossing_time =
      predictor_char_speed->min_positive_zero_crossing_time(time);
  const std::optional<double> comoving_char_speed_crossing_time =
      predictor_comoving_char_speed->min_positive_zero_crossing_time(time);
  const std::optional<double> delta_radius_crossing_time =
      predictor_delta_radius->min_positive_zero_crossing_time(time);

  // Update the info, possibly changing the state inside of info.
  std::string update_message = info->state->get_clone()->update(
      info,
      StateUpdateArgs{min_char_speed, min_comoving_char_speed,
                      control_error_delta_r},
      CrossingTimeInfo{char_speed_crossing_time,
                       comoving_char_speed_crossing_time,
                       delta_radius_crossing_time});

  const ControlErrorArgs control_error_args{
      min_char_speed, control_error_delta_r,
      avg_distorted_normal_dot_unit_coord_vector, dt_lambda_00};

  const double control_error =
      info->state->control_error(*info, control_error_args);

  return ErrorDiagnostics{
      control_error,
      info->state->number(),
      min(get(radial_distance)),
      min(get(radial_distance)) / apparent_horizon.average_radius(),
      min_comoving_char_speed,
      char_speed_crossing_time.value_or(0.0),
      comoving_char_speed_crossing_time.value_or(0.0),
      delta_radius_crossing_time.value_or(0.0),
      info->target_char_speed,
      info->suggested_time_scale.value_or(0.0),
      info->damping_time,
      control_error_args,
      std::move(update_message),
      info->discontinuous_change_has_occurred};
}
}  // namespace control_system::size

#define FRAME(data) BOOST_PP_TUPLE_ELEM(0, data)

#define INSTANTIATE(_, data)                                                   \
  template control_system::size::ErrorDiagnostics                              \
  control_system::size::control_error(                                         \
      const gsl::not_null<control_system::size::Info*> info,                   \
      const gsl::not_null<intrp::ZeroCrossingPredictor*> predictor_char_speed, \
      const gsl::not_null<intrp::ZeroCrossingPredictor*>                       \
          predictor_comoving_char_speed,                                       \
      const gsl::not_null<intrp::ZeroCrossingPredictor*>                       \
          predictor_delta_radius,                                              \
      double time, double control_error_delta_r, double dt_lambda_00,          \
      const Strahlkorper<FRAME(data)>& apparent_horizon,                       \
      const Strahlkorper<FRAME(data)>& excision_boundary,                      \
      const Scalar<DataVector>& lapse_on_excision_boundary,                    \
      const tnsr::I<DataVector, 3, FRAME(data)>&                               \
          frame_components_of_grid_shift,                                      \
      const tnsr::ii<DataVector, 3, FRAME(data)>&                              \
          spatial_metric_on_excision_boundary,                                 \
      const tnsr::II<DataVector, 3, FRAME(data)>&                              \
          inverse_spatial_metric_on_excision_boundary);

GENERATE_INSTANTIATIONS(INSTANTIATE, (::Frame::Distorted, ::Frame::Inertial))

#undef INSTANTIATE
#undef FRAME
