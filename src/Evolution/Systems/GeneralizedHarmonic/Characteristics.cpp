// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Evolution/Systems/GeneralizedHarmonic/Characteristics.hpp"

#include <algorithm>
#include <array>
#include <optional>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/EagerMath/DotProduct.hpp"
#include "DataStructures/Tensor/EagerMath/Magnitude.hpp"
#include "DataStructures/Tensor/EagerMath/RaiseOrLowerIndex.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "DataStructures/Variables.hpp"
#include "Domain/TagsTimeDependent.hpp"
#include "Utilities/GenerateInstantiations.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/MakeWithValue.hpp"

namespace {
template <size_t Dim, typename SourceFrame, typename TargetFrame>
InverseJacobian<DataVector, Dim, SourceFrame, TargetFrame>
normalize_inverse_jacobian(
    const InverseJacobian<DataVector, Dim, SourceFrame, TargetFrame>&
        inverse_jacobian,
    const tnsr::II<DataVector, Dim, TargetFrame>& inverse_spatial_metric) {
  // Normalizing the first index wrt the spatial metric
  InverseJacobian<DataVector, Dim, SourceFrame, TargetFrame>
      scaled_inv_jacobian{get_size(get<0, 0>(inverse_jacobian))};
  DataVector magnitude{get_size(get<0, 0>(inverse_jacobian))};
  for (size_t i_hat = 0; i_hat < Dim; ++i_hat) {
    magnitude = 0.;
    for (size_t j = 0; j < Dim; ++j) {
      for (size_t k = 0; k < Dim; ++k) {
        magnitude += inverse_jacobian.get(i_hat, j) *
                     inverse_spatial_metric.get(j, k) *
                     inverse_jacobian.get(i_hat, k);
      }
    }
    ASSERT(min(magnitude) > 0,
           "Trying to normalize inverse jacobian with a negative magnitude: "
               << magnitude);
    magnitude = sqrt(magnitude);
    for (size_t j = 0; j < Dim; ++j) {
      scaled_inv_jacobian.get(i_hat, j) =
          inverse_jacobian.get(i_hat, j) / magnitude;
    }
  }
  return scaled_inv_jacobian;
}

template <size_t Dim, typename Frame, typename SourceFrame>
void v_zero_plus_minus_speed_impl(
    const gsl::not_null<tnsr::I<DataVector, Dim, SourceFrame>*> char_speed,
    const Scalar<DataVector>& lapse,
    const tnsr::I<DataVector, Dim, Frame>& shift,
    const InverseJacobian<DataVector, Dim, SourceFrame, Frame>&
        inverse_jacobian,
    const tnsr::II<DataVector, Dim, Frame>& inverse_spatial_metric,
    const std::optional<tnsr::I<DataVector, Dim, Frame>>& mesh_velocity,
    const int sign) {
  const auto normalized_inverse_jacobian =
      normalize_inverse_jacobian(inverse_jacobian, inverse_spatial_metric);
  for (size_t i = 0; i < Dim; ++i) {
    (*char_speed).get(i) = static_cast<double>(sign) * get(lapse) -
                           normalized_inverse_jacobian.get(i, 0) * shift.get(0);
    for (size_t j = 1; j < Dim; ++j) {
      (*char_speed).get(i) +=
          -normalized_inverse_jacobian.get(i, j) * shift.get(j);
    }
  }
  if (mesh_velocity.has_value()) {
    for (size_t i = 0; i < Dim; ++i) {
      for (size_t j = 0; j < Dim; ++j) {
        (*char_speed).get(i) -=
            normalized_inverse_jacobian.get(i, j) * (*mesh_velocity).get(j);
      }
    }
  }
}
}  // namespace

namespace gh {
template <size_t Dim, typename Frame, typename SourceFrame>
void vspacetimemetric_speed(
    const gsl::not_null<tnsr::I<DataVector, Dim, SourceFrame>*> char_speed,
    const Scalar<DataVector>& gamma_1,
    const tnsr::I<DataVector, Dim, Frame>& shift,
    const InverseJacobian<DataVector, Dim, SourceFrame, Frame>&
        inverse_jacobian,
    const tnsr::II<DataVector, Dim, Frame>& inverse_spatial_metric,
    const std::optional<tnsr::I<DataVector, Dim, Frame>>& mesh_velocity) {
  const auto normalized_inverse_jacobian =
      normalize_inverse_jacobian(inverse_jacobian, inverse_spatial_metric);
  for (size_t i = 0; i < Dim; ++i) {
    (*char_speed).get(i) = -(1. + get(gamma_1)) *
                           normalized_inverse_jacobian.get(i, 0) * shift.get(0);
    for (size_t j = 1; j < Dim; ++j) {
      (*char_speed).get(i) += -(1. + get(gamma_1)) *
                              normalized_inverse_jacobian.get(i, j) *
                              shift.get(j);
    }
  }
  if (mesh_velocity.has_value()) {
    for (size_t i = 0; i < Dim; ++i) {
      for (size_t j = 0; j < Dim; ++j) {
        (*char_speed).get(i) -= (1. + get(gamma_1)) *
                                normalized_inverse_jacobian.get(i, j) *
                                (*mesh_velocity).get(j);
      }
    }
  }
}

template <size_t Dim, typename Frame, typename SourceFrame>
tnsr::I<DataVector, Dim, SourceFrame> vspacetimemetric_speed(
    const Scalar<DataVector>& gamma_1,
    const tnsr::I<DataVector, Dim, Frame>& shift,
    const InverseJacobian<DataVector, Dim, SourceFrame, Frame>&
        inverse_jacobian,
    const tnsr::II<DataVector, Dim, Frame>& inverse_spatial_metric,
    const std::optional<tnsr::I<DataVector, Dim, Frame>>& mesh_velocity) {
  auto char_speed =
      make_with_value<tnsr::I<DataVector, Dim, SourceFrame>>(get<0>(shift), 0.);
  vspacetimemetric_speed(make_not_null(&char_speed), gamma_1, shift,
                         inverse_jacobian, inverse_spatial_metric,
                         mesh_velocity);
  return char_speed;
}

template <size_t Dim, typename Frame, typename SourceFrame>
void vzero_speed(
    const gsl::not_null<tnsr::I<DataVector, Dim, SourceFrame>*> char_speed,
    const Scalar<DataVector>& lapse,
    const tnsr::I<DataVector, Dim, Frame>& shift,
    const InverseJacobian<DataVector, Dim, SourceFrame, Frame>&
        inverse_jacobian,
    const tnsr::II<DataVector, Dim, Frame>& inverse_spatial_metric,
    const std::optional<tnsr::I<DataVector, Dim, Frame>>& mesh_velocity) {
  v_zero_plus_minus_speed_impl(char_speed, lapse, shift, inverse_jacobian,
                               inverse_spatial_metric, mesh_velocity, 0);
}

template <size_t Dim, typename Frame, typename SourceFrame>
tnsr::I<DataVector, Dim, SourceFrame> vzero_speed(
    const Scalar<DataVector>& lapse,
    const tnsr::I<DataVector, Dim, Frame>& shift,
    const InverseJacobian<DataVector, Dim, SourceFrame, Frame>&
        inverse_jacobian,
    const tnsr::II<DataVector, Dim, Frame>& inverse_spatial_metric,
    const std::optional<tnsr::I<DataVector, Dim, Frame>>& mesh_velocity) {
  auto char_speed =
      make_with_value<tnsr::I<DataVector, Dim, SourceFrame>>(get<0>(shift), 0.);
  vzero_speed(make_not_null(&char_speed), lapse, shift, inverse_jacobian,
              inverse_spatial_metric, mesh_velocity);
  return char_speed;
}

template <size_t Dim, typename Frame, typename SourceFrame>
void vplus_speed(
    const gsl::not_null<tnsr::I<DataVector, Dim, SourceFrame>*> char_speed,
    const Scalar<DataVector>& lapse,
    const tnsr::I<DataVector, Dim, Frame>& shift,
    const InverseJacobian<DataVector, Dim, SourceFrame, Frame>&
        inverse_jacobian,
    const tnsr::II<DataVector, Dim, Frame>& inverse_spatial_metric,
    const std::optional<tnsr::I<DataVector, Dim, Frame>>& mesh_velocity) {
  v_zero_plus_minus_speed_impl(char_speed, lapse, shift, inverse_jacobian,
                               inverse_spatial_metric, mesh_velocity, 1);
}

template <size_t Dim, typename Frame, typename SourceFrame>
tnsr::I<DataVector, Dim, SourceFrame> vplus_speed(
    const Scalar<DataVector>& lapse,
    const tnsr::I<DataVector, Dim, Frame>& shift,
    const InverseJacobian<DataVector, Dim, SourceFrame, Frame>&
        inverse_jacobian,
    const tnsr::II<DataVector, Dim, Frame>& inverse_spatial_metric,
    const std::optional<tnsr::I<DataVector, Dim, Frame>>& mesh_velocity) {
  auto char_speed =
      make_with_value<tnsr::I<DataVector, Dim, SourceFrame>>(get<0>(shift), 0.);
  vplus_speed(make_not_null(&char_speed), lapse, shift, inverse_jacobian,
              inverse_spatial_metric, mesh_velocity);
  return char_speed;
}

template <size_t Dim, typename Frame, typename SourceFrame>
void vminus_speed(
    const gsl::not_null<tnsr::I<DataVector, Dim, SourceFrame>*> char_speed,
    const Scalar<DataVector>& lapse,
    const tnsr::I<DataVector, Dim, Frame>& shift,
    const InverseJacobian<DataVector, Dim, SourceFrame, Frame>&
        inverse_jacobian,
    const tnsr::II<DataVector, Dim, Frame>& inverse_spatial_metric,
    const std::optional<tnsr::I<DataVector, Dim, Frame>>& mesh_velocity) {
      v_zero_plus_minus_speed_impl(char_speed, lapse, shift, inverse_jacobian,
                                   inverse_spatial_metric, mesh_velocity, -1);
}

template <size_t Dim, typename Frame, typename SourceFrame>
tnsr::I<DataVector, Dim, SourceFrame> vminus_speed(
    const Scalar<DataVector>& lapse,
    const tnsr::I<DataVector, Dim, Frame>& shift,
    const InverseJacobian<DataVector, Dim, SourceFrame, Frame>&
        inverse_jacobian,
    const tnsr::II<DataVector, Dim, Frame>& inverse_spatial_metric,
    const std::optional<tnsr::I<DataVector, Dim, Frame>>& mesh_velocity) {
  auto char_speed =
      make_with_value<tnsr::I<DataVector, Dim, SourceFrame>>(get<0>(shift), 0.);
  vminus_speed(make_not_null(&char_speed), lapse, shift, inverse_jacobian,
               inverse_spatial_metric, mesh_velocity);
  return char_speed;
}

template <size_t Dim, typename Frame>
void characteristic_speeds(
    const gsl::not_null<std::array<DataVector, 4>*> char_speeds,
    const Scalar<DataVector>& gamma_1, const Scalar<DataVector>& lapse,
    const tnsr::I<DataVector, Dim, Frame>& shift,
    const tnsr::i<DataVector, Dim, Frame>& unit_normal_one_form,
    const std::optional<tnsr::I<DataVector, Dim, Frame>>& mesh_velocity) {
  auto shift_dot_normal = dot_product(shift, unit_normal_one_form);
  (*char_speeds)[0] =
      -(1. + get(gamma_1)) * get(shift_dot_normal);  // lambda(VSpacetimeMetric)
  (*char_speeds)[1] = -get(shift_dot_normal);        // lambda(VZero)
  (*char_speeds)[2] = -get(shift_dot_normal) + get(lapse);  // lambda(VPlus)
  (*char_speeds)[3] = -get(shift_dot_normal) - get(lapse);  // lambda(VMinus)
  if (mesh_velocity.has_value()) {
    dot_product(make_not_null(&shift_dot_normal), *mesh_velocity,
                unit_normal_one_form);
    (*char_speeds)[0] -= (1. + get(gamma_1)) * get(shift_dot_normal);
    (*char_speeds)[1] -= get(shift_dot_normal);
    (*char_speeds)[2] -= get(shift_dot_normal);
    (*char_speeds)[3] -= get(shift_dot_normal);
  }
}

template <size_t Dim, typename Frame>
std::array<DataVector, 4> characteristic_speeds(
    const Scalar<DataVector>& gamma_1, const Scalar<DataVector>& lapse,
    const tnsr::I<DataVector, Dim, Frame>& shift,
    const tnsr::i<DataVector, Dim, Frame>& unit_normal_one_form,
    const std::optional<tnsr::I<DataVector, Dim, Frame>>& mesh_velocity) {
  auto char_speeds = make_with_value<
      typename Tags::CharacteristicSpeeds<DataVector, Dim, Frame>::type>(
      get(lapse), 0.);
  characteristic_speeds(make_not_null(&char_speeds), gamma_1, lapse, shift,
                        unit_normal_one_form, mesh_velocity);
  return char_speeds;
}

template <size_t Dim, typename Frame>
void characteristic_fields(
    const gsl::not_null<
        typename Tags::CharacteristicFields<DataVector, Dim, Frame>::type*>
        char_fields,
    const Scalar<DataVector>& gamma_2,
    const tnsr::II<DataVector, Dim, Frame>& inverse_spatial_metric,
    const tnsr::aa<DataVector, Dim, Frame>& spacetime_metric,
    const tnsr::aa<DataVector, Dim, Frame>& pi,
    const tnsr::iaa<DataVector, Dim, Frame>& phi,
    const tnsr::i<DataVector, Dim, Frame>& unit_normal_one_form) {
  const auto number_of_grid_points = get(gamma_2).size();
  if (UNLIKELY(number_of_grid_points != char_fields->number_of_grid_points())) {
    char_fields->initialize(number_of_grid_points);
  }
  auto phi_dot_normal =
      make_with_value<tnsr::aa<DataVector, Dim, Frame>>(pi, 0.);
  auto unit_normal_vector =
      raise_or_lower_index(unit_normal_one_form, inverse_spatial_metric);

  // Compute phi_dot_normal_{ab} = n^i \Phi_{iab}
  for (size_t a = 0; a < Dim + 1; ++a) {
    for (size_t b = 0; b < a + 1; ++b) {
      for (size_t i = 0; i < Dim; ++i) {
        phi_dot_normal.get(a, b) +=
            unit_normal_vector.get(i) * phi.get(i, a, b);
      }
    }
  }

  // Eq.(34) of Lindblom+ (2005)
  for (size_t i = 0; i < Dim; ++i) {
    for (size_t a = 0; a < Dim + 1; ++a) {
      for (size_t b = 0; b < a + 1; ++b) {
        get<Tags::VZero<DataVector, Dim, Frame>>(*char_fields).get(i, a, b) =
            phi.get(i, a, b) -
            unit_normal_one_form.get(i) * phi_dot_normal.get(a, b);
      }
    }
  }

  // Eq.(32) of Lindblom+ (2005)
  get<Tags::VSpacetimeMetric<DataVector, Dim, Frame>>(*char_fields) =
      spacetime_metric;

  for (size_t a = 0; a < Dim + 1; ++a) {
    for (size_t b = 0; b < a + 1; ++b) {
      // Eq.(33) of Lindblom+ (2005)
      get<Tags::VPlus<DataVector, Dim, Frame>>(*char_fields).get(a, b) =
          pi.get(a, b) + phi_dot_normal.get(a, b) -
          get(gamma_2) * spacetime_metric.get(a, b);
      get<Tags::VMinus<DataVector, Dim, Frame>>(*char_fields).get(a, b) =
          pi.get(a, b) - phi_dot_normal.get(a, b) -
          get(gamma_2) * spacetime_metric.get(a, b);
    }
  }
}

template <size_t Dim, typename Frame>
typename Tags::CharacteristicFields<DataVector, Dim, Frame>::type
characteristic_fields(
    const Scalar<DataVector>& gamma_2,
    const tnsr::II<DataVector, Dim, Frame>& inverse_spatial_metric,
    const tnsr::aa<DataVector, Dim, Frame>& spacetime_metric,
    const tnsr::aa<DataVector, Dim, Frame>& pi,
    const tnsr::iaa<DataVector, Dim, Frame>& phi,
    const tnsr::i<DataVector, Dim, Frame>& unit_normal_one_form) {
  auto char_fields = make_with_value<
      typename Tags::CharacteristicFields<DataVector, Dim, Frame>::type>(
      get(gamma_2), 0.);
  characteristic_fields(make_not_null(&char_fields), gamma_2,
                        inverse_spatial_metric, spacetime_metric, pi, phi,
                        unit_normal_one_form);
  return char_fields;
}

template <size_t Dim, typename Frame>
void evolved_fields_from_characteristic_fields(
    const gsl::not_null<typename Tags::EvolvedFieldsFromCharacteristicFields<
        DataVector, Dim, Frame>::type*>
        evolved_fields,
    const Scalar<DataVector>& gamma_2,
    const tnsr::aa<DataVector, Dim, Frame>& u_psi,
    const tnsr::iaa<DataVector, Dim, Frame>& u_zero,
    const tnsr::aa<DataVector, Dim, Frame>& u_plus,
    const tnsr::aa<DataVector, Dim, Frame>& u_minus,
    const tnsr::i<DataVector, Dim, Frame>& unit_normal_one_form) {
  const auto number_of_grid_points = get(gamma_2).size();
  if (UNLIKELY(number_of_grid_points !=
               evolved_fields->number_of_grid_points())) {
    evolved_fields->initialize(number_of_grid_points);
  }
  // Invert Eq.(32) of Lindblom+ (2005) for Psi
  get<::gr::Tags::SpacetimeMetric<DataVector, Dim, Frame>>(*evolved_fields) =
      u_psi;

  for (size_t a = 0; a < Dim + 1; ++a) {
    for (size_t b = 0; b < a + 1; ++b) {
      // Invert Eq.(32) - (34) of Lindblom+ (2005) for Pi and Phi
      get<Tags::Pi<DataVector, Dim, Frame>>(*evolved_fields).get(a, b) =
          0.5 * (u_plus.get(a, b) + u_minus.get(a, b)) +
          get(gamma_2) * u_psi.get(a, b);
      for (size_t i = 0; i < Dim; ++i) {
        get<Tags::Phi<DataVector, Dim, Frame>>(*evolved_fields).get(i, a, b) =
            0.5 * (u_plus.get(a, b) - u_minus.get(a, b)) *
                unit_normal_one_form.get(i) +
            u_zero.get(i, a, b);
      }
    }
  }
}

template <size_t Dim, typename Frame>
typename Tags::EvolvedFieldsFromCharacteristicFields<DataVector, Dim,
                                                     Frame>::type
evolved_fields_from_characteristic_fields(
    const Scalar<DataVector>& gamma_2,
    const tnsr::aa<DataVector, Dim, Frame>& u_psi,
    const tnsr::iaa<DataVector, Dim, Frame>& u_zero,
    const tnsr::aa<DataVector, Dim, Frame>& u_plus,
    const tnsr::aa<DataVector, Dim, Frame>& u_minus,
    const tnsr::i<DataVector, Dim, Frame>& unit_normal_one_form) {
  auto evolved_fields =
      make_with_value<typename Tags::EvolvedFieldsFromCharacteristicFields<
          DataVector, Dim, Frame>::type>(get(gamma_2), 0.);
  evolved_fields_from_characteristic_fields(make_not_null(&evolved_fields),
                                            gamma_2, u_psi, u_zero, u_plus,
                                            u_minus, unit_normal_one_form);
  return evolved_fields;
}

template <size_t Dim, typename Frame>
void Tags::ComputeLargestCharacteristicSpeed<Dim, Frame>::function(
    const gsl::not_null<double*> speed, const Scalar<DataVector>& gamma_1,
    const Scalar<DataVector>& lapse,
    const tnsr::I<DataVector, Dim, Frame>& shift,
    const tnsr::ii<DataVector, Dim, Frame>& spatial_metric) {
  const auto shift_magnitude = magnitude(shift, spatial_metric);
  *speed = std::max(max(abs(1. + get(gamma_1)) * get(shift_magnitude)),
                    max(get(shift_magnitude) + get(lapse)));
}
}  // namespace gh

#define DIM(data) BOOST_PP_TUPLE_ELEM(0, data)
#define FRAME(data) BOOST_PP_TUPLE_ELEM(1, data)

#define INSTANTIATION(_, data)                                                 \
  template void gh::characteristic_speeds(                                     \
      const gsl::not_null<std::array<DataVector, 4>*> char_speeds,             \
      const Scalar<DataVector>& gamma_1, const Scalar<DataVector>& lapse,      \
      const tnsr::I<DataVector, DIM(data), FRAME(data)>& shift,                \
      const tnsr::i<DataVector, DIM(data), FRAME(data)>& unit_normal_one_form, \
      const std::optional<tnsr::I<DataVector, DIM(data), FRAME(data)>>&        \
          mesh_velocity);                                                      \
  template std::array<DataVector, 4> gh::characteristic_speeds(                \
      const Scalar<DataVector>& gamma_1, const Scalar<DataVector>& lapse,      \
      const tnsr::I<DataVector, DIM(data), FRAME(data)>& shift,                \
      const tnsr::i<DataVector, DIM(data), FRAME(data)>& unit_normal_one_form, \
      const std::optional<tnsr::I<DataVector, DIM(data), FRAME(data)>>&        \
          mesh_velocity);                                                      \
  template struct gh::CharacteristicSpeedsCompute<DIM(data), FRAME(data)>;     \
  template struct gh::CharacteristicSpeedsOnStrahlkorperCompute<DIM(data),     \
                                                                FRAME(data)>;  \
  template void gh::characteristic_fields(                                     \
      const gsl::not_null<typename gh::Tags::CharacteristicFields<             \
          DataVector, DIM(data), FRAME(data)>::type*>                          \
          char_fields,                                                         \
      const Scalar<DataVector>& gamma_2,                                       \
      const tnsr::II<DataVector, DIM(data), FRAME(data)>&                      \
          inverse_spatial_metric,                                              \
      const tnsr::aa<DataVector, DIM(data), FRAME(data)>& spacetime_metric,    \
      const tnsr::aa<DataVector, DIM(data), FRAME(data)>& pi,                  \
      const tnsr::iaa<DataVector, DIM(data), FRAME(data)>& phi,                \
      const tnsr::i<DataVector, DIM(data), FRAME(data)>&                       \
          unit_normal_one_form);                                               \
  template typename gh::Tags::CharacteristicFields<DataVector, DIM(data),      \
                                                   FRAME(data)>::type          \
  gh::characteristic_fields(                                                   \
      const Scalar<DataVector>& gamma_2,                                       \
      const tnsr::II<DataVector, DIM(data), FRAME(data)>&                      \
          inverse_spatial_metric,                                              \
      const tnsr::aa<DataVector, DIM(data), FRAME(data)>& spacetime_metric,    \
      const tnsr::aa<DataVector, DIM(data), FRAME(data)>& pi,                  \
      const tnsr::iaa<DataVector, DIM(data), FRAME(data)>& phi,                \
      const tnsr::i<DataVector, DIM(data), FRAME(data)>&                       \
          unit_normal_one_form);                                               \
  template struct gh::CharacteristicFieldsCompute<DIM(data), FRAME(data)>;     \
  template void gh::evolved_fields_from_characteristic_fields(                 \
      const gsl::not_null<                                                     \
          typename gh::Tags::EvolvedFieldsFromCharacteristicFields<            \
              DataVector, DIM(data), FRAME(data)>::type*>                      \
          evolved_fields,                                                      \
      const Scalar<DataVector>& gamma_2,                                       \
      const tnsr::aa<DataVector, DIM(data), FRAME(data)>& u_psi,               \
      const tnsr::iaa<DataVector, DIM(data), FRAME(data)>& u_zero,             \
      const tnsr::aa<DataVector, DIM(data), FRAME(data)>& u_plus,              \
      const tnsr::aa<DataVector, DIM(data), FRAME(data)>& u_minus,             \
      const tnsr::i<DataVector, DIM(data), FRAME(data)>&                       \
          unit_normal_one_form);                                               \
  template typename gh::Tags::EvolvedFieldsFromCharacteristicFields<           \
      DataVector, DIM(data), FRAME(data)>::type                                \
  gh::evolved_fields_from_characteristic_fields(                               \
      const Scalar<DataVector>& gamma_2,                                       \
      const tnsr::aa<DataVector, DIM(data), FRAME(data)>& u_psi,               \
      const tnsr::iaa<DataVector, DIM(data), FRAME(data)>& u_zero,             \
      const tnsr::aa<DataVector, DIM(data), FRAME(data)>& u_plus,              \
      const tnsr::aa<DataVector, DIM(data), FRAME(data)>& u_minus,             \
      const tnsr::i<DataVector, DIM(data), FRAME(data)>&                       \
          unit_normal_one_form);                                               \
  template struct gh::EvolvedFieldsFromCharacteristicFieldsCompute<            \
      DIM(data), FRAME(data)>;                                                 \
  template struct gh::Tags::ComputeLargestCharacteristicSpeed<DIM(data),       \
                                                              FRAME(data)>;    \
  template void gh::vspacetimemetric_speed(                                    \
      const gsl::not_null<                                                     \
          tnsr::I<DataVector, DIM(data), Frame::ElementLogical>*>              \
          char_speed,                                                          \
      const Scalar<DataVector>& gamma_1,                                       \
      const tnsr::I<DataVector, DIM(data), FRAME(data)>& shift,                \
      const InverseJacobian<DataVector, DIM(data), Frame::ElementLogical,      \
                            FRAME(data)>& inverse_jacobian,                    \
      const tnsr::II<DataVector, DIM(data), FRAME(data)>&                      \
          inverse_spatial_metric,                                              \
      const std::optional<tnsr::I<DataVector, DIM(data), FRAME(data)>>&        \
          mesh_velocity);                                                      \
  template void gh::vzero_speed(                                               \
      const gsl::not_null<                                                     \
          tnsr::I<DataVector, DIM(data), Frame::ElementLogical>*>              \
          char_speed,                                                          \
      const Scalar<DataVector>& lapse,                                         \
      const tnsr::I<DataVector, DIM(data), FRAME(data)>& shift,                \
      const InverseJacobian<DataVector, DIM(data), Frame::ElementLogical,      \
                            FRAME(data)>& inverse_jacobian,                    \
      const tnsr::II<DataVector, DIM(data), FRAME(data)>&                      \
          inverse_spatial_metric,                                              \
      const std::optional<tnsr::I<DataVector, DIM(data), FRAME(data)>>&        \
          mesh_velocity);                                                      \
  template void gh::vplus_speed(                                               \
      const gsl::not_null<                                                     \
          tnsr::I<DataVector, DIM(data), Frame::ElementLogical>*>              \
          char_speed,                                                          \
      const Scalar<DataVector>& lapse,                                         \
      const tnsr::I<DataVector, DIM(data), FRAME(data)>& shift,                \
      const InverseJacobian<DataVector, DIM(data), Frame::ElementLogical,      \
                            FRAME(data)>& inverse_jacobian,                    \
      const tnsr::II<DataVector, DIM(data), FRAME(data)>&                      \
          inverse_spatial_metric,                                              \
      const std::optional<tnsr::I<DataVector, DIM(data), FRAME(data)>>&        \
          mesh_velocity);                                                      \
  template void gh::vminus_speed(                                              \
      const gsl::not_null<                                                     \
          tnsr::I<DataVector, DIM(data), Frame::ElementLogical>*>              \
          char_speed,                                                          \
      const Scalar<DataVector>& lapse,                                         \
      const tnsr::I<DataVector, DIM(data), FRAME(data)>& shift,                \
      const InverseJacobian<DataVector, DIM(data), Frame::ElementLogical,      \
                            FRAME(data)>& inverse_jacobian,                    \
      const tnsr::II<DataVector, DIM(data), FRAME(data)>&                      \
          inverse_spatial_metric,                                              \
      const std::optional<tnsr::I<DataVector, DIM(data), FRAME(data)>>&        \
          mesh_velocity);                                                      \
                                                                               \
  template tnsr::I<DataVector, DIM(data), Frame::ElementLogical>               \
  gh::vspacetimemetric_speed(                                                  \
      const Scalar<DataVector>& gamma_1,                                       \
      const tnsr::I<DataVector, DIM(data), FRAME(data)>& shift,                \
      const InverseJacobian<DataVector, DIM(data), Frame::ElementLogical,      \
                            FRAME(data)>& inverse_jacobian,                    \
      const tnsr::II<DataVector, DIM(data), FRAME(data)>&                      \
          inverse_spatial_metric,                                              \
      const std::optional<tnsr::I<DataVector, DIM(data), FRAME(data)>>&        \
          mesh_velocity);                                                      \
  template tnsr::I<DataVector, DIM(data), Frame::ElementLogical>               \
  gh::vzero_speed(                                                             \
      const Scalar<DataVector>& lapse,                                         \
      const tnsr::I<DataVector, DIM(data), FRAME(data)>& shift,                \
      const InverseJacobian<DataVector, DIM(data), Frame::ElementLogical,      \
                            FRAME(data)>& inverse_jacobian,                    \
      const tnsr::II<DataVector, DIM(data), FRAME(data)>&                      \
          inverse_spatial_metric,                                              \
      const std::optional<tnsr::I<DataVector, DIM(data), FRAME(data)>>&        \
          mesh_velocity);                                                      \
  template tnsr::I<DataVector, DIM(data), Frame::ElementLogical>               \
  gh::vplus_speed(                                                             \
      const Scalar<DataVector>& lapse,                                         \
      const tnsr::I<DataVector, DIM(data), FRAME(data)>& shift,                \
      const InverseJacobian<DataVector, DIM(data), Frame::ElementLogical,      \
                            FRAME(data)>& inverse_jacobian,                    \
      const tnsr::II<DataVector, DIM(data), FRAME(data)>&                      \
          inverse_spatial_metric,                                              \
      const std::optional<tnsr::I<DataVector, DIM(data), FRAME(data)>>&        \
          mesh_velocity);                                                      \
  template tnsr::I<DataVector, DIM(data), Frame::ElementLogical>               \
  gh::vminus_speed(                                                            \
      const Scalar<DataVector>& lapse,                                         \
      const tnsr::I<DataVector, DIM(data), FRAME(data)>& shift,                \
      const InverseJacobian<DataVector, DIM(data), Frame::ElementLogical,      \
                            FRAME(data)>& inverse_jacobian,                    \
      const tnsr::II<DataVector, DIM(data), FRAME(data)>&                      \
          inverse_spatial_metric,                                              \
      const std::optional<tnsr::I<DataVector, DIM(data), FRAME(data)>>&        \
          mesh_velocity);

GENERATE_INSTANTIATIONS(INSTANTIATION, (1, 2, 3),
                        (Frame::Inertial, Frame::Grid))

#undef INSTANTIATION
#undef DIM
#undef FRAME
