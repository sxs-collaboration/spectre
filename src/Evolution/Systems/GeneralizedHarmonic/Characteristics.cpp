// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Evolution/Systems/GeneralizedHarmonic/Characteristics.hpp"

#include <algorithm>
#include <array>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/EagerMath/DotProduct.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "DataStructures/Variables.hpp"  // IWYU pragma: keep
#include "PointwiseFunctions/GeneralRelativity/IndexManipulation.hpp"
#include "Utilities/GenerateInstantiations.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/MakeWithValue.hpp"

// IWYU pragma: no_forward_declare Tensor

namespace GeneralizedHarmonic {

template <size_t Dim, typename Frame>
void compute_characteristic_speeds(
    const gsl::not_null<typename Tags::CharacteristicSpeeds<Dim, Frame>::type*>
        char_speeds,
    const Scalar<DataVector>& gamma_1, const Scalar<DataVector>& lapse,
    const tnsr::I<DataVector, Dim, Frame>& shift,
    const tnsr::i<DataVector, Dim, Frame>& unit_normal_one_form) noexcept {
  const auto shift_dot_normal = get(dot_product(shift, unit_normal_one_form));
  (*char_speeds)[0] = -(1. + get(gamma_1)) * shift_dot_normal;  // v(UPsi)
  (*char_speeds)[1] = -shift_dot_normal;                        // v(UZero)
  (*char_speeds)[2] = -shift_dot_normal + get(lapse);           // v(UPlus)
  (*char_speeds)[3] = -shift_dot_normal - get(lapse);           // v(UMinus)
}

template <size_t Dim, typename Frame>
typename Tags::CharacteristicSpeeds<Dim, Frame>::type
CharacteristicSpeedsCompute<Dim, Frame>::function(
    const Scalar<DataVector>& gamma_1, const Scalar<DataVector>& lapse,
    const tnsr::I<DataVector, Dim, Frame>& shift,
    const tnsr::i<DataVector, Dim, Frame>& unit_normal_one_form) noexcept {
  auto char_speeds =
      make_with_value<typename Tags::CharacteristicSpeeds<Dim, Frame>::type>(
          get(lapse), 0.);
  compute_characteristic_speeds(make_not_null(&char_speeds), gamma_1, lapse,
                                shift, unit_normal_one_form);
  return char_speeds;
}

template <size_t Dim, typename Frame>
void compute_characteristic_fields(
    const gsl::not_null<typename Tags::CharacteristicFields<Dim, Frame>::type*>
        char_fields,
    const Scalar<DataVector>& gamma_2,
    const tnsr::II<DataVector, Dim, Frame>& inverse_spatial_metric,
    const tnsr::aa<DataVector, Dim, Frame>& spacetime_metric,
    const tnsr::aa<DataVector, Dim, Frame>& pi,
    const tnsr::iaa<DataVector, Dim, Frame>& phi,
    const tnsr::i<DataVector, Dim, Frame>& unit_normal_one_form) noexcept {
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
        get<Tags::UZero<Dim, Frame>>(*char_fields).get(i, a, b) =
            phi.get(i, a, b) -
            unit_normal_one_form.get(i) * phi_dot_normal.get(a, b);
      }
    }
  }

  // Eq.(32) of Lindblom+ (2005)
  get<Tags::UPsi<Dim, Frame>>(*char_fields) = spacetime_metric;

  for (size_t a = 0; a < Dim + 1; ++a) {
    for (size_t b = 0; b < a + 1; ++b) {
      // Eq.(33) of Lindblom+ (2005)
      get<Tags::UPlus<Dim, Frame>>(*char_fields).get(a, b) =
          pi.get(a, b) + phi_dot_normal.get(a, b) -
          get(gamma_2) * spacetime_metric.get(a, b);
      get<Tags::UMinus<Dim, Frame>>(*char_fields).get(a, b) =
          pi.get(a, b) - phi_dot_normal.get(a, b) -
          get(gamma_2) * spacetime_metric.get(a, b);
    }
  }
}

template <size_t Dim, typename Frame>
typename Tags::CharacteristicFields<Dim, Frame>::type
CharacteristicFieldsCompute<Dim, Frame>::function(
    const Scalar<DataVector>& gamma_2,
    const tnsr::II<DataVector, Dim, Frame>& inverse_spatial_metric,
    const tnsr::aa<DataVector, Dim, Frame>& spacetime_metric,
    const tnsr::aa<DataVector, Dim, Frame>& pi,
    const tnsr::iaa<DataVector, Dim, Frame>& phi,
    const tnsr::i<DataVector, Dim, Frame>& unit_normal_one_form) noexcept {
  auto char_fields =
      make_with_value<typename Tags::CharacteristicFields<Dim, Frame>::type>(
          get(gamma_2), 0.);
  compute_characteristic_fields(make_not_null(&char_fields), gamma_2,
                                inverse_spatial_metric, spacetime_metric, pi,
                                phi, unit_normal_one_form);
  return char_fields;
}

template <size_t Dim, typename Frame>
void compute_evolved_fields_from_characteristic_fields(
    const gsl::not_null<
        typename Tags::EvolvedFieldsFromCharacteristicFields<Dim, Frame>::type*>
        evolved_fields,
    const Scalar<DataVector>& gamma_2,
    const tnsr::aa<DataVector, Dim, Frame>& u_psi,
    const tnsr::iaa<DataVector, Dim, Frame>& u_zero,
    const tnsr::aa<DataVector, Dim, Frame>& u_plus,
    const tnsr::aa<DataVector, Dim, Frame>& u_minus,
    const tnsr::i<DataVector, Dim, Frame>& unit_normal_one_form) noexcept {
  // Invert Eq.(32) of Lindblom+ (2005) for Psi
  get<::gr::Tags::SpacetimeMetric<Dim, Frame, DataVector>>(*evolved_fields) =
      u_psi;

  for (size_t a = 0; a < Dim + 1; ++a) {
    for (size_t b = 0; b < a + 1; ++b) {
      // Invert Eq.(32) - (34) of Lindblom+ (2005) for Pi and Phi
      get<Tags::Pi<Dim, Frame>>(*evolved_fields).get(a, b) =
          0.5 * (u_plus.get(a, b) + u_minus.get(a, b)) +
          get(gamma_2) * u_psi.get(a, b);
      for (size_t i = 0; i < Dim; ++i) {
        get<Tags::Phi<Dim, Frame>>(*evolved_fields).get(i, a, b) =
            0.5 * (u_plus.get(a, b) - u_minus.get(a, b)) *
                unit_normal_one_form.get(i) +
            u_zero.get(i, a, b);
      }
    }
  }
}

template <size_t Dim, typename Frame>
typename Tags::EvolvedFieldsFromCharacteristicFields<Dim, Frame>::type
EvolvedFieldsFromCharacteristicFieldsCompute<Dim, Frame>::function(
    const Scalar<DataVector>& gamma_2,
    const tnsr::aa<DataVector, Dim, Frame>& u_psi,
    const tnsr::iaa<DataVector, Dim, Frame>& u_zero,
    const tnsr::aa<DataVector, Dim, Frame>& u_plus,
    const tnsr::aa<DataVector, Dim, Frame>& u_minus,
    const tnsr::i<DataVector, Dim, Frame>& unit_normal_one_form) noexcept {
  auto evolved_fields = make_with_value<
      typename Tags::EvolvedFieldsFromCharacteristicFields<Dim, Frame>::type>(
      get(gamma_2), 0.);
  compute_evolved_fields_from_characteristic_fields(
      make_not_null(&evolved_fields), gamma_2, u_psi, u_zero, u_plus, u_minus,
      unit_normal_one_form);
  return evolved_fields;
}

template <size_t Dim, typename Frame>
double ComputeLargestCharacteristicSpeed<Dim, Frame>::apply(
    const std::array<DataVector, 4>& char_speeds) noexcept {
  std::array<double, 4> max_speeds{
      {max(abs(char_speeds.at(0))), max(abs(char_speeds.at(1))),
       max(abs(char_speeds.at(2))), max(abs(char_speeds.at(3)))}};
  return *std::max_element(max_speeds.begin(), max_speeds.end());
}
}  // namespace GeneralizedHarmonic

#define DIM(data) BOOST_PP_TUPLE_ELEM(0, data)
#define FRAME(data) BOOST_PP_TUPLE_ELEM(1, data)

#define INSTANTIATION(_, data)                                              \
  template void GeneralizedHarmonic::compute_characteristic_speeds(         \
      const gsl::not_null<                                                  \
          typename GeneralizedHarmonic::Tags::CharacteristicSpeeds<         \
              DIM(data), FRAME(data)>::type*>                               \
          char_speeds,                                                      \
      const Scalar<DataVector>& gamma_1, const Scalar<DataVector>& lapse,   \
      const tnsr::I<DataVector, DIM(data), FRAME(data)>& shift,             \
      const tnsr::i<DataVector, DIM(data), FRAME(data)>&                    \
          unit_normal_one_form) noexcept;                                   \
  template struct GeneralizedHarmonic::CharacteristicSpeedsCompute<         \
      DIM(data), FRAME(data)>;                                              \
  template void GeneralizedHarmonic::compute_characteristic_fields(         \
      const gsl::not_null<                                                  \
          typename GeneralizedHarmonic::Tags::CharacteristicFields<         \
              DIM(data), FRAME(data)>::type*>                               \
          char_fields,                                                      \
      const Scalar<DataVector>& gamma_2,                                    \
      const tnsr::II<DataVector, DIM(data), FRAME(data)>&                   \
          inverse_spatial_metric,                                           \
      const tnsr::aa<DataVector, DIM(data), FRAME(data)>& spacetime_metric, \
      const tnsr::aa<DataVector, DIM(data), FRAME(data)>& pi,               \
      const tnsr::iaa<DataVector, DIM(data), FRAME(data)>& phi,             \
      const tnsr::i<DataVector, DIM(data), FRAME(data)>&                    \
          unit_normal_one_form) noexcept;                                   \
  template struct GeneralizedHarmonic::CharacteristicFieldsCompute<         \
      DIM(data), FRAME(data)>;                                              \
  template void                                                             \
  GeneralizedHarmonic::compute_evolved_fields_from_characteristic_fields(   \
      const gsl::not_null<typename GeneralizedHarmonic::Tags::              \
                              EvolvedFieldsFromCharacteristicFields<        \
                                  DIM(data), FRAME(data)>::type*>           \
          evolved_fields,                                                   \
      const Scalar<DataVector>& gamma_2,                                    \
      const tnsr::aa<DataVector, DIM(data), FRAME(data)>& u_psi,            \
      const tnsr::iaa<DataVector, DIM(data), FRAME(data)>& u_zero,          \
      const tnsr::aa<DataVector, DIM(data), FRAME(data)>& u_plus,           \
      const tnsr::aa<DataVector, DIM(data), FRAME(data)>& u_minus,          \
      const tnsr::i<DataVector, DIM(data), FRAME(data)>&                    \
          unit_normal_one_form) noexcept;                                   \
  template struct GeneralizedHarmonic::                                     \
      EvolvedFieldsFromCharacteristicFieldsCompute<DIM(data), FRAME(data)>; \
  template struct GeneralizedHarmonic::ComputeLargestCharacteristicSpeed<   \
      DIM(data), FRAME(data)>;

GENERATE_INSTANTIATIONS(INSTANTIATION, (1, 2, 3),
                        (Frame::Inertial, Frame::Grid))

#undef INSTANTIATION
#undef DIM
#undef FRAME
