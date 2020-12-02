// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Evolution/Systems/CurvedScalarWave/UpwindPenaltyCorrection.hpp"

#include <array>

#include "DataStructures/DataBox/PrefixHelpers.hpp"
#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/Tensor.hpp"  // IWYU pragma: keep
#include "DataStructures/Variables.hpp"      // IWYU pragma: keep
#include "Evolution/Systems/CurvedScalarWave/System.hpp"
#include "Utilities/GenerateInstantiations.hpp"
#include "Utilities/Math.hpp"
#include "Utilities/TMPL.hpp"  // IWYU pragma: keep

template <typename X, typename Symm, typename IndexList>
class Tensor;

namespace CurvedScalarWave {
namespace CurvedScalarWave_detail {
template <typename FieldTag>
typename FieldTag::type weight_char_field(
    const typename FieldTag::type& char_field_int,
    const DataVector& char_speed_int,
    const typename FieldTag::type& char_field_ext,
    const DataVector& char_speed_ext) noexcept {
  const DataVector& char_speed_avg{0.5 * (char_speed_int + char_speed_ext)};
  auto weighted_char_field = char_field_int;
  auto weighted_char_field_it = weighted_char_field.begin();
  for (auto int_it = char_field_int.begin(), ext_it = char_field_ext.begin();
       int_it != char_field_int.end();
       ++int_it, ++ext_it, ++weighted_char_field_it) {
    *weighted_char_field_it *= step_function(char_speed_avg) * char_speed_avg;
    *weighted_char_field_it +=
        step_function(-char_speed_avg) * char_speed_avg * *ext_it;
  }

  return weighted_char_field;
}

template <size_t Dim>
using char_field_tags =
    tmpl::list<Tags::VPsi, Tags::VZero<Dim>, Tags::VPlus, Tags::VMinus>;

template <size_t Dim>
Variables<char_field_tags<Dim>> weight_char_fields(
    const Variables<char_field_tags<Dim>>& char_fields_int,
    const std::array<DataVector, 4>& char_speeds_int,
    const Variables<char_field_tags<Dim>>& char_fields_ext,
    const std::array<DataVector, 4>& char_speeds_ext) noexcept {
  const auto& v_psi_int = get<Tags::VPsi>(char_fields_int);
  const auto& v_zero_int = get<Tags::VZero<Dim>>(char_fields_int);
  const auto& v_plus_int = get<Tags::VPlus>(char_fields_int);
  const auto& v_minus_int = get<Tags::VMinus>(char_fields_int);

  const DataVector& char_speed_v_psi_int{char_speeds_int[0]};
  const DataVector& char_speed_v_zero_int{char_speeds_int[1]};
  const DataVector& char_speed_v_plus_int{char_speeds_int[2]};
  const DataVector& char_speed_v_minus_int{char_speeds_int[3]};

  const auto& v_psi_ext = get<Tags::VPsi>(char_fields_ext);
  const auto& v_zero_ext = get<Tags::VZero<Dim>>(char_fields_ext);
  const auto& v_plus_ext = get<Tags::VPlus>(char_fields_ext);
  const auto& v_minus_ext = get<Tags::VMinus>(char_fields_ext);

  const DataVector& char_speed_v_psi_ext{char_speeds_ext[0]};
  const DataVector& char_speed_v_zero_ext{char_speeds_ext[1]};
  const DataVector& char_speed_v_plus_ext{char_speeds_ext[2]};
  const DataVector& char_speed_v_minus_ext{char_speeds_ext[3]};

  Variables<char_field_tags<Dim>> weighted_char_fields{
      char_speeds_int[0].size()};

  get<Tags::VPsi>(weighted_char_fields) = weight_char_field<Tags::VPsi>(
      v_psi_int, char_speed_v_psi_int, v_psi_ext, char_speed_v_psi_ext);
  get<Tags::VZero<Dim>>(weighted_char_fields) =
      weight_char_field<Tags::VZero<Dim>>(v_zero_int, char_speed_v_zero_int,
                                          v_zero_ext, char_speed_v_zero_ext);
  get<Tags::VPlus>(weighted_char_fields) = weight_char_field<Tags::VPlus>(
      v_plus_int, char_speed_v_plus_int, v_plus_ext, char_speed_v_plus_ext);
  get<Tags::VMinus>(weighted_char_fields) = weight_char_field<Tags::VMinus>(
      v_minus_int, char_speed_v_minus_int, v_minus_ext, char_speed_v_minus_ext);

  return weighted_char_fields;
}
}  // namespace CurvedScalarWave_detail

/// \cond
template <size_t Dim>
void UpwindPenaltyCorrection<Dim>::package_data(
    const gsl::not_null<Scalar<DataVector>*> packaged_pi,
    const gsl::not_null<tnsr::i<DataVector, Dim, Frame::Inertial>*>
        packaged_phi,
    const gsl::not_null<Scalar<DataVector>*> packaged_psi,
    const gsl::not_null<Scalar<DataVector>*> packaged_lapse,
    const gsl::not_null<tnsr::I<DataVector, Dim, Frame::Inertial>*>
        packaged_shift,
    const gsl::not_null<tnsr::II<DataVector, Dim, Frame::Inertial>*>
        packaged_inverse_spatial_metric,
    const gsl::not_null<Scalar<DataVector>*> packaged_gamma1,
    const gsl::not_null<Scalar<DataVector>*> packaged_gamma2,
    const gsl::not_null<tnsr::i<DataVector, Dim, Frame::Inertial>*>
        packaged_interface_unit_normal,
    const Scalar<DataVector>& pi,
    const tnsr::i<DataVector, Dim, Frame::Inertial>& phi,
    const Scalar<DataVector>& psi, const Scalar<DataVector>& lapse,
    const tnsr::I<DataVector, Dim, Frame::Inertial>& shift,
    const tnsr::II<DataVector, Dim, Frame::Inertial>& inverse_spatial_metric,
    const Scalar<DataVector>& gamma1, const Scalar<DataVector>& gamma2,
    const tnsr::i<DataVector, Dim, Frame::Inertial>& interface_unit_normal)
    const noexcept {
  *packaged_psi = psi;
  *packaged_pi = pi;
  *packaged_phi = phi;
  *packaged_lapse = lapse;
  *packaged_shift = shift;
  *packaged_inverse_spatial_metric = inverse_spatial_metric;
  *packaged_gamma1 = gamma1;
  *packaged_gamma2 = gamma2;
  *packaged_interface_unit_normal = interface_unit_normal;
}

template <size_t Dim>
void UpwindPenaltyCorrection<Dim>::operator()(
    const gsl::not_null<Scalar<DataVector>*> pi_normal_dot_numerical_flux,
    const gsl::not_null<tnsr::i<DataVector, Dim, Frame::Inertial>*>
        phi_normal_dot_numerical_flux,
    const gsl::not_null<Scalar<DataVector>*> psi_normal_dot_numerical_flux,
    const Scalar<DataVector>& pi_int,
    const tnsr::i<DataVector, Dim, Frame::Inertial>& phi_int,
    const Scalar<DataVector>& psi_int, const Scalar<DataVector>& lapse_int,
    const tnsr::I<DataVector, Dim, Frame::Inertial>& shift_int,
    const tnsr::II<DataVector, Dim, Frame::Inertial>&
        inverse_spatial_metric_int,
    const Scalar<DataVector>& gamma1_int, const Scalar<DataVector>& gamma2_int,
    const tnsr::i<DataVector, Dim, Frame::Inertial>& interface_unit_normal_int,
    const Scalar<DataVector>& pi_ext,
    const tnsr::i<DataVector, Dim, Frame::Inertial>& phi_ext,
    const Scalar<DataVector>& psi_ext, const Scalar<DataVector>& lapse_ext,
    const tnsr::I<DataVector, Dim, Frame::Inertial>& shift_ext,
    const tnsr::II<DataVector, Dim, Frame::Inertial>&
        inverse_spatial_metric_ext,
    const Scalar<DataVector>& gamma1_ext, const Scalar<DataVector>& gamma2_ext,
    const tnsr::i<DataVector, Dim,
                  Frame::Inertial>& /*interface_unit_normal_ext*/)
    const noexcept {
  const Scalar<DataVector> gamma1_avg{0.5 *
                                      (get(gamma1_int) + get(gamma1_ext))};
  const Scalar<DataVector> gamma2_avg{0.5 *
                                      (get(gamma2_int) + get(gamma2_ext))};

  const auto char_fields_int =
      characteristic_fields(gamma2_avg, inverse_spatial_metric_int, psi_int,
                            pi_int, phi_int, interface_unit_normal_int);
  const auto char_speeds_int = characteristic_speeds(
      gamma1_avg, lapse_int, shift_int, interface_unit_normal_int);
  const auto char_fields_ext =
      characteristic_fields(gamma2_avg, inverse_spatial_metric_ext, psi_ext,
                            pi_ext, phi_ext, interface_unit_normal_int);
  const auto char_speeds_ext = characteristic_speeds(
      gamma1_avg, lapse_ext, shift_ext, interface_unit_normal_int);

  const auto weighted_char_fields =
      CurvedScalarWave_detail::weight_char_fields<Dim>(
          char_fields_int, char_speeds_int, char_fields_ext, char_speeds_ext);

  const auto weighted_evolved_fields =
      evolved_fields_from_characteristic_fields(
          gamma2_avg, get<Tags::VPsi>(weighted_char_fields),
          get<Tags::VZero<Dim>>(weighted_char_fields),
          get<Tags::VPlus>(weighted_char_fields),
          get<Tags::VMinus>(weighted_char_fields), interface_unit_normal_int);

  *psi_normal_dot_numerical_flux = get<Psi>(weighted_evolved_fields);
  *pi_normal_dot_numerical_flux = get<Pi>(weighted_evolved_fields);
  *phi_normal_dot_numerical_flux = get<Phi<Dim>>(weighted_evolved_fields);
}
/// \endcond
}  // namespace CurvedScalarWave

// Generate explicit instantiations
#define DIM(data) BOOST_PP_TUPLE_ELEM(0, data)

#define INSTANTIATION(_, data) \
  template struct CurvedScalarWave::UpwindPenaltyCorrection<DIM(data)>;

GENERATE_INSTANTIATIONS(INSTANTIATION, (1, 2, 3))

#undef INSTANTIATION
#undef DIM
