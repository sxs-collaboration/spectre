// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Evolution/Systems/CurvedScalarWave/BoundaryCorrections/UpwindPenalty.hpp"

#include <memory>
#include <optional>
#include <pup.h>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tags/TempTensor.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "NumericalAlgorithms/DiscontinuousGalerkin/Formulation.hpp"
#include "NumericalAlgorithms/DiscontinuousGalerkin/NormalDotFlux.hpp"
#include "Utilities/ContainerHelpers.hpp"
#include "Utilities/GenerateInstantiations.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/TMPL.hpp"  // IWYU pragma: keep

namespace CurvedScalarWave::BoundaryCorrections::CurvedScalarWave_detail {
template <typename FieldTag>
typename FieldTag::type weight_char_field(
    const typename FieldTag::type& char_field, const DataVector& char_speed,
    const double sign) noexcept {
  auto weighted_char_field = char_field;
  auto weighted_char_field_it = weighted_char_field.begin();

  // pass sign = -1 for weighting internal fields, +1 for external fields
  for (auto int_it = char_field.begin(); int_it != char_field.end();
       ++int_it, ++weighted_char_field_it) {
    *weighted_char_field_it *=
        (-sign * step_function(sign * char_speed) * char_speed);
  }

  return weighted_char_field;
}

template <size_t Dim>
using char_field_tags =
    tmpl::list<Tags::VPsi, Tags::VZero<Dim>, Tags::VPlus, Tags::VMinus>;

template <size_t Dim>
void weight_char_fields(
    const gsl::not_null<Variables<char_field_tags<Dim>>*>
        weighted_char_fields_int,
    const gsl::not_null<Variables<char_field_tags<Dim>>*>
        weighted_char_fields_ext,
    const Scalar<DataVector>& v_psi_int,
    const tnsr::i<DataVector, Dim, Frame::Inertial>& v_zero_int,
    const Scalar<DataVector>& v_plus_int, const Scalar<DataVector>& v_minus_int,
    const tnsr::a<DataVector, 3, Frame::Inertial>& char_speeds_int,
    const Scalar<DataVector>& v_psi_ext,
    const tnsr::i<DataVector, Dim, Frame::Inertial>& v_zero_ext,
    const Scalar<DataVector>& v_plus_ext, const Scalar<DataVector>& v_minus_ext,
    const tnsr::a<DataVector, 3, Frame::Inertial>& char_speeds_ext) noexcept {
  get<Tags::VPsi>(*weighted_char_fields_int) =
      weight_char_field<Tags::VPsi>(v_psi_int, get<0>(char_speeds_int), -1.);
  get<Tags::VZero<Dim>>(*weighted_char_fields_int) =
      weight_char_field<Tags::VZero<Dim>>(v_zero_int, get<1>(char_speeds_int),
                                          -1.);
  get<Tags::VPlus>(*weighted_char_fields_int) =
      weight_char_field<Tags::VPlus>(v_plus_int, get<2>(char_speeds_int), -1.);
  get<Tags::VMinus>(*weighted_char_fields_int) =
      weight_char_field<Tags::VMinus>(v_minus_int, get<3>(char_speeds_int),
                                      -1.);

  get<Tags::VPsi>(*weighted_char_fields_ext) =
      weight_char_field<Tags::VPsi>(v_psi_ext, get<0>(char_speeds_ext), 1.);
  get<Tags::VZero<Dim>>(*weighted_char_fields_ext) =
      weight_char_field<Tags::VZero<Dim>>(v_zero_ext, get<1>(char_speeds_ext),
                                          1.);
  get<Tags::VPlus>(*weighted_char_fields_ext) =
      weight_char_field<Tags::VPlus>(v_plus_ext, get<2>(char_speeds_ext), 1.);
  get<Tags::VMinus>(*weighted_char_fields_ext) =
      weight_char_field<Tags::VMinus>(v_minus_ext, get<3>(char_speeds_ext), 1.);
}
}  // namespace CurvedScalarWave::BoundaryCorrections::CurvedScalarWave_detail

namespace CurvedScalarWave::BoundaryCorrections {
template <size_t Dim>
UpwindPenalty<Dim>::UpwindPenalty(CkMigrateMessage* msg) noexcept
    : BoundaryCorrection<Dim>(msg) {}

template <size_t Dim>
std::unique_ptr<BoundaryCorrection<Dim>> UpwindPenalty<Dim>::get_clone()
    const noexcept {
  return std::make_unique<UpwindPenalty>(*this);
}

template <size_t Dim>
void UpwindPenalty<Dim>::pup(PUP::er& p) {
  BoundaryCorrection<Dim>::pup(p);
}

template <size_t Dim>
double UpwindPenalty<Dim>::dg_package_data(
    const gsl::not_null<Scalar<DataVector>*> packaged_v_psi,
    const gsl::not_null<tnsr::i<DataVector, Dim, Frame::Inertial>*>
        packaged_v_zero,
    const gsl::not_null<Scalar<DataVector>*> packaged_v_plus,
    const gsl::not_null<Scalar<DataVector>*> packaged_v_minus,
    const gsl::not_null<Scalar<DataVector>*> packaged_gamma2,
    const gsl::not_null<tnsr::i<DataVector, Dim, Frame::Inertial>*>
        packaged_interface_unit_normal,
    const gsl::not_null<tnsr::a<DataVector, 3, Frame::Inertial>*>
        packaged_char_speeds,

    const Scalar<DataVector>& pi,
    const tnsr::i<DataVector, Dim, Frame::Inertial>& phi,
    const Scalar<DataVector>& psi,

    const Scalar<DataVector>& lapse,
    const tnsr::I<DataVector, Dim, Frame::Inertial>& shift,
    const tnsr::II<DataVector, Dim, Frame::Inertial>& inverse_spatial_metric,
    const Scalar<DataVector>& constraint_gamma1,
    const Scalar<DataVector>& constraint_gamma2,

    const tnsr::i<DataVector, Dim, Frame::Inertial>& interface_unit_normal,
    const std::optional<tnsr::I<DataVector, Dim, Frame::Inertial>>&
    /*mesh_velocity*/,
    const std::optional<Scalar<DataVector>>& normal_dot_mesh_velocity)
    const noexcept {
  *packaged_gamma2 = constraint_gamma2;
  *packaged_interface_unit_normal = interface_unit_normal;

  {  // package characteristic fields
    const auto char_fields =
        characteristic_fields(constraint_gamma2, inverse_spatial_metric, psi,
                              pi, phi, interface_unit_normal);
    *packaged_v_psi = get<Tags::VPsi>(char_fields);
    *packaged_v_zero = get<Tags::VZero<Dim>>(char_fields);
    *packaged_v_plus = get<Tags::VPlus>(char_fields);
    *packaged_v_minus = get<Tags::VMinus>(char_fields);
  }

  {  // package characteristic speeds
    Scalar<DataVector> shift_dot_normal{};
    get(shift_dot_normal)
        .set_data_ref(make_not_null(&get<1>(*packaged_char_speeds)));
    dot_product(make_not_null(&shift_dot_normal), shift, interface_unit_normal);
    get(shift_dot_normal) *= -1.;

    get<0>(*packaged_char_speeds) =
        (1. + get(constraint_gamma1)) * get(shift_dot_normal);

    get<2>(*packaged_char_speeds) = get(shift_dot_normal) + get(lapse);
    get<3>(*packaged_char_speeds) = get(shift_dot_normal) - get(lapse);

    if (normal_dot_mesh_velocity.has_value()) {
      get<0>(*packaged_char_speeds) -= get(*normal_dot_mesh_velocity);
      get<1>(*packaged_char_speeds) -= get(*normal_dot_mesh_velocity);
      get<2>(*packaged_char_speeds) -= get(*normal_dot_mesh_velocity);
      get<3>(*packaged_char_speeds) -= get(*normal_dot_mesh_velocity);
    }
  }

  return max(max(get<0>(*packaged_char_speeds), get<1>(*packaged_char_speeds),
                 get<2>(*packaged_char_speeds)));
}

template <size_t Dim>
void UpwindPenalty<Dim>::dg_boundary_terms(
    const gsl::not_null<Scalar<DataVector>*> pi_boundary_correction,
    const gsl::not_null<tnsr::i<DataVector, Dim, Frame::Inertial>*>
        phi_boundary_correction,
    const gsl::not_null<Scalar<DataVector>*> psi_boundary_correction,

    const Scalar<DataVector>& v_psi_int,
    const tnsr::i<DataVector, Dim, Frame::Inertial>& v_zero_int,
    const Scalar<DataVector>& v_plus_int, const Scalar<DataVector>& v_minus_int,
    const Scalar<DataVector>& gamma2_int,
    const tnsr::i<DataVector, Dim, Frame::Inertial>& interface_unit_normal_int,
    const tnsr::a<DataVector, 3, Frame::Inertial>& char_speeds_int,

    const Scalar<DataVector>& v_psi_ext,
    const tnsr::i<DataVector, Dim, Frame::Inertial>& v_zero_ext,
    const Scalar<DataVector>& v_plus_ext, const Scalar<DataVector>& v_minus_ext,
    const Scalar<DataVector>& gamma2_ext,
    const tnsr::i<DataVector, Dim, Frame::Inertial>& interface_unit_normal_ext,
    const tnsr::a<DataVector, 3, Frame::Inertial>& char_speeds_ext,
    dg::Formulation /*dg_formulation*/) const noexcept {
  Variables<CurvedScalarWave_detail::char_field_tags<Dim>>
      weighted_char_fields_int(get_size(get(v_psi_int)), 0.);
  Variables<CurvedScalarWave_detail::char_field_tags<Dim>>
      weighted_char_fields_ext(get_size(get(v_psi_int)), 0.);
  CurvedScalarWave_detail::weight_char_fields<Dim>(
      make_not_null(&weighted_char_fields_int),
      make_not_null(&weighted_char_fields_ext), v_psi_int, v_zero_int,
      v_plus_int, v_minus_int, char_speeds_int, v_psi_ext, v_zero_ext,
      v_plus_ext, v_minus_ext, char_speeds_ext);

  const auto weighted_evolved_fields_int =
      evolved_fields_from_characteristic_fields(
          gamma2_int, get<Tags::VPsi>(weighted_char_fields_int),
          get<Tags::VZero<Dim>>(weighted_char_fields_int),
          get<Tags::VPlus>(weighted_char_fields_int),
          get<Tags::VMinus>(weighted_char_fields_int),
          interface_unit_normal_int);

  const auto weighted_evolved_fields_ext =
      evolved_fields_from_characteristic_fields(
          gamma2_ext, get<Tags::VPsi>(weighted_char_fields_ext),
          get<Tags::VZero<Dim>>(weighted_char_fields_ext),
          get<Tags::VPlus>(weighted_char_fields_ext),
          get<Tags::VMinus>(weighted_char_fields_ext),
          interface_unit_normal_ext);

  get(*psi_boundary_correction) = get(get<Psi>(weighted_evolved_fields_ext)) -
                                  get(get<Psi>(weighted_evolved_fields_int));
  get(*pi_boundary_correction) = get(get<Pi>(weighted_evolved_fields_ext)) -
                                 get(get<Pi>(weighted_evolved_fields_int));
  for (size_t i = 0; i < Dim; ++i) {
    phi_boundary_correction->get(i) =
        get<Phi<Dim>>(weighted_evolved_fields_ext).get(i) -
        get<Phi<Dim>>(weighted_evolved_fields_int).get(i);
  }
}

template <size_t Dim>
// NOLINTNEXTLINE
PUP::able::PUP_ID UpwindPenalty<Dim>::my_PUP_ID = 0;

#define DIM(data) BOOST_PP_TUPLE_ELEM(0, data)

#define INSTANTIATION(_, data) template class UpwindPenalty<DIM(data)>;

GENERATE_INSTANTIATIONS(INSTANTIATION, (1, 2, 3))

#undef INSTANTIATION
#undef DIM
}  // namespace CurvedScalarWave::BoundaryCorrections
