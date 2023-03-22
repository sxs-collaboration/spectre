// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Evolution/Systems/CurvedScalarWave/Characteristics.hpp"

#include <algorithm>
#include <array>
#include <limits>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/EagerMath/DotProduct.hpp"
#include "DataStructures/Tensor/EagerMath/Magnitude.hpp"
#include "Evolution/Systems/CurvedScalarWave/Tags.hpp"
#include "PointwiseFunctions/GeneralRelativity/IndexManipulation.hpp"
#include "Utilities/GenerateInstantiations.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/MakeWithValue.hpp"

namespace CurvedScalarWave {
template <size_t SpatialDim>
void characteristic_speeds(
    const gsl::not_null<tnsr::a<DataVector, 3, Frame::Inertial>*> char_speeds,
    const Scalar<DataVector>& gamma_1, const Scalar<DataVector>& lapse,
    const tnsr::I<DataVector, SpatialDim, Frame::Inertial>& shift,
    const tnsr::i<DataVector, SpatialDim, Frame::Inertial>&
        unit_normal_one_form) {
  destructive_resize_components(char_speeds, get(gamma_1).size());
  const auto shift_dot_normal = get(dot_product(shift, unit_normal_one_form));
  get<0>(*char_speeds) = -(1. + get(gamma_1)) * shift_dot_normal;  // v(VPsi)
  get<1>(*char_speeds) = -shift_dot_normal;                        // v(VZero)
  get<2>(*char_speeds) = -shift_dot_normal + get(lapse);           // v(VPlus)
  get<3>(*char_speeds) = -shift_dot_normal - get(lapse);           // v(VMinus)
}

template <size_t SpatialDim>
void characteristic_speeds(
    const gsl::not_null<std::array<DataVector, 4>*> char_speeds,
    const Scalar<DataVector>& gamma_1, const Scalar<DataVector>& lapse,
    const tnsr::I<DataVector, SpatialDim, Frame::Inertial>& shift,
    const tnsr::i<DataVector, SpatialDim, Frame::Inertial>&
        unit_normal_one_form) {
  const size_t size = get(gamma_1).size();
  for (auto& char_speed : *char_speeds) {
    char_speed.destructive_resize(size);
  }
  const auto shift_dot_normal = get(dot_product(shift, unit_normal_one_form));
  (*char_speeds)[0] = -(1. + get(gamma_1)) * shift_dot_normal;  // v(VPsi)
  (*char_speeds)[1] = -shift_dot_normal;                        // v(VZero)
  (*char_speeds)[2] = -shift_dot_normal + get(lapse);           // v(VPlus)
  (*char_speeds)[3] = -shift_dot_normal - get(lapse);           // v(VMinus)
}

template <size_t SpatialDim>
std::array<DataVector, 4> characteristic_speeds(
    const Scalar<DataVector>& gamma_1, const Scalar<DataVector>& lapse,
    const tnsr::I<DataVector, SpatialDim, Frame::Inertial>& shift,
    const tnsr::i<DataVector, SpatialDim, Frame::Inertial>&
        unit_normal_one_form) {
  std::array<DataVector, 4> char_speeds{};
  characteristic_speeds(make_not_null(&char_speeds), gamma_1, lapse, shift,
                        unit_normal_one_form);
  return char_speeds;
}

template <size_t SpatialDim>
void characteristic_fields(
    const gsl::not_null<Variables<tmpl::list<
        Tags::VPsi, Tags::VZero<SpatialDim>, Tags::VPlus, Tags::VMinus>>*>
        char_fields,
    const Scalar<DataVector>& gamma_2, const Scalar<DataVector>& psi,
    const Scalar<DataVector>& pi,
    const tnsr::i<DataVector, SpatialDim, Frame::Inertial>& phi,
    const tnsr::i<DataVector, SpatialDim, Frame::Inertial>&
        unit_normal_one_form,
    const tnsr::I<DataVector, SpatialDim, Frame::Inertial>&
        unit_normal_vector) {
  char_fields->initialize(get(gamma_2).size());
  dot_product(make_not_null(&get<Tags::VMinus>(*char_fields)),
              unit_normal_vector, phi);
  // Eq.(34) of Holst+ (2004) for VZero
  for (size_t i = 0; i < SpatialDim; ++i) {
    get<Tags::VZero<SpatialDim>>(*char_fields).get(i) =
        phi.get(i) -
        unit_normal_one_form.get(i) * get(get<Tags::VMinus>(*char_fields));
  }
  // Eq.(33) of Holst+ (2004) for VPsi
  get<Tags::VPsi>(*char_fields) = psi;
  // Eq.(35) of Holst+ (2004) for VPlus and VMinus
  get(get<Tags::VPlus>(*char_fields)) =
      get(pi) + get(get<Tags::VMinus>(*char_fields)) - get(gamma_2) * get(psi);
  get(get<Tags::VMinus>(*char_fields)) =
      get(pi) - get(get<Tags::VMinus>(*char_fields)) - get(gamma_2) * get(psi);
}

template <size_t SpatialDim>
Variables<
    tmpl::list<Tags::VPsi, Tags::VZero<SpatialDim>, Tags::VPlus, Tags::VMinus>>
characteristic_fields(
    const Scalar<DataVector>& gamma_2, const Scalar<DataVector>& psi,
    const Scalar<DataVector>& pi,
    const tnsr::i<DataVector, SpatialDim, Frame::Inertial>& phi,
    const tnsr::i<DataVector, SpatialDim, Frame::Inertial>&
        unit_normal_one_form,
    const tnsr::I<DataVector, SpatialDim, Frame::Inertial>&
        unit_normal_vector) {
  Variables<tmpl::list<Tags::VPsi, Tags::VZero<SpatialDim>, Tags::VPlus,
                       Tags::VMinus>>
      char_fields{get(gamma_2).size()};
  characteristic_fields(make_not_null(&char_fields), gamma_2, psi, pi, phi,
                        unit_normal_one_form, unit_normal_vector);
  return char_fields;
}

template <size_t SpatialDim>
void characteristic_fields(
    const gsl::not_null<Scalar<DataVector>*>& v_psi,
    const gsl::not_null<tnsr::i<DataVector, SpatialDim, Frame::Inertial>*>&
        v_zero,
    const gsl::not_null<Scalar<DataVector>*>& v_plus,
    const gsl::not_null<Scalar<DataVector>*>& v_minus,
    const Scalar<DataVector>& gamma_2, const Scalar<DataVector>& psi,
    const Scalar<DataVector>& pi,
    const tnsr::i<DataVector, SpatialDim, Frame::Inertial>& phi,
    const tnsr::i<DataVector, SpatialDim, Frame::Inertial>&
        unit_normal_one_form,
    const tnsr::I<DataVector, SpatialDim, Frame::Inertial>&
        unit_normal_vector) {
  const size_t size = get(gamma_2).size();
  destructive_resize_components(v_psi, size);
  destructive_resize_components(v_zero, size);
  destructive_resize_components(v_plus, size);
  destructive_resize_components(v_minus, size);

  dot_product(v_minus, unit_normal_vector, phi);
  // Eq.(34) of Holst+ (2004) for VZero
  for (size_t i = 0; i < SpatialDim; ++i) {
    v_zero->get(i) = phi.get(i) - unit_normal_one_form.get(i) * get(*v_minus);
  }
  // Eq.(33) of Holst+ (2004) for VPsi
  *v_psi = psi;
  // Eq.(35) of Holst+ (2004) for VPlus and VMinus
  get(*v_plus) = get(pi) + get(*v_minus) - get(gamma_2) * get(psi);
  get(*v_minus) = get(pi) - get(*v_minus) - get(gamma_2) * get(psi);
}

template <size_t SpatialDim>
void evolved_fields_from_characteristic_fields(
    gsl::not_null<Scalar<DataVector>*> psi,
    gsl::not_null<Scalar<DataVector>*> pi,
    gsl::not_null<tnsr::i<DataVector, SpatialDim, Frame::Inertial>*> phi,
    const Scalar<DataVector>& gamma_2, const Scalar<DataVector>& v_psi,
    const tnsr::i<DataVector, SpatialDim, Frame::Inertial>& v_zero,
    const Scalar<DataVector>& v_plus, const Scalar<DataVector>& v_minus,
    const tnsr::i<DataVector, SpatialDim, Frame::Inertial>&
        unit_normal_one_form) {
  const size_t size = get(gamma_2).size();
  destructive_resize_components(psi, size);
  destructive_resize_components(pi, size);
  destructive_resize_components(phi, size);
  // Eq.(36) of Holst+ (2005) for Psi
  *psi = v_psi;
  // Eq.(37) - (38) of Holst+ (2004) for Pi and Phi
  pi->get() = 0.5 * (get(v_plus) + get(v_minus)) + get(gamma_2) * get(v_psi);
  for (size_t i = 0; i < SpatialDim; ++i) {
    phi->get(i) =
        0.5 * (get(v_plus) - get(v_minus)) * unit_normal_one_form.get(i) +
        v_zero.get(i);
  }
}

template <size_t SpatialDim>
void evolved_fields_from_characteristic_fields(
    const gsl::not_null<
        Variables<tmpl::list<Tags::Psi, Tags::Pi, Tags::Phi<SpatialDim>>>*>
        evolved_fields,
    const Scalar<DataVector>& gamma_2, const Scalar<DataVector>& v_psi,
    const tnsr::i<DataVector, SpatialDim, Frame::Inertial>& v_zero,
    const Scalar<DataVector>& v_plus, const Scalar<DataVector>& v_minus,
    const tnsr::i<DataVector, SpatialDim, Frame::Inertial>&
        unit_normal_one_form) {
  evolved_fields->initialize(get_size(get(gamma_2)));
  // Eq.(36) of Holst+ (2005) for Psi
  get<Tags::Psi>(*evolved_fields) = v_psi;

  // Eq.(37) - (38) of Holst+ (2004) for Pi and Phi
  get<Tags::Pi>(*evolved_fields).get() =
      0.5 * (get(v_plus) + get(v_minus)) + get(gamma_2) * get(v_psi);
  for (size_t i = 0; i < SpatialDim; ++i) {
    get<Tags::Phi<SpatialDim>>(*evolved_fields).get(i) =
        0.5 * (get(v_plus) - get(v_minus)) * unit_normal_one_form.get(i) +
        v_zero.get(i);
  }
}

template <size_t SpatialDim>
Variables<tmpl::list<Tags::Psi, Tags::Pi, Tags::Phi<SpatialDim>>>
evolved_fields_from_characteristic_fields(
    const Scalar<DataVector>& gamma_2, const Scalar<DataVector>& v_psi,
    const tnsr::i<DataVector, SpatialDim, Frame::Inertial>& v_zero,
    const Scalar<DataVector>& v_plus, const Scalar<DataVector>& v_minus,
    const tnsr::i<DataVector, SpatialDim, Frame::Inertial>&
        unit_normal_one_form) {
  Variables<tmpl::list<Tags::Psi, Tags::Pi, Tags::Phi<SpatialDim>>>
      evolved_fields(get(gamma_2).size());
  evolved_fields_from_characteristic_fields(make_not_null(&evolved_fields),
                                            gamma_2, v_psi, v_zero, v_plus,
                                            v_minus, unit_normal_one_form);
  return evolved_fields;
}

namespace Tags {
template <size_t SpatialDim>
void ComputeLargestCharacteristicSpeed<SpatialDim>::function(
    const gsl::not_null<double*> max_speed, const Scalar<DataVector>& gamma_1,
    const Scalar<DataVector>& lapse,
    const tnsr::I<DataVector, SpatialDim, Frame::Inertial>& shift,
    const tnsr::ii<DataVector, SpatialDim, Frame::Inertial>& spatial_metric) {
  const auto shift_magnitude = magnitude(shift, spatial_metric);
  *max_speed =
      std::max(max(abs(1. + get(gamma_1)) * get(shift_magnitude)),  // v(VPsi)
               max(get(shift_magnitude) +
                   abs(get(lapse))));  // v(VZero), v(VPlus),v(VMinus)
}
}  // namespace Tags
}  // namespace CurvedScalarWave

#define DIM(data) BOOST_PP_TUPLE_ELEM(0, data)

#define INSTANTIATE(_, data)                                                  \
  template void CurvedScalarWave::characteristic_speeds(                      \
      const gsl::not_null<tnsr::a<DataVector, 3, Frame::Inertial>*>           \
          char_speeds,                                                        \
      const Scalar<DataVector>& gamma_1, const Scalar<DataVector>& lapse,     \
      const tnsr::I<DataVector, DIM(data), Frame::Inertial>& shift,           \
      const tnsr::i<DataVector, DIM(data), Frame::Inertial>&                  \
          unit_normal_one_form);                                              \
  template void CurvedScalarWave::characteristic_speeds(                      \
      const gsl::not_null<std::array<DataVector, 4>*> char_speeds,            \
      const Scalar<DataVector>& gamma_1, const Scalar<DataVector>& lapse,     \
      const tnsr::I<DataVector, DIM(data), Frame::Inertial>& shift,           \
      const tnsr::i<DataVector, DIM(data), Frame::Inertial>&                  \
          unit_normal_one_form);                                              \
  template std::array<DataVector, 4> CurvedScalarWave::characteristic_speeds( \
      const Scalar<DataVector>& gamma_1, const Scalar<DataVector>& lapse,     \
      const tnsr::I<DataVector, DIM(data), Frame::Inertial>& shift,           \
      const tnsr::i<DataVector, DIM(data), Frame::Inertial>&                  \
          unit_normal_one_form);                                              \
  template struct CurvedScalarWave::CharacteristicSpeedsCompute<DIM(data)>;   \
  template void CurvedScalarWave::characteristic_fields(                      \
      const gsl::not_null<Variables<tmpl::list<                               \
          CurvedScalarWave::Tags::VPsi,                                       \
          CurvedScalarWave::Tags::VZero<DIM(data)>,                           \
          CurvedScalarWave::Tags::VPlus, CurvedScalarWave::Tags::VMinus>>*>   \
          char_fields,                                                        \
      const Scalar<DataVector>& gamma_2, const Scalar<DataVector>& psi,       \
      const Scalar<DataVector>& pi,                                           \
      const tnsr::i<DataVector, DIM(data), Frame::Inertial>& phi,             \
      const tnsr::i<DataVector, DIM(data), Frame::Inertial>&                  \
          unit_normal_one_form,                                               \
      const tnsr::I<DataVector, DIM(data), Frame::Inertial>&                  \
          unit_normal_vector);                                                \
  template Variables<tmpl::list<                                              \
      CurvedScalarWave::Tags::VPsi, CurvedScalarWave::Tags::VZero<DIM(data)>, \
      CurvedScalarWave::Tags::VPlus, CurvedScalarWave::Tags::VMinus>>         \
  CurvedScalarWave::characteristic_fields(                                    \
      const Scalar<DataVector>& gamma_2, const Scalar<DataVector>& psi,       \
      const Scalar<DataVector>& pi,                                           \
      const tnsr::i<DataVector, DIM(data), Frame::Inertial>& phi,             \
      const tnsr::i<DataVector, DIM(data), Frame::Inertial>&                  \
          unit_normal_one_form,                                               \
      const tnsr::I<DataVector, DIM(data), Frame::Inertial>&                  \
          unit_normal_vector);                                                \
  template void CurvedScalarWave::characteristic_fields(                      \
      const gsl::not_null<Scalar<DataVector>*>& v_psi,                        \
      const gsl::not_null<tnsr::i<DataVector, DIM(data), Frame::Inertial>*>&  \
          v_zero,                                                             \
      const gsl::not_null<Scalar<DataVector>*>& v_plus,                       \
      const gsl::not_null<Scalar<DataVector>*>& v_minus,                      \
      const Scalar<DataVector>& gamma_2, const Scalar<DataVector>& psi,       \
      const Scalar<DataVector>& pi,                                           \
      const tnsr::i<DataVector, DIM(data), Frame::Inertial>& phi,             \
      const tnsr::i<DataVector, DIM(data), Frame::Inertial>&                  \
          unit_normal_one_form,                                               \
      const tnsr::I<DataVector, DIM(data), Frame::Inertial>&                  \
          unit_normal_vector);                                                \
  template struct CurvedScalarWave::CharacteristicFieldsCompute<DIM(data)>;   \
  template void CurvedScalarWave::evolved_fields_from_characteristic_fields(  \
      const gsl::not_null<Variables<                                          \
          tmpl::list<CurvedScalarWave::Tags::Psi, CurvedScalarWave::Tags::Pi, \
                     CurvedScalarWave::Tags::Phi<DIM(data)>>>*>               \
          evolved_fields,                                                     \
      const Scalar<DataVector>& gamma_2, const Scalar<DataVector>& v_psi,     \
      const tnsr::i<DataVector, DIM(data), Frame::Inertial>& v_zero,          \
      const Scalar<DataVector>& v_plus, const Scalar<DataVector>& v_minus,    \
      const tnsr::i<DataVector, DIM(data), Frame::Inertial>&                  \
          unit_normal_one_form);                                              \
  template Variables<                                                         \
      tmpl::list<CurvedScalarWave::Tags::Psi, CurvedScalarWave::Tags::Pi,     \
                 CurvedScalarWave::Tags::Phi<DIM(data)>>>                     \
  CurvedScalarWave::evolved_fields_from_characteristic_fields(                \
      const Scalar<DataVector>& gamma_2, const Scalar<DataVector>& v_psi,     \
      const tnsr::i<DataVector, DIM(data), Frame::Inertial>& v_zero,          \
      const Scalar<DataVector>& v_plus, const Scalar<DataVector>& v_minus,    \
      const tnsr::i<DataVector, DIM(data), Frame::Inertial>&                  \
          unit_normal_one_form);                                              \
  template void CurvedScalarWave::evolved_fields_from_characteristic_fields(  \
      gsl::not_null<Scalar<DataVector>*> psi,                                 \
      gsl::not_null<Scalar<DataVector>*> pi,                                  \
      gsl::not_null<tnsr::i<DataVector, DIM(data), Frame::Inertial>*> phi,    \
      const Scalar<DataVector>& gamma_2, const Scalar<DataVector>& v_psi,     \
      const tnsr::i<DataVector, DIM(data), Frame::Inertial>& v_zero,          \
      const Scalar<DataVector>& v_plus, const Scalar<DataVector>& v_minus,    \
      const tnsr::i<DataVector, DIM(data), Frame::Inertial>&                  \
          unit_normal_one_form);                                              \
  template struct CurvedScalarWave::                                          \
      EvolvedFieldsFromCharacteristicFieldsCompute<DIM(data)>;                \
  template struct CurvedScalarWave::Tags::ComputeLargestCharacteristicSpeed<  \
      DIM(data)>;

GENERATE_INSTANTIATIONS(INSTANTIATE, (1, 2, 3))

#undef INSTANTIATE
#undef DIM
