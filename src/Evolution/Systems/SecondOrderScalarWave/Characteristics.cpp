// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Evolution/Systems/SecondOrderScalarWave/Characteristics.hpp"

#include <array>

#include "DataStructures/Tensor/EagerMath/DotProduct.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "DataStructures/Variables.hpp"
#include "Utilities/ContainerHelpers.hpp"
#include "Utilities/GenerateInstantiations.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/SetNumberOfGridPoints.hpp"
#include "Utilities/TMPL.hpp"

namespace SecondOrderScalarWave {
template <size_t Dim>
void characteristic_speeds(
    const gsl::not_null<std::array<DataVector, 3>*> char_speeds,
    const tnsr::i<DataVector, Dim, Frame::Inertial>& unit_normal_one_form) {
  set_number_of_grid_points(char_speeds, unit_normal_one_form);
  (*char_speeds)[0] = 0.;   // VZero
  (*char_speeds)[1] = 1.;   // VPlus
  (*char_speeds)[2] = -1.;  // VMinus
}

template <size_t Dim>
std::array<DataVector, 3> characteristic_speeds(
    const tnsr::i<DataVector, Dim, Frame::Inertial>& unit_normal_one_form) {
  auto char_speeds = make_with_value<std::array<DataVector, 3>>(
      get<0>(unit_normal_one_form), 0.);
  characteristic_speeds(make_not_null(&char_speeds), unit_normal_one_form);
  return char_speeds;
}

template <size_t Dim>
void characteristic_fields(
    const gsl::not_null<
        Variables<tmpl::list<Tags::VZero<Dim>, Tags::VPlus, Tags::VMinus>>*>
        char_fields,
    const Scalar<DataVector>& pi,
    const tnsr::i<DataVector, Dim, Frame::Inertial>& phi,
    const tnsr::i<DataVector, Dim, Frame::Inertial>& unit_normal_one_form) {
  if (UNLIKELY(char_fields->number_of_grid_points() != get(pi).size())) {
    char_fields->initialize(get(pi).size());
  }
  const auto phi_dot_normal = dot_product(unit_normal_one_form, phi);

  for (size_t i = 0; i < Dim; ++i) {
    get<Tags::VZero<Dim>>(*char_fields).get(i) =
        phi.get(i) - unit_normal_one_form.get(i) * get(phi_dot_normal);
  }

  get(get<Tags::VPlus>(*char_fields)) = get(pi) + get(phi_dot_normal);
  get(get<Tags::VMinus>(*char_fields)) = get(pi) - get(phi_dot_normal);
}

template <size_t Dim>
Variables<tmpl::list<Tags::VZero<Dim>, Tags::VPlus, Tags::VMinus>>
characteristic_fields(
    const Scalar<DataVector>& pi,
    const tnsr::i<DataVector, Dim, Frame::Inertial>& phi,
    const tnsr::i<DataVector, Dim, Frame::Inertial>& unit_normal_one_form) {
  Variables<tmpl::list<Tags::VZero<Dim>, Tags::VPlus, Tags::VMinus>>
      char_fields(get_size(get(pi)));
  characteristic_fields(make_not_null(&char_fields), pi, phi,
                        unit_normal_one_form);
  return char_fields;
}

template <size_t Dim>
void fields_from_inverse_characteristic_transform(
    const gsl::not_null<Variables<tmpl::list<Tags::Pi, Tags::Phi<Dim>>>*>
        evolved_fields,
    const tnsr::i<DataVector, Dim, Frame::Inertial>& v_zero,
    const Scalar<DataVector>& v_plus, const Scalar<DataVector>& v_minus,
    const tnsr::i<DataVector, Dim, Frame::Inertial>& unit_normal_one_form) {
  if (UNLIKELY(evolved_fields->number_of_grid_points() != get(v_plus).size())) {
    evolved_fields->initialize(get(v_plus).size());
  }
  get(get<Tags::Pi>(*evolved_fields)) = 0.5 * (get(v_plus) + get(v_minus));
  for (size_t i = 0; i < Dim; ++i) {
    get<Tags::Phi<Dim>>(*evolved_fields).get(i) =
        0.5 * (get(v_plus) - get(v_minus)) * unit_normal_one_form.get(i) +
        v_zero.get(i);
  }
}

template <size_t Dim>
Variables<tmpl::list<Tags::Pi, Tags::Phi<Dim>>>
fields_from_inverse_characteristic_transform(
    const tnsr::i<DataVector, Dim, Frame::Inertial>& v_zero,
    const Scalar<DataVector>& v_plus, const Scalar<DataVector>& v_minus,
    const tnsr::i<DataVector, Dim, Frame::Inertial>& unit_normal_one_form) {
  Variables<tmpl::list<Tags::Pi, Tags::Phi<Dim>>> evolved_fields(
      get_size(get(v_plus)));
  fields_from_inverse_characteristic_transform(make_not_null(&evolved_fields),
                                               v_zero, v_plus, v_minus,
                                               unit_normal_one_form);
  return evolved_fields;
}
}  // namespace SecondOrderScalarWave

#define DIM(data) BOOST_PP_TUPLE_ELEM(0, data)

#define INSTANTIATE(_, data)                                                   \
  template void SecondOrderScalarWave::characteristic_speeds(                  \
      const gsl::not_null<std::array<DataVector, 3>*> char_speeds,             \
      const tnsr::i<DataVector, DIM(data), Frame::Inertial>&                   \
          unit_normal_one_form);                                               \
  template std::array<DataVector, 3>                                           \
  SecondOrderScalarWave::characteristic_speeds(                                \
      const tnsr::i<DataVector, DIM(data), Frame::Inertial>&                   \
          unit_normal_one_form);                                               \
  template struct SecondOrderScalarWave::Tags::CharacteristicSpeedsCompute<    \
      DIM(data)>;                                                              \
  template void SecondOrderScalarWave::characteristic_fields(                  \
      const gsl::not_null<                                                     \
          Variables<tmpl::list<SecondOrderScalarWave::Tags::VZero<DIM(data)>,  \
                               SecondOrderScalarWave::Tags::VPlus,             \
                               SecondOrderScalarWave::Tags::VMinus>>*>         \
          char_fields,                                                         \
      const Scalar<DataVector>& pi,                                            \
      const tnsr::i<DataVector, DIM(data), Frame::Inertial>& phi,              \
      const tnsr::i<DataVector, DIM(data), Frame::Inertial>&                   \
          unit_normal_one_form);                                               \
  template Variables<tmpl::list<SecondOrderScalarWave::Tags::VZero<DIM(data)>, \
                                SecondOrderScalarWave::Tags::VPlus,            \
                                SecondOrderScalarWave::Tags::VMinus>>          \
  SecondOrderScalarWave::characteristic_fields(                                \
      const Scalar<DataVector>& pi,                                            \
      const tnsr::i<DataVector, DIM(data), Frame::Inertial>& phi,              \
      const tnsr::i<DataVector, DIM(data), Frame::Inertial>&                   \
          unit_normal_one_form);                                               \
  template struct SecondOrderScalarWave::Tags::CharacteristicFieldsCompute<    \
      DIM(data)>;                                                              \
  template void                                                                \
  SecondOrderScalarWave::fields_from_inverse_characteristic_transform(         \
      const gsl::not_null<                                                     \
          Variables<tmpl::list<SecondOrderScalarWave::Tags::Pi,                \
                               SecondOrderScalarWave::Tags::Phi<DIM(data)>>>*> \
          evolved_fields,                                                      \
      const tnsr::i<DataVector, DIM(data), Frame::Inertial>& v_zero,           \
      const Scalar<DataVector>& v_plus, const Scalar<DataVector>& v_minus,     \
      const tnsr::i<DataVector, DIM(data), Frame::Inertial>&                   \
          unit_normal_one_form);                                               \
  template Variables<tmpl::list<SecondOrderScalarWave::Tags::Pi,               \
                                SecondOrderScalarWave::Tags::Phi<DIM(data)>>>  \
  SecondOrderScalarWave::fields_from_inverse_characteristic_transform(         \
      const tnsr::i<DataVector, DIM(data), Frame::Inertial>& v_zero,           \
      const Scalar<DataVector>& v_plus, const Scalar<DataVector>& v_minus,     \
      const tnsr::i<DataVector, DIM(data), Frame::Inertial>&                   \
          unit_normal_one_form);                                               \
  template struct SecondOrderScalarWave::Tags::                                \
      FieldsFromInverseCharacteristicTransformCompute<DIM(data)>;

GENERATE_INSTANTIATIONS(INSTANTIATE, (1, 2, 3))

#undef INSTANTIATE
#undef DIM
