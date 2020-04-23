// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Evolution/Systems/ScalarWave/Characteristics.hpp"

#include <array>

#include "DataStructures/Tensor/EagerMath/DotProduct.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "DataStructures/Variables.hpp"  // IWYU pragma: keep
#include "Utilities/ContainerHelpers.hpp"
#include "Utilities/GenerateInstantiations.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/TMPL.hpp"

namespace ScalarWave {
template <size_t Dim>
void characteristic_speeds(
    const gsl::not_null<std::array<DataVector, 4>*> char_speeds,
    const tnsr::i<DataVector, Dim, Frame::Inertial>&
        unit_normal_one_form) noexcept {
  destructive_resize_components(char_speeds,
                                get<0>(unit_normal_one_form).size());
  (*char_speeds)[0] = 0.;   // v(VPsi)
  (*char_speeds)[1] = 0.;   // v(VZero)
  (*char_speeds)[2] = 1.;   // v(VPlus)
  (*char_speeds)[3] = -1.;  // v(VMinus)
}

template <size_t Dim>
std::array<DataVector, 4> characteristic_speeds(
    const tnsr::i<DataVector, Dim, Frame::Inertial>&
        unit_normal_one_form) noexcept {
  auto char_speeds = make_with_value<std::array<DataVector, 4>>(
      get<0>(unit_normal_one_form), 0.);
  characteristic_speeds(make_not_null(&char_speeds), unit_normal_one_form);
  return char_speeds;
}

template <size_t Dim>
void characteristic_fields(
    const gsl::not_null<Variables<
        tmpl::list<Tags::VPsi, Tags::VZero<Dim>, Tags::VPlus, Tags::VMinus>>*>
        char_fields,
    const Scalar<DataVector>& gamma_2, const Scalar<DataVector>& psi,
    const Scalar<DataVector>& pi,
    const tnsr::i<DataVector, Dim, Frame::Inertial>& phi,
    const tnsr::i<DataVector, Dim, Frame::Inertial>&
        unit_normal_one_form) noexcept {
  if (UNLIKELY(char_fields->number_of_grid_points() != get(psi).size())) {
    char_fields->initialize(get(psi).size());
  }
  // Compute phi_dot_normal = n^i \Phi_{i} = \sum_i n_i \Phi_{i}
  // (we use normal_one_form and normal_vector interchangeably in flat space)
  const auto phi_dot_normal = dot_product(unit_normal_one_form, phi);

  // Eq.(34) of Holst+ (2004) for VZero
  for (size_t i = 0; i < Dim; ++i) {
    get<Tags::VZero<Dim>>(*char_fields).get(i) =
        phi.get(i) - unit_normal_one_form.get(i) * get(phi_dot_normal);
  }

  // Eq.(33) of Holst+ (2004) for VPsi
  get<Tags::VPsi>(*char_fields) = psi;

  // Eq.(35) of Holst+ (2004) for VPlus and VMinus
  get(get<Tags::VPlus>(*char_fields)) =
      get(pi) + get(phi_dot_normal) - get(gamma_2) * get(psi);
  get(get<Tags::VMinus>(*char_fields)) =
      get(pi) - get(phi_dot_normal) - get(gamma_2) * get(psi);
}

template <size_t Dim>
Variables<tmpl::list<Tags::VPsi, Tags::VZero<Dim>, Tags::VPlus, Tags::VMinus>>
characteristic_fields(const Scalar<DataVector>& gamma_2,
                      const Scalar<DataVector>& psi,
                      const Scalar<DataVector>& pi,
                      const tnsr::i<DataVector, Dim, Frame::Inertial>& phi,
                      const tnsr::i<DataVector, Dim, Frame::Inertial>&
                          unit_normal_one_form) noexcept {
  Variables<tmpl::list<Tags::VPsi, Tags::VZero<Dim>, Tags::VPlus, Tags::VMinus>>
      char_fields(get_size(get(gamma_2)));
  characteristic_fields(make_not_null(&char_fields), gamma_2, psi, pi, phi,
                        unit_normal_one_form);
  return char_fields;
}

template <size_t Dim>
void evolved_fields_from_characteristic_fields(
    const gsl::not_null<Variables<tmpl::list<Psi, Pi, Phi<Dim>>>*>
        evolved_fields,
    const Scalar<DataVector>& gamma_2, const Scalar<DataVector>& v_psi,
    const tnsr::i<DataVector, Dim, Frame::Inertial>& v_zero,
    const Scalar<DataVector>& v_plus, const Scalar<DataVector>& v_minus,
    const tnsr::i<DataVector, Dim, Frame::Inertial>&
        unit_normal_one_form) noexcept {
  if (UNLIKELY(evolved_fields->number_of_grid_points() != get(v_psi).size())) {
    evolved_fields->initialize(get(v_psi).size());
  }
  // Eq.(36) of Holst+ (2004) for Psi
  get<Psi>(*evolved_fields) = v_psi;

  // Eq.(37) - (38) of Holst+ (2004) for Pi and Phi
  get<Pi>(*evolved_fields).get() =
      0.5 * (get(v_plus) + get(v_minus)) + get(gamma_2) * get(v_psi);
  for (size_t i = 0; i < Dim; ++i) {
    get<Phi<Dim>>(*evolved_fields).get(i) =
        0.5 * (get(v_plus) - get(v_minus)) * unit_normal_one_form.get(i) +
        v_zero.get(i);
  }
}

template <size_t Dim>
Variables<tmpl::list<Psi, Pi, Phi<Dim>>>
evolved_fields_from_characteristic_fields(
    const Scalar<DataVector>& gamma_2, const Scalar<DataVector>& v_psi,
    const tnsr::i<DataVector, Dim, Frame::Inertial>& v_zero,
    const Scalar<DataVector>& v_plus, const Scalar<DataVector>& v_minus,
    const tnsr::i<DataVector, Dim, Frame::Inertial>&
        unit_normal_one_form) noexcept {
  Variables<tmpl::list<Psi, Pi, Phi<Dim>>> evolved_fields(
      get_size(get(gamma_2)));
  evolved_fields_from_characteristic_fields(make_not_null(&evolved_fields),
                                            gamma_2, v_psi, v_zero, v_plus,
                                            v_minus, unit_normal_one_form);
  return evolved_fields;
}
}  // namespace ScalarWave

#define DIM(data) BOOST_PP_TUPLE_ELEM(0, data)

#define INSTANTIATE(_, data)                                                   \
  template void ScalarWave::characteristic_speeds(                             \
      const gsl::not_null<std::array<DataVector, 4>*> char_speeds,             \
      const tnsr::i<DataVector, DIM(data), Frame::Inertial>&                   \
          unit_normal_one_form) noexcept;                                      \
  template std::array<DataVector, 4> ScalarWave::characteristic_speeds(        \
      const tnsr::i<DataVector, DIM(data), Frame::Inertial>&                   \
          unit_normal_one_form) noexcept;                                      \
  template struct ScalarWave::Tags::CharacteristicSpeedsCompute<DIM(data)>;    \
  template void ScalarWave::characteristic_fields(                             \
      const gsl::not_null<Variables<tmpl::list<                                \
          ScalarWave::Tags::VPsi, ScalarWave::Tags::VZero<DIM(data)>,          \
          ScalarWave::Tags::VPlus, ScalarWave::Tags::VMinus>>*>                \
          char_fields,                                                         \
      const Scalar<DataVector>& gamma_2, const Scalar<DataVector>& psi,        \
      const Scalar<DataVector>& pi,                                            \
      const tnsr::i<DataVector, DIM(data), Frame::Inertial>& phi,              \
      const tnsr::i<DataVector, DIM(data), Frame::Inertial>&                   \
          unit_normal_one_form) noexcept;                                      \
  template Variables<                                                          \
      tmpl::list<ScalarWave::Tags::VPsi, ScalarWave::Tags::VZero<DIM(data)>,   \
                 ScalarWave::Tags::VPlus, ScalarWave::Tags::VMinus>>           \
  ScalarWave::characteristic_fields(                                           \
      const Scalar<DataVector>& gamma_2, const Scalar<DataVector>& psi,        \
      const Scalar<DataVector>& pi,                                            \
      const tnsr::i<DataVector, DIM(data), Frame::Inertial>& phi,              \
      const tnsr::i<DataVector, DIM(data), Frame::Inertial>&                   \
          unit_normal_one_form) noexcept;                                      \
  template struct ScalarWave::Tags::CharacteristicFieldsCompute<DIM(data)>;    \
  template void ScalarWave::evolved_fields_from_characteristic_fields(         \
      const gsl::not_null<Variables<tmpl::list<                                \
          ScalarWave::Psi, ScalarWave::Pi, ScalarWave::Phi<DIM(data)>>>*>      \
          evolved_fields,                                                      \
      const Scalar<DataVector>& gamma_2, const Scalar<DataVector>& v_psi,      \
      const tnsr::i<DataVector, DIM(data), Frame::Inertial>& v_zero,           \
      const Scalar<DataVector>& v_plus, const Scalar<DataVector>& v_minus,     \
      const tnsr::i<DataVector, DIM(data), Frame::Inertial>&                   \
          unit_normal_one_form) noexcept;                                      \
  template Variables<                                                          \
      tmpl::list<ScalarWave::Psi, ScalarWave::Pi, ScalarWave::Phi<DIM(data)>>> \
  ScalarWave::evolved_fields_from_characteristic_fields(                       \
      const Scalar<DataVector>& gamma_2, const Scalar<DataVector>& v_psi,      \
      const tnsr::i<DataVector, DIM(data), Frame::Inertial>& v_zero,           \
      const Scalar<DataVector>& v_plus, const Scalar<DataVector>& v_minus,     \
      const tnsr::i<DataVector, DIM(data), Frame::Inertial>&                   \
          unit_normal_one_form) noexcept;                                      \
  template struct ScalarWave::Tags::                                           \
      EvolvedFieldsFromCharacteristicFieldsCompute<DIM(data)>;

GENERATE_INSTANTIATIONS(INSTANTIATE, (1, 2, 3))

#undef INSTANTIATE
#undef DIM
