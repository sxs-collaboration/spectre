// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Evolution/Systems/NewtonianMhd/Limiters/CharacteristicHelpers.hpp"

#include <array>
#include <boost/functional/hash.hpp>
#include <cstddef>
#include <limits>
#include <string>
#include <type_traits>
#include <unordered_map>
#include <utility>

#include "DataStructures/TaggedTuple.hpp"
#include "DataStructures/Tags.hpp"
#include "DataStructures/Tensor/EagerMath/DotProduct.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "DataStructures/Variables.hpp"
#include "Domain/Structure/Element.hpp"
#include "Domain/Tags.hpp"
#include "Evolution/Systems/NewtonianMhd/Characteristics.hpp"
#include "Evolution/Systems/NewtonianMhd/SoundSpeedSquared.hpp"
#include "Evolution/Systems/NewtonianMhd/Tags.hpp"
#include "NumericalAlgorithms/Spectral/Mesh.hpp"
#include "Utilities/ErrorHandling/Assert.hpp"
#include "Utilities/GenerateInstantiations.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/TMPL.hpp"

namespace NewtonianMhd::Limiters {

template <size_t VolumeDim>
std::pair<Matrix, Matrix> right_and_left_eigenvectors(
    const Scalar<double>& mean_density,
    const tnsr::I<double, VolumeDim>& mean_momentum,
    const Scalar<double>& mean_energy,
    const EquationsOfState::EquationOfState<false, 2>& equation_of_state,
    const tnsr::i<double, VolumeDim>& unit_vector,
    const bool compute_char_transformation_numerically) {
  // Compute fluid primitives from mean conserved state
  const auto velocity = [&mean_density, &mean_momentum]() {
    auto result = mean_momentum;
    for (size_t i = 0; i < VolumeDim; ++i) {
      result.get(i) /= get(mean_density);
    }
    return result;
  }();
  const auto specific_internal_energy = [&mean_density, &mean_energy,
                                         &mean_momentum]() {
    auto result = mean_energy;
    get(result) /= get(mean_density);
    get(result) -= 0.5 * get(dot_product(mean_momentum, mean_momentum)) /
                   square(get(mean_density));
    return result;
  }();

  Scalar<double> pressure{};
  Scalar<double> kappa_over_density{};
  pressure = equation_of_state.pressure_from_density_and_energy(
      mean_density, specific_internal_energy);
  get(kappa_over_density) =
      get(equation_of_state
              .kappa_times_p_over_rho_squared_from_density_and_energy(
                  mean_density, specific_internal_energy)) *
      get(mean_density) / get(pressure);

  const Scalar<double> specific_enthalpy{
      {{(get(mean_energy) + get(pressure)) / get(mean_density)}}};
  const Scalar<double> sound_speed_squared = NewtonianMhd::sound_speed_squared(
      mean_density, specific_internal_energy, equation_of_state);

  if (compute_char_transformation_numerically) {
    return numerical_eigensystem(velocity, sound_speed_squared,
                                 specific_enthalpy, kappa_over_density,
                                 unit_vector)
        .second;
  } else {
    return std::make_pair(right_eigenvectors<VolumeDim>(
                              velocity, sound_speed_squared, specific_enthalpy,
                              kappa_over_density, unit_vector),
                          left_eigenvectors<VolumeDim>(
                              velocity, sound_speed_squared, specific_enthalpy,
                              kappa_over_density, unit_vector));
  }
}

template <size_t VolumeDim>
void characteristic_fields(
    const gsl::not_null<tuples::TaggedTuple<
        ::Tags::Mean<NewtonianMhd::Tags::VMinus>,
        ::Tags::Mean<NewtonianMhd::Tags::VMomentum<VolumeDim>>,
        ::Tags::Mean<NewtonianMhd::Tags::VPlus>>*>
        char_means,
    const tuples::TaggedTuple<
        ::Tags::Mean<NewtonianMhd::Tags::MassDensityCons>,
        ::Tags::Mean<NewtonianMhd::Tags::MomentumDensity<VolumeDim>>,
        ::Tags::Mean<NewtonianMhd::Tags::EnergyDensity>>& cons_means,
    const Matrix& left) {
  auto& char_v_minus =
      get<::Tags::Mean<NewtonianMhd::Tags::VMinus>>(*char_means);
  auto& char_v_momentum =
      get<::Tags::Mean<NewtonianMhd::Tags::VMomentum<VolumeDim>>>(*char_means);
  auto& char_v_plus = get<::Tags::Mean<NewtonianMhd::Tags::VPlus>>(*char_means);

  const auto& cons_mass_density =
      get<::Tags::Mean<NewtonianMhd::Tags::MassDensityCons>>(cons_means);
  const auto& cons_momentum_density =
      get<::Tags::Mean<NewtonianMhd::Tags::MomentumDensity<VolumeDim>>>(
          cons_means);
  const auto& cons_energy_density =
      get<::Tags::Mean<NewtonianMhd::Tags::EnergyDensity>>(cons_means);

  get(char_v_minus) = left(0, 0) * get(cons_mass_density);
  for (size_t j = 0; j < VolumeDim; ++j) {
    char_v_momentum.get(j) = left(j + 1, 0) * get(cons_mass_density);
  }
  get(char_v_plus) = left(VolumeDim + 1, 0) * get(cons_mass_density);

  for (size_t i = 0; i < VolumeDim; ++i) {
    get(char_v_minus) += left(0, i + 1) * cons_momentum_density.get(i);
    for (size_t j = 0; j < VolumeDim; ++j) {
      char_v_momentum.get(j) +=
          left(j + 1, i + 1) * cons_momentum_density.get(i);
    }
    get(char_v_plus) +=
        left(VolumeDim + 1, i + 1) * cons_momentum_density.get(i);
  }

  get(char_v_minus) += left(0, VolumeDim + 1) * get(cons_energy_density);
  for (size_t j = 0; j < VolumeDim; ++j) {
    char_v_momentum.get(j) +=
        left(j + 1, VolumeDim + 1) * get(cons_energy_density);
  }
  get(char_v_plus) +=
      left(VolumeDim + 1, VolumeDim + 1) * get(cons_energy_density);
}

template <size_t VolumeDim>
void characteristic_fields(
    const gsl::not_null<Scalar<DataVector>*> char_v_minus,
    const gsl::not_null<tnsr::I<DataVector, VolumeDim>*> char_v_momentum,
    const gsl::not_null<Scalar<DataVector>*> char_v_plus,
    const Scalar<DataVector>& cons_mass_density,
    const tnsr::I<DataVector, VolumeDim>& cons_momentum_density,
    const Scalar<DataVector>& cons_energy_density, const Matrix& left) {
  get(*char_v_minus) = left(0, 0) * get(cons_mass_density);
  for (size_t j = 0; j < VolumeDim; ++j) {
    char_v_momentum->get(j) = left(j + 1, 0) * get(cons_mass_density);
  }
  get(*char_v_plus) = left(VolumeDim + 1, 0) * get(cons_mass_density);

  for (size_t i = 0; i < VolumeDim; ++i) {
    get(*char_v_minus) += left(0, i + 1) * cons_momentum_density.get(i);
    for (size_t j = 0; j < VolumeDim; ++j) {
      char_v_momentum->get(j) +=
          left(j + 1, i + 1) * cons_momentum_density.get(i);
    }
    get(*char_v_plus) +=
        left(VolumeDim + 1, i + 1) * cons_momentum_density.get(i);
  }

  get(*char_v_minus) += left(0, VolumeDim + 1) * get(cons_energy_density);
  for (size_t j = 0; j < VolumeDim; ++j) {
    char_v_momentum->get(j) +=
        left(j + 1, VolumeDim + 1) * get(cons_energy_density);
  }
  get(*char_v_plus) +=
      left(VolumeDim + 1, VolumeDim + 1) * get(cons_energy_density);
}

template <size_t VolumeDim>
void characteristic_fields(
    const gsl::not_null<Variables<tmpl::list<
        NewtonianMhd::Tags::VMinus, NewtonianMhd::Tags::VMomentum<VolumeDim>,
        NewtonianMhd::Tags::VPlus>>*>
        char_vars,
    const Variables<tmpl::list<NewtonianMhd::Tags::MassDensityCons,
                               NewtonianMhd::Tags::MomentumDensity<VolumeDim>,
                               NewtonianMhd::Tags::EnergyDensity>>& cons_vars,
    const Matrix& left) {
  characteristic_fields(
      make_not_null(&get<NewtonianMhd::Tags::VMinus>(*char_vars)),
      make_not_null(&get<NewtonianMhd::Tags::VMomentum<VolumeDim>>(*char_vars)),
      make_not_null(&get<NewtonianMhd::Tags::VPlus>(*char_vars)),
      get<NewtonianMhd::Tags::MassDensityCons>(cons_vars),
      get<NewtonianMhd::Tags::MomentumDensity<VolumeDim>>(cons_vars),
      get<NewtonianMhd::Tags::EnergyDensity>(cons_vars), left);
}

template <size_t VolumeDim>
void conserved_fields_from_characteristic_fields(
    const gsl::not_null<Scalar<DataVector>*> cons_mass_density,
    const gsl::not_null<tnsr::I<DataVector, VolumeDim>*> cons_momentum_density,
    const gsl::not_null<Scalar<DataVector>*> cons_energy_density,
    const Scalar<DataVector>& char_v_minus,
    const tnsr::I<DataVector, VolumeDim>& char_v_momentum,
    const Scalar<DataVector>& char_v_plus, const Matrix& right) {
  get(*cons_mass_density) = right(0, 0) * get(char_v_minus);
  for (size_t j = 0; j < VolumeDim; ++j) {
    cons_momentum_density->get(j) = right(j + 1, 0) * get(char_v_minus);
  }
  get(*cons_energy_density) = right(VolumeDim + 1, 0) * get(char_v_minus);

  for (size_t i = 0; i < VolumeDim; ++i) {
    get(*cons_mass_density) += right(0, i + 1) * char_v_momentum.get(i);
    for (size_t j = 0; j < VolumeDim; ++j) {
      cons_momentum_density->get(j) +=
          right(j + 1, i + 1) * char_v_momentum.get(i);
    }
    get(*cons_energy_density) +=
        right(VolumeDim + 1, i + 1) * char_v_momentum.get(i);
  }

  get(*cons_mass_density) += right(0, VolumeDim + 1) * get(char_v_plus);
  for (size_t j = 0; j < VolumeDim; ++j) {
    cons_momentum_density->get(j) +=
        right(j + 1, VolumeDim + 1) * get(char_v_plus);
  }
  get(*cons_energy_density) +=
      right(VolumeDim + 1, VolumeDim + 1) * get(char_v_plus);
}

#define DIM(data) BOOST_PP_TUPLE_ELEM(0, data)

#define INSTANTIATE(_, data)                                      \
  template std::pair<Matrix, Matrix> right_and_left_eigenvectors( \
      const Scalar<double>&, const tnsr::I<double, DIM(data)>&,   \
      const Scalar<double>&,                                      \
      const EquationsOfState::EquationOfState<false, 2>&,         \
      const tnsr::i<double, DIM(data)>&, const bool);

GENERATE_INSTANTIATIONS(INSTANTIATE, (1, 2, 3))

#undef INSTANTIATE

#define INSTANTIATE(_, data)                                               \
  template void characteristic_fields(                                     \
      const gsl::not_null<tuples::TaggedTuple<                             \
          ::Tags::Mean<NewtonianMhd::Tags::VMinus>,                        \
          ::Tags::Mean<NewtonianMhd::Tags::VMomentum<DIM(data)>>,          \
          ::Tags::Mean<NewtonianMhd::Tags::VPlus>>*>,                      \
      const tuples::TaggedTuple<                                           \
          ::Tags::Mean<NewtonianMhd::Tags::MassDensityCons>,               \
          ::Tags::Mean<NewtonianMhd::Tags::MomentumDensity<DIM(data)>>,    \
          ::Tags::Mean<NewtonianMhd::Tags::EnergyDensity>>&,               \
      const Matrix&);                                                      \
  template void characteristic_fields(                                     \
      const gsl::not_null<Scalar<DataVector>*>,                            \
      const gsl::not_null<tnsr::I<DataVector, DIM(data)>*>,                \
      const gsl::not_null<Scalar<DataVector>*>, const Scalar<DataVector>&, \
      const tnsr::I<DataVector, DIM(data)>&, const Scalar<DataVector>&,    \
      const Matrix&);                                                      \
  template void characteristic_fields(                                     \
      const gsl::not_null<                                                 \
          Variables<tmpl::list<NewtonianMhd::Tags::VMinus,                 \
                               NewtonianMhd::Tags::VMomentum<DIM(data)>,   \
                               NewtonianMhd::Tags::VPlus>>*>,              \
      const Variables<                                                     \
          tmpl::list<NewtonianMhd::Tags::MassDensityCons,                  \
                     NewtonianMhd::Tags::MomentumDensity<DIM(data)>,       \
                     NewtonianMhd::Tags::EnergyDensity>>&,                 \
      const Matrix&);                                                      \
  template void conserved_fields_from_characteristic_fields(               \
      const gsl::not_null<Scalar<DataVector>*>,                            \
      const gsl::not_null<tnsr::I<DataVector, DIM(data)>*>,                \
      const gsl::not_null<Scalar<DataVector>*>, const Scalar<DataVector>&, \
      const tnsr::I<DataVector, DIM(data)>&, const Scalar<DataVector>&,    \
      const Matrix&);

GENERATE_INSTANTIATIONS(INSTANTIATE, (1, 2, 3))

#undef DIM
#undef INSTANTIATE

}  // namespace NewtonianMhd::Limiters
