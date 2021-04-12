// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Evolution/Systems/GrMhd/GhValenciaDivClean/Characteristics.hpp"

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/EagerMath/DotProduct.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "DataStructures/Variables.hpp"
#include "Evolution/Systems/GeneralizedHarmonic/Characteristics.hpp"
#include "Evolution/Systems/GrMhd/ValenciaDivClean/Characteristics.hpp"
#include "Evolution/Systems/RelativisticEuler/Valencia/Characteristics.hpp"
#include "PointwiseFunctions/GeneralRelativity/IndexManipulation.hpp"
#include "PointwiseFunctions/GeneralRelativity/Tags.hpp"  // IWYU pragma: keep
#include "PointwiseFunctions/Hydro/Tags.hpp"              // IWYU pragma: keep
#include "Utilities/ConstantExpressions.hpp"
#include "Utilities/GenerateInstantiations.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/MakeWithValue.hpp"

// IWYU pragma: no_forward_declare EquationsOfState::EquationOfState
// IWYU pragma: no_forward_declare Tensor

namespace grmhd::GhValenciaDivClean {
template <size_t ThermodynamicDim>
void characteristic_speeds(
    const gsl::not_null<std::array<DataVector, 13>*> char_speeds,
    const Scalar<DataVector>& rest_mass_density,
    const Scalar<DataVector>& specific_internal_energy,
    const Scalar<DataVector>& specific_enthalpy,
    const tnsr::I<DataVector, 3, Frame::Inertial>& spatial_velocity,
    const Scalar<DataVector>& lorentz_factor,
    const tnsr::I<DataVector, 3, Frame::Inertial>& magnetic_field,
    const Scalar<DataVector>& lapse, const tnsr::I<DataVector, 3>& shift,
    const tnsr::ii<DataVector, 3, Frame::Inertial>& spatial_metric,
    const tnsr::i<DataVector, 3>& unit_normal,
    const EquationsOfState::EquationOfState<true, ThermodynamicDim>&
        equation_of_state,
    const Scalar<DataVector>& gh_constraint_gamma1) noexcept {
  const size_t num_grid_points = get(lapse).size();
  auto& achar_speeds = *char_speeds;
  for (size_t idx = 0; idx < 13; ++idx) {
    if (achar_speeds.at(idx).size() != num_grid_points) {
      achar_speeds.at(idx) = DataVector(num_grid_points);
    }
  }

  const auto char_speeds_valencia =
      grmhd::ValenciaDivClean::characteristic_speeds(
          rest_mass_density, specific_internal_energy, specific_enthalpy,
          spatial_velocity, lorentz_factor, magnetic_field, lapse, shift,
          spatial_metric, unit_normal, equation_of_state);
  for (size_t idx = 0; idx < 9; ++idx) {
    char_speeds->at(idx) = char_speeds_valencia.at(idx);
  }

  const auto char_speeds_gh =
      GeneralizedHarmonic::characteristic_speeds<3, Frame::Inertial>(
          gh_constraint_gamma1, lapse, shift, unit_normal);
  for (size_t idx = 0; idx < 4; ++idx) {
    char_speeds->at(9 + idx) = char_speeds_gh.at(idx);
  }
}

template <size_t ThermodynamicDim>
std::array<DataVector, 13> characteristic_speeds(
    const Scalar<DataVector>& rest_mass_density,
    const Scalar<DataVector>& specific_internal_energy,
    const Scalar<DataVector>& specific_enthalpy,
    const tnsr::I<DataVector, 3, Frame::Inertial>& spatial_velocity,
    const Scalar<DataVector>& lorentz_factor,
    const tnsr::I<DataVector, 3, Frame::Inertial>& magnetic_field,
    const Scalar<DataVector>& lapse, const tnsr::I<DataVector, 3>& shift,
    const tnsr::ii<DataVector, 3, Frame::Inertial>& spatial_metric,
    const tnsr::i<DataVector, 3>& unit_normal,
    const EquationsOfState::EquationOfState<true, ThermodynamicDim>&
        equation_of_state,
    const Scalar<DataVector>& gh_constraint_gamma1) noexcept {
  auto char_speeds = make_with_value<typename Tags::CharacteristicSpeeds::type>(
      get(lapse), 0.);
  characteristic_speeds(make_not_null(&char_speeds), rest_mass_density,
                        specific_internal_energy, specific_enthalpy,
                        spatial_velocity, lorentz_factor, magnetic_field, lapse,
                        shift, spatial_metric, unit_normal, equation_of_state,
                        gh_constraint_gamma1);
  return char_speeds;
}

void Tags::ComputeLargestCharacteristicSpeed::function(
    const gsl::not_null<double*> speed,
    const Scalar<DataVector>& gh_constraint_gamma1,
    const Scalar<DataVector>& lapse,
    const tnsr::I<DataVector, 3, Frame::Inertial>& shift,
    const tnsr::ii<DataVector, 3, Frame::Inertial>& spatial_metric) noexcept {
  double max_gh_speed = std::numeric_limits<double>::signaling_NaN();
  GeneralizedHarmonic::Tags::ComputeLargestCharacteristicSpeed<
      3, Frame::Inertial>::function(make_not_null(&max_gh_speed),
                                    gh_constraint_gamma1, lapse, shift,
                                    spatial_metric);
  // note: the 1.0 is an approximation valid only in flat spacetime. For
  // better CFL estimates, this should be updated to calculate the
  // characteristic speed based on the lapse and shift.
  *speed = std::max(max_gh_speed, 1.0);
}

#define GET_DIM(data) BOOST_PP_TUPLE_ELEM(0, data)

#define INSTANTIATION(r, data)                                              \
  template std::array<DataVector, 13> characteristic_speeds<GET_DIM(data)>( \
      const Scalar<DataVector>& rest_mass_density,                          \
      const Scalar<DataVector>& specific_internal_energy,                   \
      const Scalar<DataVector>& specific_enthalpy,                          \
      const tnsr::I<DataVector, 3, Frame::Inertial>& spatial_velocity,      \
      const Scalar<DataVector>& lorentz_factor,                             \
      const tnsr::I<DataVector, 3, Frame::Inertial>& magnetic_field,        \
      const Scalar<DataVector>& lapse, const tnsr::I<DataVector, 3>& shift, \
      const tnsr::ii<DataVector, 3, Frame::Inertial>& spatial_metric,       \
      const tnsr::i<DataVector, 3>& unit_normal,                            \
      const EquationsOfState::EquationOfState<true, GET_DIM(data)>&         \
          equation_of_state,                                                \
      const Scalar<DataVector>& gh_constraint_gamma1) noexcept;             \
  template void characteristic_speeds<GET_DIM(data)>(                       \
      const gsl::not_null<std::array<DataVector, 13>*> char_speeds,         \
      const Scalar<DataVector>& rest_mass_density,                          \
      const Scalar<DataVector>& specific_internal_energy,                   \
      const Scalar<DataVector>& specific_enthalpy,                          \
      const tnsr::I<DataVector, 3, Frame::Inertial>& spatial_velocity,      \
      const Scalar<DataVector>& lorentz_factor,                             \
      const tnsr::I<DataVector, 3, Frame::Inertial>& magnetic_field,        \
      const Scalar<DataVector>& lapse, const tnsr::I<DataVector, 3>& shift, \
      const tnsr::ii<DataVector, 3, Frame::Inertial>& spatial_metric,       \
      const tnsr::i<DataVector, 3>& unit_normal,                            \
      const EquationsOfState::EquationOfState<true, GET_DIM(data)>&         \
          equation_of_state,                                                \
      const Scalar<DataVector>& gh_constraint_gamma1) noexcept;

GENERATE_INSTANTIATIONS(INSTANTIATION, (1, 2))

#undef GET_DIM
#undef INSTANTIATION
}  // namespace grmhd::GhValenciaDivClean
