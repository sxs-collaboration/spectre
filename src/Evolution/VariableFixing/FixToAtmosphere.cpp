// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Evolution/VariableFixing/FixToAtmosphere.hpp"

#include <pup.h>  // IWYU pragma: keep

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Utilities/GenerateInstantiations.hpp"

// IWYU pragma: no_include <array>

// IWYU pragma: no_forward_declare EquationsOfState::EquationOfState
// IWYU pragma: no_forward_declare Tensor

namespace VariableFixing {

template <size_t Dim>
FixToAtmosphere<Dim>::FixToAtmosphere(const double density_of_atmosphere,
                                      const double density_cutoff,
                                      const double transition_density_cutoff,
                                      const double max_velocity_magnitude,
                                      const Options::Context& context)
    : density_of_atmosphere_(density_of_atmosphere),
      density_cutoff_(density_cutoff),
      transition_density_cutoff_(transition_density_cutoff),
      max_velocity_magnitude_(max_velocity_magnitude) {
  if (density_of_atmosphere_ > density_cutoff_) {
    PARSE_ERROR(context, "The cutoff density ("
                             << density_cutoff_
                             << ") must be greater than or equal to the "
                                "density value in the atmosphere ("
                             << density_of_atmosphere_ << ')');
  }
  if (transition_density_cutoff_ < density_of_atmosphere_ or
      transition_density_cutoff_ > 10.0 * density_of_atmosphere_) {
    PARSE_ERROR(context, "The transition density must be in ["
                             << density_of_atmosphere_ << ", "
                             << 10 * density_of_atmosphere_ << "], but is "
                             << transition_density_cutoff_);
  }
  if (transition_density_cutoff_ <= density_cutoff_) {
    PARSE_ERROR(context, "The transition density cutoff ("
                             << transition_density_cutoff_
                             << ") must be bigger than the density cutoff ("
                             << density_cutoff_ << ")");
  }
}

// clang-tidy: google-runtime-references
template <size_t Dim>
void FixToAtmosphere<Dim>::pup(PUP::er& p) {  // NOLINT
  p | density_of_atmosphere_;
  p | density_cutoff_;
  p | transition_density_cutoff_;
  p | max_velocity_magnitude_;
}

template <size_t Dim>
template <size_t ThermodynamicDim>
void FixToAtmosphere<Dim>::operator()(
    const gsl::not_null<Scalar<DataVector>*> rest_mass_density,
    const gsl::not_null<Scalar<DataVector>*> specific_internal_energy,
    const gsl::not_null<tnsr::I<DataVector, Dim, Frame::Inertial>*>
        spatial_velocity,
    const gsl::not_null<Scalar<DataVector>*> lorentz_factor,
    const gsl::not_null<Scalar<DataVector>*> pressure,
    const gsl::not_null<Scalar<DataVector>*> specific_enthalpy,
    const tnsr::ii<DataVector, Dim, Frame::Inertial>& spatial_metric,
    const EquationsOfState::EquationOfState<true, ThermodynamicDim>&
        equation_of_state) const {
  for (size_t i = 0; i < rest_mass_density->get().size(); i++) {
    if (UNLIKELY(rest_mass_density->get()[i] < density_cutoff_)) {
      set_density_to_atmosphere(rest_mass_density, specific_internal_energy,
                                pressure, specific_enthalpy, equation_of_state,
                                i);
      for (size_t d = 0; d < Dim; ++d) {
        spatial_velocity->get(d)[i] = 0.0;
      }
      lorentz_factor->get()[i] = 1.0;
    } else if (UNLIKELY(rest_mass_density->get()[i] <
                        transition_density_cutoff_)) {
      set_to_magnetic_free_transition(rest_mass_density, spatial_velocity,
                                      lorentz_factor, spatial_metric, i);
    }
  }
}

template <size_t Dim>
template <size_t ThermodynamicDim>
void FixToAtmosphere<Dim>::set_density_to_atmosphere(
    const gsl::not_null<Scalar<DataVector>*> rest_mass_density,
    const gsl::not_null<Scalar<DataVector>*> specific_internal_energy,
    const gsl::not_null<Scalar<DataVector>*> pressure,
    const gsl::not_null<Scalar<DataVector>*> specific_enthalpy,
    const EquationsOfState::EquationOfState<true, ThermodynamicDim>&
        equation_of_state,
    const size_t grid_index) const {
  rest_mass_density->get()[grid_index] = density_of_atmosphere_;
  Scalar<double> atmosphere_density{density_of_atmosphere_};
  if constexpr (ThermodynamicDim == 1) {
    pressure->get()[grid_index] =
        get(equation_of_state.pressure_from_density(atmosphere_density));
    specific_internal_energy->get()[grid_index] =
        get(equation_of_state.specific_internal_energy_from_density(
            atmosphere_density));
    specific_enthalpy->get()[grid_index] = get(
        equation_of_state.specific_enthalpy_from_density(atmosphere_density));
  } else if constexpr (ThermodynamicDim == 2) {
    Scalar<double> atmosphere_energy{0.0};
    pressure->get()[grid_index] =
        get(equation_of_state.pressure_from_density_and_energy(
            atmosphere_density, atmosphere_energy));
    specific_internal_energy->get()[grid_index] = get(atmosphere_energy);
    specific_enthalpy->get()[grid_index] =
        get(equation_of_state.specific_enthalpy_from_density_and_energy(
            atmosphere_density, atmosphere_energy));
  }
}

template <size_t Dim>
void FixToAtmosphere<Dim>::set_to_magnetic_free_transition(
    const gsl::not_null<Scalar<DataVector>*> rest_mass_density,
    const gsl::not_null<tnsr::I<DataVector, Dim, Frame::Inertial>*>
        spatial_velocity,
    const gsl::not_null<Scalar<DataVector>*> lorentz_factor,
    const tnsr::ii<DataVector, Dim, Frame::Inertial>& spatial_metric,
    const size_t grid_index) const {
  double magnitude_of_velocity = 0.0;
  for (size_t j = 0; j < Dim; ++j) {
    magnitude_of_velocity += spatial_velocity->get(j)[grid_index] *
                             spatial_velocity->get(j)[grid_index] *
                             spatial_metric.get(j, j)[grid_index];
    for (size_t k = j + 1; k < Dim; ++k) {
      magnitude_of_velocity += 2.0 * spatial_velocity->get(j)[grid_index] *
                               spatial_velocity->get(k)[grid_index] *
                               spatial_metric.get(j, k)[grid_index];
    }
  }
  magnitude_of_velocity = sqrt(magnitude_of_velocity);
  const double scale_factor =
      (get(*rest_mass_density)[grid_index] - density_cutoff_) /
      (transition_density_cutoff_ - density_cutoff_);
  if (const double max_mag_of_velocity = scale_factor * max_velocity_magnitude_;
      magnitude_of_velocity > max_mag_of_velocity) {
    for (size_t j = 0; j < Dim; ++j) {
      spatial_velocity->get(j)[grid_index] *=
          max_mag_of_velocity / magnitude_of_velocity;
    }
    get(*lorentz_factor)[grid_index] =
        1.0 / sqrt(1.0 - max_mag_of_velocity * max_mag_of_velocity);
  }
}

template <size_t Dim>
bool operator==(const FixToAtmosphere<Dim>& lhs,
                const FixToAtmosphere<Dim>& rhs) {
  return lhs.density_of_atmosphere_ == rhs.density_of_atmosphere_ and
         lhs.density_cutoff_ == rhs.density_cutoff_ and
         lhs.transition_density_cutoff_ == rhs.transition_density_cutoff_ and
         lhs.max_velocity_magnitude_ == rhs.max_velocity_magnitude_;
}

template <size_t Dim>
bool operator!=(const FixToAtmosphere<Dim>& lhs,
                const FixToAtmosphere<Dim>& rhs) {
  return not(lhs == rhs);
}

#define DIM(data) BOOST_PP_TUPLE_ELEM(0, data)
#define THERMO_DIM(data) BOOST_PP_TUPLE_ELEM(1, data)

#define INSTANTIATION(r, data)                                     \
  template class FixToAtmosphere<DIM(data)>;                       \
  template bool operator==(const FixToAtmosphere<DIM(data)>& lhs,  \
                           const FixToAtmosphere<DIM(data)>& rhs); \
  template bool operator!=(const FixToAtmosphere<DIM(data)>& lhs,  \
                           const FixToAtmosphere<DIM(data)>& rhs);

GENERATE_INSTANTIATIONS(INSTANTIATION, (1, 2, 3))

#undef INSTANTIATION

#define INSTANTIATION(r, data)                                                \
  template void FixToAtmosphere<DIM(data)>::operator()(                       \
      const gsl::not_null<Scalar<DataVector>*> rest_mass_density,             \
      const gsl::not_null<Scalar<DataVector>*> specific_internal_energy,      \
      const gsl::not_null<tnsr::I<DataVector, DIM(data), Frame::Inertial>*>   \
          spatial_velocity,                                                   \
      const gsl::not_null<Scalar<DataVector>*> lorentz_factor,                \
      const gsl::not_null<Scalar<DataVector>*> pressure,                      \
      const gsl::not_null<Scalar<DataVector>*> specific_enthalpy,             \
      const tnsr::ii<DataVector, DIM(data), Frame::Inertial>& spatial_metric, \
      const EquationsOfState::EquationOfState<true, THERMO_DIM(data)>&        \
          equation_of_state) const;

GENERATE_INSTANTIATIONS(INSTANTIATION, (1, 2, 3), (1, 2))

#undef DIM
#undef THERMO_DIM
#undef INSTANTIATION

}  // namespace VariableFixing
