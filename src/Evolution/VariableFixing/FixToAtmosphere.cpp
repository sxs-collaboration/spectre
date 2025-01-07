// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Evolution/VariableFixing/FixToAtmosphere.hpp"

#include <limits>
#include <optional>
#include <ostream>
#include <pup.h>
#include <string>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "NumericalAlgorithms/RootFinding/TOMS748.hpp"
#include "Options/Options.hpp"
#include "Options/ParseError.hpp"
#include "Options/ParseOptions.hpp"
#include "Utilities/ErrorHandling/Error.hpp"
#include "Utilities/GenerateInstantiations.hpp"
#include "Utilities/GetOutput.hpp"
#include "Utilities/Serialization/PupStlCpp17.hpp"

namespace VariableFixing {
std::ostream& operator<<(std::ostream& os,
                         const FixReconstructedStateToAtmosphere& t) {
  switch (t) {
    case FixReconstructedStateToAtmosphere::Always:
      return os << "Always";
    case FixReconstructedStateToAtmosphere::AtDgFdInterfaceOnly:
      return os << "AtDgFdInterfaceOnly";
    case FixReconstructedStateToAtmosphere::OnFdOnly:
      return os << "OnFdOnly";
    case FixReconstructedStateToAtmosphere::Never:
      return os << "Never";
    default:
      ERROR("Unknown floating point type, must be Float or Double");
  }
}
}  // namespace VariableFixing

template <>
VariableFixing::FixReconstructedStateToAtmosphere
Options::create_from_yaml<VariableFixing::FixReconstructedStateToAtmosphere>::
    create<void>(const Options::Option& options) {
  const auto type_read = options.parse_as<std::string>();
  for (const auto t :
       {VariableFixing::FixReconstructedStateToAtmosphere::Always,
        VariableFixing::FixReconstructedStateToAtmosphere::AtDgFdInterfaceOnly,
        VariableFixing::FixReconstructedStateToAtmosphere::OnFdOnly,
        VariableFixing::FixReconstructedStateToAtmosphere::Never}) {
    if (type_read == get_output(t)) {
      return t;
    }
  }
  PARSE_ERROR(options.context(),
              "Failed to convert \""
                  << type_read << "\" to FixReconstructedStateToAtmosphere.");
}

namespace VariableFixing {
template <size_t Dim>
FixToAtmosphere<Dim>::FixToAtmosphere(
    const double density_of_atmosphere, const double density_cutoff,
    const std::optional<VelocityLimitingOptions> velocity_limiting,
    const std::optional<KappaLimitingOptions> kappa_limiting,
    const Options::Context& context)
    : density_of_atmosphere_(density_of_atmosphere),
      density_cutoff_(density_cutoff),
      velocity_limiting_(velocity_limiting),
      kappa_limiting_(kappa_limiting) {
  if (density_of_atmosphere_ > density_cutoff_) {
    PARSE_ERROR(context, "The cutoff density ("
                             << density_cutoff_
                             << ") must be greater than or equal to the "
                                "density value in the atmosphere ("
                             << density_of_atmosphere_ << ')');
  }

  if (velocity_limiting_.has_value()) {
    if (velocity_limiting_->atmosphere_max_velocity < 0.0) {
      PARSE_ERROR(context,
                  "The AtmosphereMaxVelocity must be non-negative but is "
                      << velocity_limiting_->atmosphere_max_velocity);
    }
    if (velocity_limiting_->near_atmosphere_max_velocity < 0.0) {
      PARSE_ERROR(context,
                  "The NearAtmosphereMaxVelocity must be non-negative but is "
                      << velocity_limiting_->near_atmosphere_max_velocity);
    }
    if (velocity_limiting_->atmosphere_max_velocity >
        velocity_limiting_->near_atmosphere_max_velocity) {
      PARSE_ERROR(context,
                  "The AtmosphereMaxVelocity ("
                      << velocity_limiting_->atmosphere_max_velocity
                      << ") must be smaller NearAtmosphereMaxVelocity ("
                      << velocity_limiting_->near_atmosphere_max_velocity
                      << ").");
    }
    if (velocity_limiting_->atmosphere_density_cutoff < 0.0) {
      PARSE_ERROR(context,
                  "The AtmosphereDensityCutoff must be non-negative but is "
                      << velocity_limiting_->atmosphere_density_cutoff);
    }
    if (velocity_limiting_->transition_density_bound < 0.0) {
      PARSE_ERROR(context,
                  "The TransitionDensityBound must be non-negative but is "
                      << velocity_limiting_->transition_density_bound);
    }
    if (velocity_limiting_->atmosphere_density_cutoff <
        density_of_atmosphere_) {
      PARSE_ERROR(
          context,
          "The AtmosphereDensityCutoff ("
              << velocity_limiting_->atmosphere_density_cutoff
              << ") must be greater than or equal to the DensityOfAtmosphere ("
              << density_of_atmosphere_ << ").");
    }
    if (velocity_limiting_->transition_density_bound <
        velocity_limiting_->atmosphere_density_cutoff) {
      PARSE_ERROR(context, "The TransitionDensityBound ("
                               << velocity_limiting_->transition_density_bound
                               << ") must be greater than or equal to the "
                                  "AtmosphereDensityCutoff ("
                               << velocity_limiting_->atmosphere_density_cutoff
                               << ").");
    }
  }

  if (kappa_limiting_.has_value()) {
    if (kappa_limiting_->density_lower_bound >
        kappa_limiting_->density_upper_bound) {
      PARSE_ERROR(context,
                  "The DensityLowerBound ("
                      << kappa_limiting_->density_lower_bound
                      << ") must be less than or equal to DensityUpperBound("
                      << kappa_limiting_->density_upper_bound << ").");
    }
    if (kappa_limiting_->min_temperature.has_value() and
        kappa_limiting_->min_temperature.value() < 0.0) {
      PARSE_ERROR(context, "The MinTemperature must be non-negative but is "
                               << kappa_limiting_->min_temperature.value());
    }
  }
}

template <size_t Dim>
// NOLINTNEXTLINE(google-runtime-references)
void FixToAtmosphere<Dim>::pup(PUP::er& p) {
  p | density_of_atmosphere_;
  p | density_cutoff_;
  p | velocity_limiting_;
  p | kappa_limiting_;
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
    const gsl::not_null<Scalar<DataVector>*> temperature,
    const Scalar<DataVector>& electron_fraction,
    const tnsr::ii<DataVector, Dim, Frame::Inertial>& spatial_metric,
    const EquationsOfState::EquationOfState<true, ThermodynamicDim>&
        equation_of_state) const {
  for (size_t i = 0; i < rest_mass_density->get().size(); i++) {
    if (UNLIKELY(rest_mass_density->get()[i] < density_cutoff_)) {
      set_density_to_atmosphere(rest_mass_density, specific_internal_energy,
                                temperature, pressure, electron_fraction,
                                equation_of_state, i);
    }
    if (velocity_limiting_.has_value()) {
      apply_velocity_limit(spatial_velocity, lorentz_factor, *rest_mass_density,
                           spatial_metric, i);
    }

    // For 2D & 3D EoS, we also need to limit the temperature / energy
    if constexpr (ThermodynamicDim > 1) {
      bool changed_temperature = false;
      if (const double min_temperature =
              equation_of_state.temperature_lower_bound();
          get(*temperature)[i] < min_temperature) {
        get(*temperature)[i] = min_temperature;
        changed_temperature = true;
      }

      // We probably need a better maximum temperature as well, but this is not
      // as well defined. To be discussed once implementation needs improvement.
      if (const double max_temperature =
              equation_of_state.temperature_upper_bound();
          get(*temperature)[i] > max_temperature) {
        get(*temperature)[i] = max_temperature;
        changed_temperature = true;
      }

      if (kappa_limiting_.has_value()) {
        changed_temperature |=
            apply_kappa_limit(temperature, *rest_mass_density,
                              electron_fraction, equation_of_state, i);
      }

      if (changed_temperature) {
        if constexpr (ThermodynamicDim == 2) {
          specific_internal_energy->get()[i] =
              get(equation_of_state
                      .specific_internal_energy_from_density_and_temperature(
                          Scalar<double>{rest_mass_density->get()[i]},
                          Scalar<double>{get(*temperature)[i]}));
          pressure->get()[i] =
              get(equation_of_state.pressure_from_density_and_energy(
                  Scalar<double>{rest_mass_density->get()[i]},
                  Scalar<double>{specific_internal_energy->get()[i]}));
        } else {
          specific_internal_energy->get()[i] =
              get(equation_of_state
                      .specific_internal_energy_from_density_and_temperature(
                          Scalar<double>{rest_mass_density->get()[i]},
                          Scalar<double>{get(*temperature)[i]},
                          Scalar<double>{get(electron_fraction)[i]}));
          pressure->get()[i] =
              get(equation_of_state.pressure_from_density_and_temperature(
                  Scalar<double>{rest_mass_density->get()[i]},
                  Scalar<double>{temperature->get()[i]},
                  Scalar<double>{get(electron_fraction)[i]}));
        }
      }
    }
  }
}

template <size_t Dim>
template <size_t ThermodynamicDim>
void FixToAtmosphere<Dim>::set_density_to_atmosphere(
    const gsl::not_null<Scalar<DataVector>*> rest_mass_density,
    const gsl::not_null<Scalar<DataVector>*> specific_internal_energy,
    const gsl::not_null<Scalar<DataVector>*> temperature,
    const gsl::not_null<Scalar<DataVector>*> pressure,
    const Scalar<DataVector>& electron_fraction,
    const EquationsOfState::EquationOfState<true, ThermodynamicDim>&
        equation_of_state,
    const size_t grid_index) const {
  const Scalar<double> atmosphere_density{density_of_atmosphere_};
  rest_mass_density->get()[grid_index] = get(atmosphere_density);
  get(*temperature)[grid_index] = equation_of_state.temperature_lower_bound();

  if constexpr (ThermodynamicDim == 1) {
    pressure->get()[grid_index] =
        get(equation_of_state.pressure_from_density(atmosphere_density));
    specific_internal_energy->get()[grid_index] =
        get(equation_of_state.specific_internal_energy_from_density(
            atmosphere_density));
  } else {
    const Scalar<double> atmosphere_temperature{get(*temperature)[grid_index]};
    if constexpr (ThermodynamicDim == 2) {
      specific_internal_energy->get()[grid_index] =
          get(equation_of_state
                  .specific_internal_energy_from_density_and_temperature(
                      atmosphere_density, atmosphere_temperature));
      pressure->get()[grid_index] =
          get(equation_of_state.pressure_from_density_and_energy(
              atmosphere_density,
              Scalar<double>{specific_internal_energy->get()[grid_index]}));
    } else {
      specific_internal_energy->get()[grid_index] =
          get(equation_of_state
                  .specific_internal_energy_from_density_and_temperature(
                      Scalar<double>{get(*rest_mass_density)[grid_index]},
                      Scalar<double>{get(*temperature)[grid_index]},
                      Scalar<double>{get(electron_fraction)[grid_index]}));
      pressure->get()[grid_index] =
          get(equation_of_state.pressure_from_density_and_temperature(
              Scalar<double>{get(*rest_mass_density)[grid_index]},
              Scalar<double>{get(*temperature)[grid_index]},
              Scalar<double>{get(electron_fraction)[grid_index]}));
    }
  }
}

template <size_t Dim>
void FixToAtmosphere<Dim>::apply_velocity_limit(
    const gsl::not_null<tnsr::I<DataVector, Dim, Frame::Inertial>*>
        spatial_velocity,
    const gsl::not_null<Scalar<DataVector>*> lorentz_factor,
    const Scalar<DataVector>& rest_mass_density,
    const tnsr::ii<DataVector, Dim, Frame::Inertial>& spatial_metric,
    const size_t grid_index) const {
  if (get(rest_mass_density)[grid_index] >
      velocity_limiting_->transition_density_bound) {
    return;
  }

  if (get(rest_mass_density)[grid_index] <
      velocity_limiting_->atmosphere_density_cutoff) {
    for (size_t i = 0; i < Dim; ++i) {
      spatial_velocity->get(i)[grid_index] =
          velocity_limiting_->atmosphere_max_velocity;
    }
    if (LIKELY(velocity_limiting_->atmosphere_max_velocity == 0.0)) {
      get(*lorentz_factor)[grid_index] = 1.0;
      return;
    }
  }

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
  if (UNLIKELY(get(rest_mass_density)[grid_index] <
               velocity_limiting_->atmosphere_density_cutoff)) {
    // Note: magnitude_of_velocity is squared still.
    get(*lorentz_factor)[grid_index] = 1.0 / sqrt(1.0 - magnitude_of_velocity);
    return;
  }
  magnitude_of_velocity = sqrt(magnitude_of_velocity);
  const double scale_factor = (get(rest_mass_density)[grid_index] -
                               velocity_limiting_->atmosphere_density_cutoff) /
                              (velocity_limiting_->transition_density_bound -
                               velocity_limiting_->atmosphere_density_cutoff);
  if (const double max_mag_of_velocity =
          scale_factor * velocity_limiting_->near_atmosphere_max_velocity;
      magnitude_of_velocity > max_mag_of_velocity) {
    const double one_over_max_mag_of_velocity = 1.0 / magnitude_of_velocity;
    for (size_t j = 0; j < Dim; ++j) {
      spatial_velocity->get(j)[grid_index] *=
          max_mag_of_velocity * one_over_max_mag_of_velocity;
    }
    get(*lorentz_factor)[grid_index] =
        1.0 / sqrt(1.0 - max_mag_of_velocity * max_mag_of_velocity);
  }
}

template <size_t Dim>
template <size_t ThermodynamicDim>
bool FixToAtmosphere<Dim>::apply_kappa_limit(
    const gsl::not_null<Scalar<DataVector>*> temperature,
    const Scalar<DataVector>& rest_mass_density,
    const Scalar<DataVector>& electron_fraction,
    const EquationsOfState::EquationOfState<true, ThermodynamicDim>&
        equation_of_state,
    const size_t grid_index) const {
  const KappaLimitingOptions& opts = kappa_limiting_.value();
  double& temp = temperature->get()[grid_index];
  const double& density = rest_mass_density.get()[grid_index];
  const double& y_e = electron_fraction.get()[grid_index];
  const double min_temperature = kappa_limiting_->min_temperature.value_or(
      equation_of_state.temperature_lower_bound());

  const auto get_pressure = [&density, &equation_of_state,
                             &y_e](const double local_temperature) -> double {
    if constexpr (ThermodynamicDim == 2) {
      (void)y_e;
      return get(equation_of_state.pressure_from_density_and_energy(
          Scalar<double>{density},
          equation_of_state
              .specific_internal_energy_from_density_and_temperature(
                  Scalar<double>{density}, Scalar<double>{local_temperature})));
    } else {
      return get(equation_of_state.pressure_from_density_and_temperature(
          Scalar<double>{density}, Scalar<double>{local_temperature},
          Scalar<double>{y_e}));
    }
  };

  const auto impl = [&get_pressure, &min_temperature, &temp,
                     &y_e](const double local_kappa_max) -> bool {
    const double p_temp = get_pressure(temp);
    const double p_min = get_pressure(min_temperature);
    const double p_max = p_min * local_kappa_max;
    if (p_temp > p_max) {
      if (UNLIKELY((p_temp - p_max) * (p_min - p_max) > 0.0)) {
        ERROR(
            "The root for the pressure function while applying the kappa "
            "limiting strategy was not bound.\n"
            "T_min="
            << min_temperature << "\np_min=" << p_min << "\nT=" << temp
            << "\np_temp=" << p_temp << "\np_max=" << p_max << "\nYe=" << y_e);
      }
      if (temp - min_temperature > 1.0e-13) {
        temp = RootFinder::toms748(
            [&get_pressure, &p_max](const double local_temperature) {
              return get_pressure(local_temperature) - p_max;
            },
            min_temperature, temp, 1e-50, 1.0e-13);
      } else {
        temp = 0.5 * (temp + min_temperature);
      }
      return true;
    }
    return false;
  };

  using std::abs;
  if (density < opts.density_lower_bound) {
    if (abs(temp - min_temperature) > opts.eplison_kappa_minus * abs(temp)) {
      temp = min_temperature;
    }
    return true;
  } else if (density < opts.density_upper_bound) {
    const double kappa_max =
        1.0 +
        opts.epsilon_kappa_max *
            std::min((density - opts.density_lower_bound) /
                         (opts.density_upper_bound - opts.density_lower_bound),
                     1.0);
    return impl(kappa_max);
  } else if (kappa_limiting_.value().limit_above_density_upper_bound) {
    return impl(1.0 + opts.epsilon_kappa_max);
  }
  return false;
}

template <size_t Dim>
bool operator==(const FixToAtmosphere<Dim>& lhs,
                const FixToAtmosphere<Dim>& rhs) {
  return lhs.density_of_atmosphere_ == rhs.density_of_atmosphere_ and
         lhs.density_cutoff_ == rhs.density_cutoff_ and
         lhs.velocity_limiting_ == rhs.velocity_limiting_ and
         lhs.kappa_limiting_ == rhs.kappa_limiting_;
}

template <size_t Dim>
bool operator!=(const FixToAtmosphere<Dim>& lhs,
                const FixToAtmosphere<Dim>& rhs) {
  return not(lhs == rhs);
}

template <size_t Dim>
void FixToAtmosphere<Dim>::VelocityLimitingOptions::pup(PUP::er& p) {
  p | atmosphere_max_velocity;
  p | near_atmosphere_max_velocity;
  p | atmosphere_density_cutoff;
  p | transition_density_bound;
}

template <size_t Dim>
bool FixToAtmosphere<Dim>::VelocityLimitingOptions::operator==(
    const VelocityLimitingOptions& rhs) const {
  return atmosphere_max_velocity == rhs.atmosphere_max_velocity and
         near_atmosphere_max_velocity == rhs.near_atmosphere_max_velocity and
         atmosphere_density_cutoff == rhs.atmosphere_density_cutoff and
         transition_density_bound == rhs.transition_density_bound;
}

template <size_t Dim>
bool FixToAtmosphere<Dim>::VelocityLimitingOptions::operator!=(
    const VelocityLimitingOptions& rhs) const {
  return not(*this == rhs);
}

template <size_t Dim>
void FixToAtmosphere<Dim>::KappaLimitingOptions::pup(PUP::er& p) {
  p | density_lower_bound;
  p | eplison_kappa_minus;
  p | density_upper_bound;
  p | epsilon_kappa_max;
  p | min_temperature;
  p | limit_above_density_upper_bound;
}

template <size_t Dim>
bool FixToAtmosphere<Dim>::KappaLimitingOptions::operator==(
    const KappaLimitingOptions& rhs) const {
  return density_lower_bound == rhs.density_lower_bound and
         eplison_kappa_minus == rhs.eplison_kappa_minus and
         density_upper_bound == rhs.density_upper_bound and
         epsilon_kappa_max == rhs.epsilon_kappa_max and
         min_temperature == rhs.min_temperature and
         limit_above_density_upper_bound == rhs.limit_above_density_upper_bound;
}

template <size_t Dim>
bool FixToAtmosphere<Dim>::KappaLimitingOptions::operator!=(
    const KappaLimitingOptions& rhs) const {
  return not(*this == rhs);
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
      const gsl::not_null<Scalar<DataVector>*> temperature,                   \
      const Scalar<DataVector>& electron_fraction,                            \
      const tnsr::ii<DataVector, DIM(data), Frame::Inertial>& spatial_metric, \
      const EquationsOfState::EquationOfState<true, THERMO_DIM(data)>&        \
          equation_of_state) const;

GENERATE_INSTANTIATIONS(INSTANTIATION, (1, 2, 3), (1, 2, 3))

#undef DIM
#undef THERMO_DIM
#undef INSTANTIATION

}  // namespace VariableFixing
