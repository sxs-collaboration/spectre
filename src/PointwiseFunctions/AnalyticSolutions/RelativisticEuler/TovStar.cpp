// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "PointwiseFunctions/AnalyticSolutions/RelativisticEuler/TovStar.hpp"

#include <cstddef>
#include <pup.h>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "PointwiseFunctions/AnalyticSolutions/GeneralRelativity/Tov.hpp"
#include "PointwiseFunctions/Hydro/Tags.hpp"
#include "Utilities/ConstantExpressions.hpp"
#include "Utilities/ErrorHandling/Error.hpp"
#include "Utilities/GenerateInstantiations.hpp"
#include "Utilities/MakeWithValue.hpp"

namespace RelativisticEuler::Solutions {

TovStar::TovStar(CkMigrateMessage* msg) : InitialData(msg) {}

TovStar::TovStar(const double central_rest_mass_density,
                 const double polytropic_constant,
                 const double polytropic_exponent,
                 const gr::Solutions::TovCoordinates coordinate_system)
    : central_rest_mass_density_(central_rest_mass_density),
      polytropic_constant_(polytropic_constant),
      polytropic_exponent_(polytropic_exponent),
      equation_of_state_{polytropic_constant_, polytropic_exponent_},
      coordinate_system_(coordinate_system),
      radial_solution_(equation_of_state_, central_rest_mass_density_,
                       coordinate_system_) {}

void TovStar::pup(PUP::er& p) {
  InitialData::pup(p);
  p | central_rest_mass_density_;
  p | polytropic_constant_;
  p | polytropic_exponent_;
  p | equation_of_state_;
  p | coordinate_system_;
  p | radial_solution_;
}

namespace tov_detail {
#if defined(__GNUC__) && !defined(__clang__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wsuggest-attribute=noreturn"
#endif  // defined(__GNUC__) && !defined(__clang__)

template <typename DataType, StarRegion Region>
void TovVariables<DataType, Region>::operator()(
    [[maybe_unused]] const gsl::not_null<Scalar<DataType>*> mass_over_radius,
    const gsl::not_null<Cache*> /*cache*/,
    Tags::MassOverRadius<DataType> /*meta*/) const {
  if constexpr (Region == StarRegion::Exterior) {
    ERROR(
        "The mass-over-radius quantity should not be needed in the exterior of "
        "the star.");
  } else {
    get(*mass_over_radius) = radial_solution.mass_over_radius(radius);
  }
}

template <typename DataType, StarRegion Region>
void TovVariables<DataType, Region>::operator()(
    [[maybe_unused]] const gsl::not_null<Scalar<DataType>*>
        log_specific_enthalpy,
    const gsl::not_null<Cache*> /*cache*/,
    Tags::LogSpecificEnthalpy<DataType> /*meta*/) const {
  if constexpr (Region == StarRegion::Exterior) {
    ERROR(
        "The log-specific-enthalpy quantity should not be needed in the "
        "exterior of the star (just use the specific enthalpy).");
  } else {
    get(*log_specific_enthalpy) = radial_solution.log_specific_enthalpy(radius);
  }
}

template <typename DataType, StarRegion Region>
void TovVariables<DataType, Region>::operator()(
    const gsl::not_null<Scalar<DataType>*> specific_enthalpy,
    [[maybe_unused]] const gsl::not_null<Cache*> cache,
    hydro::Tags::SpecificEnthalpy<DataType> /*meta*/) const {
  if constexpr (Region == StarRegion::Exterior) {
    get(*specific_enthalpy) = 1.;
  } else {
    const auto& log_specific_enthalpy =
        get(cache->get_var(*this, Tags::LogSpecificEnthalpy<DataType>{}));
    get(*specific_enthalpy) = exp(log_specific_enthalpy);
  }
}

template <typename DataType, StarRegion Region>
void TovVariables<DataType, Region>::operator()(
    const gsl::not_null<Scalar<DataType>*> conformal_factor,
    const gsl::not_null<Cache*> /*cache*/,
    Tags::ConformalFactor<DataType> /*meta*/) const {
  if (radial_solution.coordinate_system() ==
      gr::Solutions::TovCoordinates::Isotropic) {
    if constexpr (Region == StarRegion::Exterior) {
      get(*conformal_factor) = 1. + 0.5 * radial_solution.total_mass() / radius;
    } else {
      get(*conformal_factor) = radial_solution.conformal_factor(radius);
    }
  } else {
    ERROR(
        "The conformal factor should not be needed in Schwarzschild "
        "coordinates.");
  }
}

template <typename DataType, StarRegion Region>
void TovVariables<DataType, Region>::operator()(
    [[maybe_unused]] const gsl::not_null<Scalar<DataType>*> dr_conformal_factor,
    [[maybe_unused]] const gsl::not_null<Cache*> cache,
    Tags::DrConformalFactor<DataType> /*meta*/) const {
  if (radial_solution.coordinate_system() ==
      gr::Solutions::TovCoordinates::Isotropic) {
    if constexpr (Region == StarRegion::Exterior) {
      get(*dr_conformal_factor) =
          -0.5 * radial_solution.total_mass() / square(radius);
    } else if constexpr (Region == StarRegion::Center) {
      ERROR("The 'DrConformalFactor' should not be needed at the star center.");
    } else {
      const auto& conformal_factor =
          get(cache->get_var(*this, Tags::ConformalFactor<DataType>{}));
      const auto& mass_over_areal_radius =
          get(cache->get_var(*this, Tags::MassOverRadius<DataType>{}));
      get(*dr_conformal_factor) = 0.5 * conformal_factor / radius *
                                  (sqrt(1. - 2. * mass_over_areal_radius) - 1.);
    }
  } else {
    ERROR(
        "The conformal factor should not be needed in Schwarzschild "
        "coordinates.");
  }
}

template <typename DataType, StarRegion Region>
void TovVariables<DataType, Region>::operator()(
    const gsl::not_null<Scalar<DataType>*> areal_radius,
    const gsl::not_null<Cache*> cache,
    Tags::ArealRadius<DataType> /*meta*/) const {
  if (radial_solution.coordinate_system() ==
      gr::Solutions::TovCoordinates::Isotropic) {
    const auto& conformal_factor =
        get(cache->get_var(*this, Tags::ConformalFactor<DataType>{}));
    get(*areal_radius) = square(conformal_factor) * radius;
  } else {
    ERROR(
        "No need to compute the areal radius in Schwarzschild coordinates, "
        "just use 'radius'.");
  }
}

template <typename DataType, StarRegion Region>
void TovVariables<DataType, Region>::operator()(
    [[maybe_unused]] const gsl::not_null<Scalar<DataType>*> dr_areal_radius,
    [[maybe_unused]] const gsl::not_null<Cache*> cache,
    Tags::DrArealRadius<DataType> /*meta*/) const {
  if (radial_solution.coordinate_system() ==
      gr::Solutions::TovCoordinates::Isotropic) {
    if constexpr (Region == StarRegion::Exterior) {
      const auto& areal_radius =
          get(cache->get_var(*this, Tags::ArealRadius<DataType>{}));
      get(*dr_areal_radius) =
          areal_radius / radius *
          sqrt(1. - 2. * radial_solution.total_mass() / areal_radius);
    } else if constexpr (Region == StarRegion::Center) {
      ERROR("The 'DrArealRadius' should not be needed at the star center.");
    } else {
      const auto& areal_radius =
          get(cache->get_var(*this, Tags::ArealRadius<DataType>{}));
      const auto& mass_over_radius =
          get(cache->get_var(*this, Tags::MassOverRadius<DataType>{}));
      get(*dr_areal_radius) =
          areal_radius / radius * sqrt(1. - 2. * mass_over_radius);
    }
  } else {
    ERROR(
        "No need to compute the areal radius in Schwarzschild coordinates, "
        "just use 'radius'.");
  }
}

template <typename DataType, StarRegion Region>
void TovVariables<DataType, Region>::operator()(
    const gsl::not_null<Scalar<DataType>*> rest_mass_density,
    [[maybe_unused]] const gsl::not_null<Cache*> cache,
    hydro::Tags::RestMassDensity<DataType> /*meta*/) const {
  if constexpr (Region == StarRegion::Exterior) {
    get(*rest_mass_density) = 0.;
  } else {
    const auto& specific_enthalpy =
        cache->get_var(*this, hydro::Tags::SpecificEnthalpy<DataType>{});
    *rest_mass_density = eos.rest_mass_density_from_enthalpy(specific_enthalpy);
  }
}

template <typename DataType, StarRegion Region>
void TovVariables<DataType, Region>::operator()(
    const gsl::not_null<Scalar<DataType>*> pressure,
    [[maybe_unused]] const gsl::not_null<Cache*> cache,
    hydro::Tags::Pressure<DataType> /*meta*/) const {
  if constexpr (Region == StarRegion::Exterior) {
    get(*pressure) = 0.;
  } else {
    const auto& rest_mass_density =
        cache->get_var(*this, hydro::Tags::RestMassDensity<DataType>{});
    *pressure = eos.pressure_from_density(rest_mass_density);
  }
}

template <typename DataType, StarRegion Region>
void TovVariables<DataType, Region>::operator()(
    const gsl::not_null<Scalar<DataType>*> specific_internal_energy,
    [[maybe_unused]] const gsl::not_null<Cache*> cache,
    hydro::Tags::SpecificInternalEnergy<DataType> /*meta*/) const {
  if constexpr (Region == StarRegion::Exterior) {
    get(*specific_internal_energy) = 0.;
  } else {
    const auto& rest_mass_density =
        cache->get_var(*this, hydro::Tags::RestMassDensity<DataType>{});
    *specific_internal_energy =
        eos.specific_internal_energy_from_density(rest_mass_density);
  }
}

template <typename DataType, StarRegion Region>
void TovVariables<DataType, Region>::operator()(
    const gsl::not_null<Scalar<DataType>*> metric_time_potential,
    [[maybe_unused]] const gsl::not_null<Cache*> cache,
    Tags::MetricTimePotential<DataType> /*meta*/) const {
  // Compute phi in Eq. (1.73) in BaumgarteShapiro, which is the logarithm of
  // the lapse:
  //   lapse = e^phi
  if constexpr (Region == StarRegion::Exterior) {
    const auto& areal_radius =
        radial_solution.coordinate_system() ==
                gr::Solutions::TovCoordinates::Isotropic
            ? get(cache->get_var(*this, Tags::ArealRadius<DataType>{}))
            : radius;
    get(*metric_time_potential) =
        0.5 * log(1. - 2. * radial_solution.total_mass() / areal_radius);
  } else {
    const auto& log_specific_enthalpy =
        get(cache->get_var(*this, Tags::LogSpecificEnthalpy<DataType>{}));
    const double log_lapse_at_outer_radius =
        log(radial_solution.injection_energy());
    get(*metric_time_potential) =
        log_lapse_at_outer_radius - log_specific_enthalpy;
  }
}

template <typename DataType, StarRegion Region>
void TovVariables<DataType, Region>::operator()(
    const gsl::not_null<Scalar<DataType>*> dr_metric_time_potential,
    [[maybe_unused]] const gsl::not_null<Cache*> cache,
    Tags::DrMetricTimePotential<DataType> /*meta*/) const {
  if constexpr (Region == StarRegion::Exterior) {
    const auto& areal_radius =
        radial_solution.coordinate_system() ==
                gr::Solutions::TovCoordinates::Isotropic
            ? get(cache->get_var(*this, Tags::ArealRadius<DataType>{}))
            : radius;
    get(*dr_metric_time_potential) =
        radial_solution.total_mass() / square(areal_radius) /
        (1. - 2. * radial_solution.total_mass() / areal_radius);
    if (radial_solution.coordinate_system() ==
        gr::Solutions::TovCoordinates::Isotropic) {
      const auto& dr_areal_radius =
          get(cache->get_var(*this, Tags::DrArealRadius<DataType>{}));
      get(*dr_metric_time_potential) *= dr_areal_radius;
    }
  } else if constexpr (Region == StarRegion::Center) {
    get(*dr_metric_time_potential) = 0.;
  } else {
    // Compute dphi/dr from the TOV equations, e.g. Eq. (1.79) in
    // BaumgarteShapiro.
    const auto& mass_over_radius =
        get(cache->get_var(*this, Tags::MassOverRadius<DataType>{}));
    const auto& pressure =
        get(cache->get_var(*this, hydro::Tags::Pressure<DataType>{}));
    const auto& areal_radius =
        radial_solution.coordinate_system() ==
                gr::Solutions::TovCoordinates::Isotropic
            ? get(cache->get_var(*this, Tags::ArealRadius<DataType>{}))
            : radius;
    get(*dr_metric_time_potential) = (mass_over_radius / areal_radius +
                                      4. * M_PI * pressure * areal_radius) /
                                     (1. - 2. * mass_over_radius);
    if (radial_solution.coordinate_system() ==
        gr::Solutions::TovCoordinates::Isotropic) {
      const auto& dr_areal_radius =
          get(cache->get_var(*this, Tags::DrArealRadius<DataType>{}));
      get(*dr_metric_time_potential) *= dr_areal_radius;
    }
  }
}

template <typename DataType, StarRegion Region>
void TovVariables<DataType, Region>::operator()(
    const gsl::not_null<Scalar<DataType>*> metric_radial_potential,
    const gsl::not_null<Cache*> cache,
    Tags::MetricRadialPotential<DataType> /*meta*/) const {
  // Compute lambda in Eq. (1.73) in BaumgarteShapiro
  if (radial_solution.coordinate_system() ==
      gr::Solutions::TovCoordinates::Isotropic) {
    const auto& conformal_factor =
        get(cache->get_var(*this, Tags::ConformalFactor<DataType>{}));
    get(*metric_radial_potential) = 2. * log(conformal_factor);
  } else {
    if constexpr (Region == StarRegion::Exterior) {
      const auto& metric_time_potential =
          get(cache->get_var(*this, Tags::MetricTimePotential<DataType>{}));
      get(*metric_radial_potential) = -metric_time_potential;
    } else {
      const auto& mass_over_radius =
          get(cache->get_var(*this, Tags::MassOverRadius<DataType>{}));
      // Eq. (1.76) in BaumgarteShapiro (equation in book is missing a sqrt)
      get(*metric_radial_potential) = -0.5 * log(1. - 2. * mass_over_radius);
    }
  }
}

template <typename DataType, StarRegion Region>
void TovVariables<DataType, Region>::operator()(
    const gsl::not_null<Scalar<DataType>*> dr_metric_radial_potential,
    [[maybe_unused]] const gsl::not_null<Cache*> cache,
    Tags::DrMetricRadialPotential<DataType> /*meta*/) const {
  if (radial_solution.coordinate_system() ==
      gr::Solutions::TovCoordinates::Isotropic) {
    const auto& conformal_factor =
        get(cache->get_var(*this, Tags::ConformalFactor<DataType>{}));
    const auto& dr_conformal_factor =
        get(cache->get_var(*this, Tags::DrConformalFactor<DataType>{}));
    get(*dr_metric_radial_potential) =
        2. / conformal_factor * dr_conformal_factor;
  } else {
    if constexpr (Region == StarRegion::Exterior) {
      const auto& dr_metric_time_potential =
          get(cache->get_var(*this, Tags::DrMetricTimePotential<DataType>{}));
      get(*dr_metric_radial_potential) = -dr_metric_time_potential;
    } else if constexpr (Region == StarRegion::Center) {
      get(*dr_metric_radial_potential) = 0.;
    } else {
      const auto& mass_over_radius =
          get(cache->get_var(*this, Tags::MassOverRadius<DataType>{}));
      const auto& specific_enthalpy =
          get(cache->get_var(*this, hydro::Tags::SpecificEnthalpy<DataType>{}));
      const auto& rest_mass_density =
          get(cache->get_var(*this, hydro::Tags::RestMassDensity<DataType>{}));
      const auto& pressure =
          get(cache->get_var(*this, hydro::Tags::Pressure<DataType>{}));
      // Compute dm/dr from the TOV equations, e.g. Eq. (1.77) in
      // BaumgarteShapiro.
      get(*dr_metric_radial_potential) =
          (4. * M_PI * radius *
               (specific_enthalpy * rest_mass_density - pressure) -
           mass_over_radius / radius) /
          (1. - 2. * mass_over_radius);
    }
  }
}

template <typename DataType, StarRegion Region>
void TovVariables<DataType, Region>::operator()(
    const gsl::not_null<Scalar<DataType>*> metric_angular_potential,
    const gsl::not_null<Cache*> cache,
    Tags::MetricAngularPotential<DataType> /*meta*/) const {
  if (radial_solution.coordinate_system() ==
      gr::Solutions::TovCoordinates::Isotropic) {
    const auto& metric_radial_potential =
        cache->get_var(*this, Tags::MetricRadialPotential<DataType>{});
    *metric_angular_potential = metric_radial_potential;
  } else {
    get(*metric_angular_potential) = 0.;
  }
}

template <typename DataType, StarRegion Region>
void TovVariables<DataType, Region>::operator()(
    const gsl::not_null<Scalar<DataType>*> dr_metric_angular_potential,
    const gsl::not_null<Cache*> cache,
    Tags::DrMetricAngularPotential<DataType> /*meta*/) const {
  if (radial_solution.coordinate_system() ==
      gr::Solutions::TovCoordinates::Isotropic) {
    const auto& dr_metric_radial_potential =
        cache->get_var(*this, Tags::DrMetricRadialPotential<DataType>{});
    *dr_metric_angular_potential = dr_metric_radial_potential;
  } else {
    get(*dr_metric_angular_potential) = 0.;
  }
}

template <typename DataType, StarRegion Region>
void TovVariables<DataType, Region>::operator()(
    const gsl::not_null<Scalar<DataType>*> dr_pressure,
    [[maybe_unused]] const gsl::not_null<Cache*> cache,
    Tags::DrPressure<DataType> /*meta*/) const {
  if constexpr (Region == StarRegion::Exterior) {
    get(*dr_pressure) = 0.;
  } else {
    // Compute dP/dr from the TOV equations, e.g. Eq. (1.78) in
    // BaumgarteShapiro.
    const auto& dr_metric_time_potential =
        get(cache->get_var(*this, Tags::DrMetricTimePotential<DataType>{}));
    const auto& rest_mass_density =
        get(cache->get_var(*this, hydro::Tags::RestMassDensity<DataType>{}));
    const auto& specific_enthalpy =
        get(cache->get_var(*this, hydro::Tags::SpecificEnthalpy<DataType>{}));
    get(*dr_pressure) =
        -dr_metric_time_potential * rest_mass_density * specific_enthalpy;
  }
}

template <typename DataType, StarRegion Region>
void TovVariables<DataType, Region>::operator()(
    const gsl::not_null<Scalar<DataType>*> lorentz_factor,
    const gsl::not_null<Cache*> /*cache*/,
    hydro::Tags::LorentzFactor<DataType> /*meta*/) const {
  get(*lorentz_factor) = 1.;
}

template <typename DataType, StarRegion Region>
void TovVariables<DataType, Region>::operator()(
    const gsl::not_null<tnsr::I<DataType, 3>*> spatial_velocity,
    const gsl::not_null<Cache*> /*cache*/,
    hydro::Tags::SpatialVelocity<DataType, 3> /*meta*/) const {
  get<0>(*spatial_velocity) = 0.;
  get<1>(*spatial_velocity) = 0.;
  get<2>(*spatial_velocity) = 0.;
}

template <typename DataType, StarRegion Region>
void TovVariables<DataType, Region>::operator()(
    const gsl::not_null<tnsr::I<DataType, 3>*> magnetic_field,
    const gsl::not_null<Cache*> /*cache*/,
    hydro::Tags::MagneticField<DataType, 3> /*meta*/) const {
  get<0>(*magnetic_field) = 0.;
  get<1>(*magnetic_field) = 0.;
  get<2>(*magnetic_field) = 0.;
}

template <typename DataType, StarRegion Region>
void TovVariables<DataType, Region>::operator()(
    const gsl::not_null<Scalar<DataType>*> div_cleaning_field,
    const gsl::not_null<Cache*> /*cache*/,
    hydro::Tags::DivergenceCleaningField<DataType> /*meta*/) const {
  get(*div_cleaning_field) = 0.;
}

template <typename DataType, StarRegion Region>
void TovVariables<DataType, Region>::operator()(
    const gsl::not_null<Scalar<DataType>*> lapse,
    const gsl::not_null<Cache*> cache,
    gr::Tags::Lapse<DataType> /*meta*/) const {
  const auto& metric_time_potential =
      get(cache->get_var(*this, Tags::MetricTimePotential<DataType>{}));
  get(*lapse) = exp(metric_time_potential);
}

template <typename DataType, StarRegion Region>
void TovVariables<DataType, Region>::operator()(
    const gsl::not_null<Scalar<DataType>*> dt_lapse,
    const gsl::not_null<Cache*> /*cache*/,
    ::Tags::dt<gr::Tags::Lapse<DataType>> /*meta*/) const {
  get(*dt_lapse) = 0.;
}

template <typename DataType, StarRegion Region>
void TovVariables<DataType, Region>::operator()(
    const gsl::not_null<tnsr::i<DataType, 3>*> deriv_lapse,
    [[maybe_unused]] const gsl::not_null<Cache*> cache,
    ::Tags::deriv<gr::Tags::Lapse<DataType>, tmpl::size_t<3>,
                  Frame::Inertial> /*meta*/) const {
  if constexpr (Region == StarRegion::Center) {
    get<0>(*deriv_lapse) = 0.;
    get<1>(*deriv_lapse) = 0.;
    get<2>(*deriv_lapse) = 0.;
  } else {
    const auto& lapse = get(cache->get_var(*this, gr::Tags::Lapse<DataType>{}));
    const auto& dr_metric_time_potential =
        get(cache->get_var(*this, Tags::DrMetricTimePotential<DataType>{}));
    // lapse = exp(phi), so dlapse/dr = lapse * dphi/dr * x/r
    get<0>(*deriv_lapse) = lapse * dr_metric_time_potential / radius;
    get<1>(*deriv_lapse) = get<0>(*deriv_lapse);
    get<2>(*deriv_lapse) = get<0>(*deriv_lapse);
    for (size_t i = 0; i < 3; ++i) {
      deriv_lapse->get(i) *= coords.get(i);
    }
  }
}

template <typename DataType, StarRegion Region>
void TovVariables<DataType, Region>::operator()(
    const gsl::not_null<tnsr::I<DataType, 3>*> shift,
    const gsl::not_null<Cache*> /*cache*/,
    gr::Tags::Shift<3, Frame::Inertial, DataType> /*meta*/) const {
  get<0>(*shift) = 0.;
  get<1>(*shift) = 0.;
  get<2>(*shift) = 0.;
}

template <typename DataType, StarRegion Region>
void TovVariables<DataType, Region>::operator()(
    const gsl::not_null<tnsr::I<DataType, 3>*> dt_shift,
    const gsl::not_null<Cache*> /*cache*/,
    ::Tags::dt<gr::Tags::Shift<3, Frame::Inertial, DataType>> /*meta*/) const {
  get<0>(*dt_shift) = 0.;
  get<1>(*dt_shift) = 0.;
  get<2>(*dt_shift) = 0.;
}

template <typename DataType, StarRegion Region>
void TovVariables<DataType, Region>::operator()(
    const gsl::not_null<tnsr::iJ<DataType, 3>*> deriv_shift,
    const gsl::not_null<Cache*> /*cache*/,
    ::Tags::deriv<gr::Tags::Shift<3, Frame::Inertial, DataType>,
                  tmpl::size_t<3>, Frame::Inertial> /*meta*/) const {
  *deriv_shift = make_with_value<tnsr::iJ<DataType, 3>>(coords, 0.);
}

template <typename DataType, StarRegion Region>
void TovVariables<DataType, Region>::operator()(
    const gsl::not_null<tnsr::ii<DataType, 3>*> spatial_metric,
    const gsl::not_null<Cache*> cache,
    gr::Tags::SpatialMetric<3, Frame::Inertial, DataType> /*meta*/) const {
  if constexpr (Region == StarRegion::Center) {
    const auto& metric_angular_potential =
        get(cache->get_var(*this, Tags::MetricAngularPotential<DataType>{}));
    get<0, 0>(*spatial_metric) = exp(2. * metric_angular_potential);
    get<1, 1>(*spatial_metric) = get<0, 0>(*spatial_metric);
    get<2, 2>(*spatial_metric) = get<0, 0>(*spatial_metric);
    get<0, 1>(*spatial_metric) = 0.;
    get<0, 2>(*spatial_metric) = 0.;
    get<1, 2>(*spatial_metric) = 0.;
  } else {
    const auto& metric_radial_potential =
        get(cache->get_var(*this, Tags::MetricRadialPotential<DataType>{}));
    const auto& metric_angular_potential =
        get(cache->get_var(*this, Tags::MetricAngularPotential<DataType>{}));
    *spatial_metric = tnsr::ii<DataType, 3, Frame::Inertial>{
        (exp(2. * metric_radial_potential) -
         exp(2. * metric_angular_potential)) /
        square(radius)};
    for (size_t i = 0; i < 3; ++i) {
      for (size_t j = i; j < 3; ++j) {
        spatial_metric->get(i, j) *= coords.get(i) * coords.get(j);
      }
      spatial_metric->get(i, i) += exp(2. * metric_angular_potential);
    }
  }
}

template <typename DataType, StarRegion Region>
void TovVariables<DataType, Region>::operator()(
    const gsl::not_null<tnsr::ii<DataType, 3>*> dt_spatial_metric,
    const gsl::not_null<Cache*> /*cache*/,
    ::Tags::dt<gr::Tags::SpatialMetric<3, Frame::Inertial, DataType>> /*meta*/)
    const {
  *dt_spatial_metric = make_with_value<tnsr::ii<DataType, 3>>(coords, 0.);
}

template <typename DataType, StarRegion Region>
void TovVariables<DataType, Region>::operator()(
    const gsl::not_null<tnsr::ijj<DataType, 3>*> deriv_spatial_metric,
    [[maybe_unused]] const gsl::not_null<Cache*> cache,
    ::Tags::deriv<gr::Tags::SpatialMetric<3, Frame::Inertial, DataType>,
                  tmpl::size_t<3>, Frame::Inertial> /*meta*/) const {
  if constexpr (Region == StarRegion::Center) {
    *deriv_spatial_metric = make_with_value<tnsr::ijj<DataType, 3>>(coords, 0.);
  } else {
    const auto& metric_radial_potential =
        get(cache->get_var(*this, Tags::MetricRadialPotential<DataType>{}));
    const auto& dr_metric_radial_potential =
        get(cache->get_var(*this, Tags::DrMetricRadialPotential<DataType>{}));
    const auto& metric_angular_potential =
        get(cache->get_var(*this, Tags::MetricAngularPotential<DataType>{}));
    const auto& dr_metric_angular_potential =
        get(cache->get_var(*this, Tags::DrMetricAngularPotential<DataType>{}));
    *deriv_spatial_metric = tnsr::ijj<DataType, 3, Frame::Inertial>{
        2.0 *
        (exp(2.0 * metric_radial_potential) *
             (dr_metric_radial_potential - 1.0 / radius) -
         exp(2.0 * metric_angular_potential) *
             (dr_metric_angular_potential - 1.0 / radius)) /
        cube(radius)};
    for (size_t k = 0; k < 3; ++k) {
      for (size_t i = 0; i < 3; ++i) {
        for (size_t j = i; j < 3; ++j) {
          deriv_spatial_metric->get(k, i, j) *=
              coords.get(k) * coords.get(i) * coords.get(j);
          if (k == i) {
            deriv_spatial_metric->get(k, i, j) +=
                (exp(2.0 * metric_radial_potential) -
                 exp(2.0 * metric_angular_potential)) /
                square(radius) * coords.get(j);
          }
          if (k == j) {
            deriv_spatial_metric->get(k, i, j) +=
                (exp(2.0 * metric_radial_potential) -
                 exp(2.0 * metric_angular_potential)) /
                square(radius) * coords.get(i);
          }
          if (i == j) {
            deriv_spatial_metric->get(k, i, j) +=
                2.0 * exp(2.0 * metric_angular_potential) *
                dr_metric_angular_potential * coords.get(k) / radius;
          }
        }
      }
    }
  }
}

template <typename DataType, StarRegion Region>
void TovVariables<DataType, Region>::operator()(
    const gsl::not_null<Scalar<DataType>*> sqrt_det_spatial_metric,
    const gsl::not_null<Cache*> cache,
    gr::Tags::SqrtDetSpatialMetric<DataType> /*meta*/) const {
  const auto& metric_radial_potential =
      get(cache->get_var(*this, Tags::MetricRadialPotential<DataType>{}));
  const auto& metric_angular_potential =
      get(cache->get_var(*this, Tags::MetricAngularPotential<DataType>{}));
  get(*sqrt_det_spatial_metric) =
      exp(metric_radial_potential + 2.0 * metric_angular_potential);
}

template <typename DataType, StarRegion Region>
void TovVariables<DataType, Region>::operator()(
    const gsl::not_null<tnsr::II<DataType, 3>*> inv_spatial_metric,
    const gsl::not_null<Cache*> cache,
    gr::Tags::InverseSpatialMetric<3, Frame::Inertial, DataType> /*meta*/)
    const {
  if constexpr (Region == StarRegion::Center) {
    const auto& metric_angular_potential =
        get(cache->get_var(*this, Tags::MetricAngularPotential<DataType>{}));
    get<0, 0>(*inv_spatial_metric) = exp(-2. * metric_angular_potential);
    get<1, 1>(*inv_spatial_metric) = get<0, 0>(*inv_spatial_metric);
    get<2, 2>(*inv_spatial_metric) = get<0, 0>(*inv_spatial_metric);
    get<0, 1>(*inv_spatial_metric) = 0.;
    get<0, 2>(*inv_spatial_metric) = 0.;
    get<1, 2>(*inv_spatial_metric) = 0.;
  } else {
    const auto& metric_radial_potential =
        get(cache->get_var(*this, Tags::MetricRadialPotential<DataType>{}));
    const auto& metric_angular_potential =
        get(cache->get_var(*this, Tags::MetricAngularPotential<DataType>{}));
    *inv_spatial_metric = tnsr::II<DataType, 3, Frame::Inertial>{
        (exp(-2. * metric_radial_potential) -
         exp(-2. * metric_angular_potential)) /
        square(radius)};
    for (size_t d0 = 0; d0 < 3; ++d0) {
      for (size_t d1 = d0; d1 < 3; ++d1) {
        inv_spatial_metric->get(d0, d1) *= coords.get(d0) * coords.get(d1);
      }
      inv_spatial_metric->get(d0, d0) += exp(-2.0 * metric_angular_potential);
    }
  }
}

template <typename DataType, StarRegion Region>
void TovVariables<DataType, Region>::operator()(
    const gsl::not_null<tnsr::ii<DataType, 3>*> extrinsic_curvature,
    const gsl::not_null<Cache*> /*cache*/,
    gr::Tags::ExtrinsicCurvature<3, Frame::Inertial, DataType> /*meta*/) const {
  *extrinsic_curvature = make_with_value<tnsr::ii<DataType, 3>>(coords, 0.);
}

#if defined(__GNUC__) && !defined(__clang__)
#pragma GCC diagnostic pop
#endif  // defined(__GNUC__) && !defined(__clang__)
}  // namespace tov_detail

PUP::able::PUP_ID TovStar::my_PUP_ID = 0;

bool operator==(const TovStar& lhs, const TovStar& rhs) {
  // there is no comparison operator for the EoS, but should be okay as
  // the `polytropic_exponent`s and `polytropic_constant`s are compared
  return lhs.central_rest_mass_density_ == rhs.central_rest_mass_density_ and
         lhs.polytropic_constant_ == rhs.polytropic_constant_ and
         lhs.polytropic_exponent_ == rhs.polytropic_exponent_ and
         lhs.coordinate_system_ == rhs.coordinate_system_;
}

bool operator!=(const TovStar& lhs, const TovStar& rhs) {
  return not(lhs == rhs);
}

#define DTYPE(data) BOOST_PP_TUPLE_ELEM(0, data)
#define REGION(data) BOOST_PP_TUPLE_ELEM(1, data)
#define INSTANTIATE(_, data) \
  template class tov_detail::TovVariables<DTYPE(data), REGION(data)>;

GENERATE_INSTANTIATIONS(INSTANTIATE, (double, DataVector),
                        (tov_detail::StarRegion::Center,
                         tov_detail::StarRegion::Interior,
                         tov_detail::StarRegion::Exterior))

#undef DTYPE
#undef REGION
#undef INSTANTIATE
}  // namespace RelativisticEuler::Solutions
