// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "PointwiseFunctions/AnalyticData/BnsInitialData/SpectreData.hpp"

#include <memory>
#include <pup.h>
#include <string>
#include <utility>
#include <vector>

#include "DataStructures/Tensor/EagerMath/DeterminantAndInverse.hpp"
#include "DataStructures/Tensor/EagerMath/DotProduct.hpp"
#include "DataStructures/Tensor/EagerMath/RaiseOrLowerIndex.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "IO/Exporter/PointwiseInterpolator.hpp"
#include "PointwiseFunctions/GeneralRelativity/Tags.hpp"
#include "PointwiseFunctions/Hydro/SpecificEnthalpy.hpp"
#include "PointwiseFunctions/Hydro/Tags.hpp"
#include "PointwiseFunctions/InitialDataUtilities/Background.hpp"
#include "PointwiseFunctions/InitialDataUtilities/InitialGuess.hpp"
#include "Utilities/ContainerHelpers.hpp"
#include "Utilities/ErrorHandling/Error.hpp"
#include "Utilities/GenerateInstantiations.hpp"
#include "Utilities/System/ParallelInfo.hpp"
#include "Utilities/TaggedTuple.hpp"

namespace BnsInitialData::AnalyticData {

template <size_t ThermodynamicDim>
SpectreData<ThermodynamicDim>::SpectreData(
    std::string volume_file_glob, std::string subfile_name,
    int observation_step,
    std::unique_ptr<equation_of_state_type> equation_of_state,
    const double density_cutoff, const double orbital_angular_velocity,
    const double euler_enthalpy_constant)
    : volume_file_glob_(std::move(volume_file_glob)),
      subfile_name_(std::move(subfile_name)),
      observation_step_(observation_step),
      equation_of_state_(std::move(equation_of_state)),
      density_cutoff_(density_cutoff),
      orbital_angular_velocity_(orbital_angular_velocity),
      euler_enthalpy_constant_(euler_enthalpy_constant) {}

template <size_t ThermodynamicDim>
SpectreData<ThermodynamicDim>& SpectreData<ThermodynamicDim>::operator=(
    const SpectreData& rhs) {
  volume_file_glob_ = rhs.volume_file_glob_;
  subfile_name_ = rhs.subfile_name_;
  observation_step_ = rhs.observation_step_;
  equation_of_state_ = rhs.equation_of_state_->get_clone();
  density_cutoff_ = rhs.density_cutoff_;
  orbital_angular_velocity_ = rhs.orbital_angular_velocity_;
  euler_enthalpy_constant_ = rhs.euler_enthalpy_constant_;
  return *this;
}

template <size_t ThermodynamicDim>
SpectreData<ThermodynamicDim>::SpectreData(const SpectreData& rhs)
    : PUP::able(),
      elliptic::analytic_data::Background(rhs),
      elliptic::analytic_data::InitialGuess(rhs) {
  *this = rhs;
}

template <size_t ThermodynamicDim>
std::unique_ptr<elliptic::analytic_data::Background>
SpectreData<ThermodynamicDim>::get_clone() const {
  return std::make_unique<SpectreData>(*this);
}

template <size_t ThermodynamicDim>
SpectreData<ThermodynamicDim>::SpectreData(CkMigrateMessage* msg)
    : elliptic::analytic_data::Background(msg) {}

template <size_t ThermodynamicDim>
void SpectreData<ThermodynamicDim>::pup(PUP::er& p) {
  elliptic::analytic_data::InitialGuess::pup(p);
  p | volume_file_glob_;
  p | subfile_name_;
  p | observation_step_;
  p | equation_of_state_;
  p | density_cutoff_;
  p | orbital_angular_velocity_;
  p | euler_enthalpy_constant_;
}

template <size_t ThermodynamicDim>
PUP::able::PUP_ID SpectreData<ThermodynamicDim>::my_PUP_ID = 0;  // NOLINT

template <size_t ThermodynamicDim>
tuples::tagged_tuple_from_typelist<
    typename SpectreData<ThermodynamicDim>::interpolated_tags>
SpectreData<ThermodynamicDim>::interpolate_from_spectre(
    const tnsr::I<DataVector, 3>& x) const {
  return spectre::Exporter::interpolate_to_points<interpolated_tags>(
      volume_file_glob_, subfile_name_,
      spectre::Exporter::ObservationStep(observation_step_), x, false);
}

// Deriv of velocity potential is only
// used for validation
template <size_t ThermodynamicDim>

tnsr::i<DataVector, 3>
SpectreData<ThermodynamicDim>::deriv_of_velocity_potential(
    const tnsr::I<DataVector, 3, Frame::Inertial>& x) const {
  const auto interpolated_vars = interpolate_from_spectre(x);
  const auto& lower_spatial_four_velocity =
      get<hydro::Tags::LowerSpatialFourVelocity<DataVector, 3>>(
          interpolated_vars);
  auto rest_mass_density =
      get<hydro::Tags::RestMassDensity<DataVector>>(interpolated_vars);
  get(rest_mass_density) += 1e-16;
  const DataVector specific_enthalpy = select(
      step_function(get(rest_mass_density) - 0),
      get(equation_of_state_->pressure_from_density(rest_mass_density)) /
              get(rest_mass_density) +
          (1.0 + get(equation_of_state_->specific_internal_energy_from_density(
                     rest_mass_density))),
      make_with_value<DataVector>(
          get(rest_mass_density),
          equation_of_state_->specific_enthalpy_lower_bound()));
  auto result = make_with_value<tnsr::i<DataVector, 3>>(x, 0.0);

  tenex::evaluate<ti::i>(make_not_null(&result),
                         Scalar<DataVector>{specific_enthalpy}() *
                             lower_spatial_four_velocity(ti::i));
  return result;
}

// The velocity potential is used in the initial guess
template <size_t ThermodynamicDim>

tuples::TaggedTuple<Tags::VelocityPotential<DataVector>>
SpectreData<ThermodynamicDim>::variables(
    const tnsr::I<DataVector, 3, Frame::Inertial>& x,
    tmpl::list<Tags::VelocityPotential<DataVector>> /*meta*/) const {
  // return velocity potential (only a guess)
  Scalar<DataVector> velocity_potential =
      make_with_value<Scalar<DataVector>>(x, orbital_angular_velocity_);
  // This is not a good guess, but it's better than nothing.
  // It's not clear to me this should actually be used
  get(velocity_potential) *= (get<1>(x) * get<0>(x));

  return {std::move(velocity_potential)};
}
// The fixed sources are used in initialization
template <size_t ThermodynamicDim>

tuples::TaggedTuple<::Tags::FixedSource<Tags::VelocityPotential<DataVector>>>
SpectreData<ThermodynamicDim>::variables(
    const tnsr::I<DataVector, 3, Frame::Inertial>& x, const Mesh<3>& mesh,
    const InverseJacobian<DataVector, 3, Frame::ElementLogical,
                          Frame::Inertial>& inv_jacobian,
    tmpl::list<
        ::Tags::FixedSource<Tags::VelocityPotential<DataVector>>> /*meta*/)
    const {
  auto background_values = variables(x, mesh, inv_jacobian, background_tags{});
  tuples::TaggedTuple<::Tags::FixedSource<Tags::VelocityPotential<DataVector>>>
      result{};
  const auto& lapse = get<gr::Tags::Lapse<DataVector>>(background_values);
  const auto& shift = get<gr::Tags::Shift<DataVector, 3>>(background_values);
  const auto& rotational_shift =
      get<Tags::RotationalShift<DataVector>>(background_values);
  const auto& deriv_log_lapse_times_density_over_specific_enthalpy =
      get<Tags::DerivLogLapseTimesDensityOverSpecificEnthalpy<DataVector>>(
          background_values);

  const auto& deriv_of_lapse =
      get<::Tags::deriv<gr::Tags::Lapse<DataVector>,
                        tmpl::integral_constant<size_t, 3>, Frame::Inertial>>(
          background_values);

  const auto& deriv_of_shift =
      get<::Tags::deriv<gr::Tags::Shift<DataVector, 3>,
                        tmpl::integral_constant<size_t, 3>, Frame::Inertial>>(
          background_values);
  const auto& spatial_christoffel_second_kind_contracted =
      get<gr::Tags::SpatialChristoffelSecondKindContracted<DataVector, 3>>(
          background_values);

  ::tenex::evaluate<>(
      make_not_null(
          &get<::Tags::FixedSource<Tags::VelocityPotential<DataVector>>>(
              result)),
      -euler_enthalpy_constant_ *
          (1.0 / square(lapse()) * rotational_shift(ti::I) *
               deriv_log_lapse_times_density_over_specific_enthalpy(ti::i) -
           2.0 / cube(lapse()) * rotational_shift(ti::I) *
               deriv_of_lapse(ti::i) +
           1.0 / square(lapse()) * deriv_of_shift(ti::i, ti::I) +
           // Christoffel terms, assume the spatial rotational
           // killing vector is (spatially) covariantly constant
           1.0 / square(lapse()) *
               (shift(ti::I) *
                spatial_christoffel_second_kind_contracted(ti::i))));

  return result;
}
template <size_t ThermodynamicDim>

tuples::TaggedTuple<gr::Tags::InverseSpatialMetric<DataVector, 3>>
SpectreData<ThermodynamicDim>::variables(
    const tnsr::I<DataVector, 3, Frame::Inertial>& x,
    tmpl::list<gr::Tags::InverseSpatialMetric<DataVector, 3>> /*meta*/) const {
  // interpolate from spec, then set gamma
  const auto interpolated_vars = interpolate_from_spectre(x);

  const auto& spatial_metric =
      get<gr::Tags::SpatialMetric<DataVector, 3>>(interpolated_vars);
  tuples::TaggedTuple<gr::Tags::InverseSpatialMetric<DataVector, 3>> result{};
  get<gr::Tags::InverseSpatialMetric<DataVector, 3>>(result) =
      determinant_and_inverse(spatial_metric).second;
  return result;
}
template <size_t ThermodynamicDim>
tuples::tagged_tuple_from_typelist<
    typename SpectreData<ThermodynamicDim>::background_tags>
SpectreData<ThermodynamicDim>::variables(
    const tnsr::I<DataVector, 3, Frame::Inertial>& x, const Mesh<3>& mesh,
    const InverseJacobian<DataVector, 3, Frame::ElementLogical,
                          Frame::Inertial>& inv_jacobian,
    background_tags /*meta*/) const {
  // interpolate from spec, take num derivatives, return
  // Shift, lapse spatial metric imported
  auto result = tuples::tagged_tuple_from_typelist<background_tags>{};
  const auto interpolated_vars = interpolate_from_spectre(x);
  const auto& spatial_metric =
      get<gr::Tags::SpatialMetric<DataVector, 3>>(interpolated_vars);
  const auto spatial_metric_determinant_and_inverse =
      determinant_and_inverse(spatial_metric);
  const auto& inv_spatial_metric =
      spatial_metric_determinant_and_inverse.second;
  get<gr::Tags::InverseSpatialMetric<DataVector, 3>>(result) =
      inv_spatial_metric;
  const auto sqrt_det_spatial_metric = Scalar<DataVector>{
      sqrt(get(spatial_metric_determinant_and_inverse.first))};
  const auto deriv_sqrt_det_spatial_metric =
      partial_derivative(sqrt_det_spatial_metric, mesh, inv_jacobian);
  // Get the one contracted Christoffel needed for fluxes
  const auto spatial_christoffel_second_kind_contracted =
      tenex::evaluate<ti::i>(deriv_sqrt_det_spatial_metric(ti::i) /
                             sqrt_det_spatial_metric());
  get<gr::Tags::SpatialChristoffelSecondKindContracted<DataVector, 3>>(result) =
      spatial_christoffel_second_kind_contracted;
  get<gr::Tags::Lapse<DataVector>>(result) =
      get<gr::Tags::Lapse<DataVector>>(interpolated_vars);

  // Get Lapse and shift derivatives
  get<::Tags::deriv<gr::Tags::Lapse<DataVector>,
                    tmpl::integral_constant<size_t, 3>, Frame::Inertial>>(
      result) =
      partial_derivative(get<gr::Tags::Lapse<DataVector>>(interpolated_vars),
                         mesh, inv_jacobian);
  get<gr::Tags::Shift<DataVector, 3>>(result) =
      get<gr::Tags::Shift<DataVector, 3>>(interpolated_vars);
  get<::Tags::deriv<gr::Tags::Shift<DataVector, 3>,
                    tmpl::integral_constant<size_t, 3>, Frame::Inertial>>(
      result) =
      partial_derivative(get<gr::Tags::Shift<DataVector, 3>>(interpolated_vars),
                         mesh, inv_jacobian);
  // Get the rotational shift + deriv of log lapse over enthalpy + stress
  const auto spatial_rotational_killing_vector =
      hydro::initial_data::irrotational_bns::spatial_rotational_killing_vector(
          x, orbital_angular_velocity_);
  const auto rotational_shift =
      hydro::initial_data::irrotational_bns::rotational_shift(
          get<gr::Tags::Shift<DataVector, 3>>(interpolated_vars),
          spatial_rotational_killing_vector);
  get<Tags::RotationalShift<DataVector>>(result) = rotational_shift;
  auto rest_mass_density =
      get<hydro::Tags::RestMassDensity<DataVector>>(interpolated_vars);
  get(rest_mass_density) += 1.0e-16;
  get<hydro::Tags::RestMassDensity<DataVector>>(result) = rest_mass_density;
  const DataVector enthalpy_density = select(
      step_function(get(rest_mass_density) - 0.0),
      get(equation_of_state_->pressure_from_density(rest_mass_density)) +
          get(rest_mass_density) *
              (1.0 +
               get(equation_of_state_->specific_internal_energy_from_density(
                   rest_mass_density))),
      make_with_value<DataVector>(
          get(rest_mass_density),
          equation_of_state_->specific_enthalpy_lower_bound()));
  const auto deriv_log_lapse_times_density_over_specific_enthalpy =
      partial_derivative(
          Scalar<DataVector>{
              log(get(get<gr::Tags::Lapse<DataVector>>(interpolated_vars)) *
                  square(get(rest_mass_density)) / enthalpy_density)},
          mesh, inv_jacobian);
  get<Tags::DerivLogLapseTimesDensityOverSpecificEnthalpy<DataVector>>(result) =
      deriv_log_lapse_times_density_over_specific_enthalpy;
  const auto rotational_shift_stress =
      hydro::initial_data::irrotational_bns::rotational_shift_stress(
          rotational_shift,
          get<gr::Tags::Lapse<DataVector>>(interpolated_vars));
  get<Tags::RotationalShiftStress<DataVector>>(result) =
      rotational_shift_stress;
  return result;
}

template class SpectreData<1>;
}  // namespace BnsInitialData::AnalyticData
