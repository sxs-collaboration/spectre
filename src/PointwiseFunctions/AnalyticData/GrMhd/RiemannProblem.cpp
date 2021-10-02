// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "PointwiseFunctions/AnalyticData/GrMhd/RiemannProblem.hpp"

#include <pup.h>

#include "DataStructures/DataVector.hpp"  // IWYU pragma: keep
#include "DataStructures/Tensor/EagerMath/DotProduct.hpp"
#include "DataStructures/Tensor/Tensor.hpp"  // IWYU pragma: keep
#include "PointwiseFunctions/Hydro/LorentzFactor.hpp"
#include "PointwiseFunctions/Hydro/Tags.hpp"
#include "Utilities/ContainerHelpers.hpp"
#include "Utilities/GenerateInstantiations.hpp"
#include "Utilities/MakeWithValue.hpp"
#include "Utilities/Math.hpp"  // IWYU pragma: keep

namespace grmhd::AnalyticData {

RiemannProblem::RiemannProblem(
    const double adiabatic_index, const double left_rest_mass_density,
    const double right_rest_mass_density, const double left_pressure,
    const double right_pressure,
    const std::array<double, 3>& left_spatial_velocity,
    const std::array<double, 3>& right_spatial_velocity,
    const std::array<double, 3>& left_magnetic_field,
    const std::array<double, 3>& right_magnetic_field, const double lapse,
    const double shift)
    : equation_of_state_(adiabatic_index),
      adiabatic_index_(adiabatic_index),
      left_rest_mass_density_(left_rest_mass_density),
      right_rest_mass_density_(right_rest_mass_density),
      left_pressure_(left_pressure),
      right_pressure_(right_pressure),
      left_spatial_velocity_(left_spatial_velocity),
      right_spatial_velocity_(right_spatial_velocity),
      left_magnetic_field_(left_magnetic_field),
      right_magnetic_field_(right_magnetic_field),
      lapse_(lapse),
      shift_(shift) {}

RiemannProblem::RiemannProblem(CkMigrateMessage* /*unused*/) {}

void RiemannProblem::pup(PUP::er& p) {
  p | equation_of_state_;
  p | background_spacetime_;
  p | adiabatic_index_;
  p | left_rest_mass_density_;
  p | right_rest_mass_density_;
  p | left_pressure_;
  p | right_pressure_;
  p | left_spatial_velocity_;
  p | right_spatial_velocity_;
  p | left_magnetic_field_;
  p | right_magnetic_field_;
  p | lapse_;
  p | shift_;
}

template <typename DataType>
tuples::TaggedTuple<hydro::Tags::RestMassDensity<DataType>>
RiemannProblem::variables(
    const tnsr::I<DataType, 3>& x,
    tmpl::list<hydro::Tags::RestMassDensity<DataType>> /*meta*/) const {
  auto mass_density = make_with_value<Scalar<DataType>>(x, 0.0);
  for (size_t i = 0; i < get_size(get<0>(x)); ++i) {
    if (get_element(get<0>(x), i) <= discontinuity_location_) {
      get_element(get(mass_density), i) = left_rest_mass_density_;
    } else {
      get_element(get(mass_density), i) = right_rest_mass_density_;
    }
  }
  return mass_density;
}

template <typename DataType>
tuples::TaggedTuple<hydro::Tags::SpatialVelocity<DataType, 3>>
RiemannProblem::variables(
    const tnsr::I<DataType, 3>& x,
    tmpl::list<hydro::Tags::SpatialVelocity<DataType, 3>> /*meta*/) const {
  auto spatial_velocity =
      make_with_value<tnsr::I<DataType, 3, Frame::Inertial>>(x, 0.0);
  for (size_t i = 0; i < get_size(get<0>(x)); ++i) {
    if (get_element(get<0>(x), i) <= discontinuity_location_) {
      get_element(get<0>(spatial_velocity), i) = left_spatial_velocity_[0];
      get_element(get<1>(spatial_velocity), i) = left_spatial_velocity_[1];
      get_element(get<2>(spatial_velocity), i) = left_spatial_velocity_[2];
    } else {
      get_element(get<0>(spatial_velocity), i) = right_spatial_velocity_[0];
      get_element(get<1>(spatial_velocity), i) = right_spatial_velocity_[1];
      get_element(get<2>(spatial_velocity), i) = right_spatial_velocity_[2];
    }
  }
  return spatial_velocity;
}

template <typename DataType>
tuples::TaggedTuple<hydro::Tags::SpecificInternalEnergy<DataType>>
RiemannProblem::variables(
    const tnsr::I<DataType, 3>& x,
    tmpl::list<hydro::Tags::SpecificInternalEnergy<DataType>> /*meta*/) const {
  return equation_of_state_.specific_internal_energy_from_density_and_pressure(
      get<hydro::Tags::RestMassDensity<DataType>>(
          variables(x, tmpl::list<hydro::Tags::RestMassDensity<DataType>>{})),
      get<hydro::Tags::Pressure<DataType>>(
          variables(x, tmpl::list<hydro::Tags::Pressure<DataType>>{})));
}

template <typename DataType>
tuples::TaggedTuple<hydro::Tags::Pressure<DataType>> RiemannProblem::variables(
    const tnsr::I<DataType, 3>& x,
    tmpl::list<hydro::Tags::Pressure<DataType>> /*meta*/) const {
  auto pressure = make_with_value<Scalar<DataType>>(x, 0.0);
  for (size_t i = 0; i < get_size(get<0>(x)); ++i) {
    if (get_element(get<0>(x), i) <= discontinuity_location_) {
      get_element(get(pressure), i) = left_pressure_;
    } else {
      get_element(get(pressure), i) = right_pressure_;
    }
  }
  return pressure;
}

template <typename DataType>
tuples::TaggedTuple<hydro::Tags::MagneticField<DataType, 3>>
RiemannProblem::variables(
    const tnsr::I<DataType, 3>& x,
    tmpl::list<hydro::Tags::MagneticField<DataType, 3>> /*meta*/) const {
  auto magnetic_field =
      make_with_value<tnsr::I<DataType, 3, Frame::Inertial>>(x, 0.0);
  for (size_t i = 0; i < get_size(get<0>(x)); ++i) {
    if (get_element(get<0>(x), i) <= discontinuity_location_) {
      get_element(get<0>(magnetic_field), i) = left_magnetic_field_[0];
      get_element(get<1>(magnetic_field), i) = left_magnetic_field_[1];
      get_element(get<2>(magnetic_field), i) = left_magnetic_field_[2];
    } else {
      get_element(get<0>(magnetic_field), i) = right_magnetic_field_[0];
      get_element(get<1>(magnetic_field), i) = right_magnetic_field_[1];
      get_element(get<2>(magnetic_field), i) = right_magnetic_field_[2];
    }
  }
  return magnetic_field;
}

template <typename DataType>
tuples::TaggedTuple<hydro::Tags::DivergenceCleaningField<DataType>>
RiemannProblem::variables(
    const tnsr::I<DataType, 3>& x,
    tmpl::list<hydro::Tags::DivergenceCleaningField<DataType>> /*meta*/) const {
  return {make_with_value<Scalar<DataType>>(x, 0.0)};
}

template <typename DataType>
tuples::TaggedTuple<hydro::Tags::LorentzFactor<DataType>>
RiemannProblem::variables(
    const tnsr::I<DataType, 3>& x,
    tmpl::list<hydro::Tags::LorentzFactor<DataType>> /*meta*/) const {
  const auto spatial_velocity = get<hydro::Tags::SpatialVelocity<DataType, 3>>(
      variables(x, tmpl::list<hydro::Tags::SpatialVelocity<DataType, 3>>{}));
  return {
      hydro::lorentz_factor(dot_product(spatial_velocity, spatial_velocity))};
}

template <typename DataType>
tuples::TaggedTuple<hydro::Tags::SpecificEnthalpy<DataType>>
RiemannProblem::variables(
    const tnsr::I<DataType, 3>& x,
    tmpl::list<hydro::Tags::SpecificEnthalpy<DataType>> /*meta*/) const {
  return equation_of_state_.specific_enthalpy_from_density_and_energy(
      get<hydro::Tags::RestMassDensity<DataType>>(
          variables(x, tmpl::list<hydro::Tags::RestMassDensity<DataType>>{})),
      get<hydro::Tags::SpecificInternalEnergy<DataType>>(variables(
          x, tmpl::list<hydro::Tags::SpecificInternalEnergy<DataType>>{})));
}

template <typename DataType>
tuples::TaggedTuple<gr::Tags::Lapse<DataType>> RiemannProblem::variables(
    const tnsr::I<DataType, 3>& x,
    tmpl::list<gr::Tags::Lapse<DataType>> /*meta*/) const {
  return {make_with_value<Scalar<DataType>>(x, lapse_)};
}

template <typename DataType>
tuples::TaggedTuple<gr::Tags::Shift<3, Frame::Inertial, DataType>>
RiemannProblem::variables(
    const tnsr::I<DataType, 3>& x,
    tmpl::list<gr::Tags::Shift<3, Frame::Inertial, DataType>> /*meta*/) const {
  auto shift = make_with_value<tnsr::I<DataType, 3, Frame::Inertial>>(x, 0.0);
  get<0>(shift) = shift_;
  return {std::move(shift)};
}

bool operator==(const RiemannProblem& lhs, const RiemannProblem& rhs) {
  return lhs.adiabatic_index_ == rhs.adiabatic_index_ and
         lhs.left_rest_mass_density_ == rhs.left_rest_mass_density_ and
         lhs.right_rest_mass_density_ == rhs.right_rest_mass_density_ and
         lhs.left_pressure_ == rhs.left_pressure_ and
         lhs.right_pressure_ == rhs.right_pressure_ and
         lhs.left_spatial_velocity_ == rhs.left_spatial_velocity_ and
         lhs.right_spatial_velocity_ == rhs.right_spatial_velocity_ and
         lhs.left_magnetic_field_ == rhs.left_magnetic_field_ and
         lhs.right_magnetic_field_ == rhs.right_magnetic_field_ and
         lhs.lapse_ == rhs.lapse_ and lhs.shift_ == rhs.shift_;
}

bool operator!=(const RiemannProblem& lhs, const RiemannProblem& rhs) {
  return not(lhs == rhs);
}

#define DTYPE(data) BOOST_PP_TUPLE_ELEM(0, data)
#define TAG(data) BOOST_PP_TUPLE_ELEM(1, data)

#define INSTANTIATE_SCALARS(_, data)                      \
  template tuples::TaggedTuple<TAG(data) < DTYPE(data)> > \
      RiemannProblem::variables(                          \
          const tnsr::I<DTYPE(data), 3>& x,               \
          tmpl::list<TAG(data) < DTYPE(data)> > /*meta*/) const;

GENERATE_INSTANTIATIONS(INSTANTIATE_SCALARS, (double, DataVector),
                        (hydro::Tags::RestMassDensity,
                         hydro::Tags::SpecificInternalEnergy,
                         hydro::Tags::Pressure,
                         hydro::Tags::DivergenceCleaningField,
                         hydro::Tags::LorentzFactor,
                         hydro::Tags::SpecificEnthalpy, gr::Tags::Lapse))

#define INSTANTIATE_VECTORS(_, data)                         \
  template tuples::TaggedTuple<TAG(data) < DTYPE(data), 3> > \
      RiemannProblem::variables(                             \
          const tnsr::I<DTYPE(data), 3>& x,                  \
          tmpl::list<TAG(data) < DTYPE(data), 3> > /*meta*/) const;

GENERATE_INSTANTIATIONS(INSTANTIATE_VECTORS, (double, DataVector),
                        (hydro::Tags::SpatialVelocity,
                         hydro::Tags::MagneticField))

#undef DTYPE
#undef TAG
#undef INSTANTIATE_SCALARS
#undef INSTANTIATE_VECTORS

// The GR tags have a different template parameter ordering than the rest of the
// code, so need to instantiate separately.
template tuples::TaggedTuple<gr::Tags::Shift<3, Frame::Inertial, double>>
RiemannProblem::variables(
    const tnsr::I<double, 3>& x,
    tmpl::list<gr::Tags::Shift<3, Frame::Inertial, double>> /*meta*/) const;
template tuples::TaggedTuple<gr::Tags::Shift<3, Frame::Inertial, DataVector>>
RiemannProblem::variables(
    const tnsr::I<DataVector, 3>& x,
    tmpl::list<gr::Tags::Shift<3, Frame::Inertial, DataVector>> /*meta*/) const;
}  // namespace grmhd::AnalyticData
