// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "PointwiseFunctions/AnalyticData/GrMhd/SlabJet.hpp"

#include <cmath>
#include <ostream>
#include <pup.h>

#include "DataStructures/DataBox/DataBoxTag.hpp"
#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/EagerMath/DotProduct.hpp"
#include "PointwiseFunctions/Hydro/LorentzFactor.hpp"
#include "PointwiseFunctions/Hydro/SpecificEnthalpy.hpp"
#include "PointwiseFunctions/Hydro/Tags.hpp"
#include "Utilities/ConstantExpressions.hpp"
#include "Utilities/ContainerHelpers.hpp"
#include "Utilities/GenerateInstantiations.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/MakeWithValue.hpp"
#include "Utilities/Math.hpp"
#include "Utilities/Serialization/PupStlCpp11.hpp"

namespace {
template <typename DataType>
Scalar<DataType> compute_piecewise(const tnsr::I<DataType, 3>& x,
                                   const double inlet_radius,
                                   const double ambient_value,
                                   const double jet_value) {
  auto result = make_with_value<Scalar<DataType>>(x, ambient_value);
  for (size_t i = 0; i < get_size(get(result)); ++i) {
    if (get_element(get<0>(x), i) <= 0. and
        abs(get_element(get<1>(x), i)) <= inlet_radius) {
      get_element(get(result), i) = jet_value;
    }
  }
  return result;
}

template <typename DataType>
tnsr::I<DataType, 3> compute_piecewise_vector(
    const tnsr::I<DataType, 3>& x, const double inlet_radius,
    const std::array<double, 3>& ambient_value,
    const std::array<double, 3>& jet_value) {
  auto result = make_with_value<tnsr::I<DataType, 3>>(x, 0.);
  for (size_t i = 0; i < get_size(get<0>(result)); ++i) {
    if (get_element(get<0>(x), i) <= 0. and
        abs(get_element(get<1>(x), i)) <= inlet_radius) {
      get_element(get<0>(result), i) = jet_value[0];
      get_element(get<1>(result), i) = jet_value[1];
      get_element(get<2>(result), i) = jet_value[2];
    } else {
      get_element(get<0>(result), i) = ambient_value[0];
      get_element(get<1>(result), i) = ambient_value[1];
      get_element(get<2>(result), i) = ambient_value[2];
    }
  }
  return result;
}
}  // namespace

namespace grmhd::AnalyticData {

SlabJet::SlabJet(double adiabatic_index, double ambient_density,
                 double ambient_pressure, double ambient_electron_fraction,
                 double jet_density, double jet_pressure,
                 double jet_electron_fraction,
                 std::array<double, 3> jet_velocity, double inlet_radius,
                 std::array<double, 3> magnetic_field)
    : equation_of_state_(adiabatic_index),
      ambient_density_(ambient_density),
      ambient_pressure_(ambient_pressure),
      ambient_electron_fraction_(ambient_electron_fraction),
      jet_density_(jet_density),
      jet_pressure_(jet_pressure),
      jet_electron_fraction_(jet_electron_fraction),
      jet_velocity_(jet_velocity),
      inlet_radius_(inlet_radius),
      magnetic_field_(magnetic_field) {}

std::unique_ptr<evolution::initial_data::InitialData> SlabJet::get_clone()
    const {
  return std::make_unique<SlabJet>(*this);
}

SlabJet::SlabJet(CkMigrateMessage* msg) : InitialData(msg) {}

void SlabJet::pup(PUP::er& p) {
  InitialData::pup(p);
  p | equation_of_state_;
  p | background_spacetime_;
  p | ambient_density_;
  p | ambient_pressure_;
  p | ambient_electron_fraction_;
  p | jet_density_;
  p | jet_pressure_;
  p | jet_electron_fraction_;
  p | jet_velocity_;
  p | inlet_radius_;
  p | magnetic_field_;
}

template <typename DataType>
tuples::TaggedTuple<hydro::Tags::RestMassDensity<DataType>> SlabJet::variables(
    const tnsr::I<DataType, 3>& x,
    tmpl::list<hydro::Tags::RestMassDensity<DataType>> /*meta*/) const {
  return compute_piecewise(x, inlet_radius_, ambient_density_, jet_density_);
}

template <typename DataType>
tuples::TaggedTuple<hydro::Tags::ElectronFraction<DataType>> SlabJet::variables(
    const tnsr::I<DataType, 3>& x,
    tmpl::list<hydro::Tags::ElectronFraction<DataType>> /*meta*/) const {
  return compute_piecewise(x, inlet_radius_, ambient_electron_fraction_,
                           jet_electron_fraction_);
}

template <typename DataType>
tuples::TaggedTuple<hydro::Tags::SpatialVelocity<DataType, 3>>
SlabJet::variables(
    const tnsr::I<DataType, 3>& x,
    tmpl::list<hydro::Tags::SpatialVelocity<DataType, 3>> /*meta*/) const {
  return compute_piecewise_vector(
      x, inlet_radius_, std::array<double, 3>{{0., 0., 0.}}, jet_velocity_);
}

template <typename DataType>
tuples::TaggedTuple<hydro::Tags::SpecificInternalEnergy<DataType>>
SlabJet::variables(
    const tnsr::I<DataType, 3>& x,
    tmpl::list<hydro::Tags::SpecificInternalEnergy<DataType>> /*meta*/) const {
  using density_tag = hydro::Tags::RestMassDensity<DataType>;
  using pressure_tag = hydro::Tags::Pressure<DataType>;
  const auto data = variables(x, tmpl::list<density_tag, pressure_tag>{});
  return equation_of_state_.specific_internal_energy_from_density_and_pressure(
      get<density_tag>(data), get<pressure_tag>(data));
}

template <typename DataType>
tuples::TaggedTuple<hydro::Tags::Pressure<DataType>> SlabJet::variables(
    const tnsr::I<DataType, 3>& x,
    tmpl::list<hydro::Tags::Pressure<DataType>> /*meta*/) const {
  return compute_piecewise(x, inlet_radius_, ambient_pressure_, jet_pressure_);
}

template <typename DataType>
tuples::TaggedTuple<hydro::Tags::MagneticField<DataType, 3>> SlabJet::variables(
    const tnsr::I<DataType, 3>& x,
    tmpl::list<hydro::Tags::MagneticField<DataType, 3>> /*meta*/) const {
  auto magnetic_field = make_with_value<tnsr::I<DataType, 3>>(x, 0.);
  get<0>(magnetic_field) = magnetic_field_[0];
  get<1>(magnetic_field) = magnetic_field_[1];
  get<2>(magnetic_field) = magnetic_field_[2];
  return {std::move(magnetic_field)};
}

template <typename DataType>
tuples::TaggedTuple<hydro::Tags::DivergenceCleaningField<DataType>>
SlabJet::variables(
    const tnsr::I<DataType, 3>& x,
    tmpl::list<hydro::Tags::DivergenceCleaningField<DataType>> /*meta*/) const {
  return {make_with_value<Scalar<DataType>>(x, 0.)};
}

template <typename DataType>
tuples::TaggedTuple<hydro::Tags::LorentzFactor<DataType>> SlabJet::variables(
    const tnsr::I<DataType, 3>& x,
    tmpl::list<hydro::Tags::LorentzFactor<DataType>> /*meta*/) const {
  const auto spatial_velocity = get<hydro::Tags::SpatialVelocity<DataType, 3>>(
      variables(x, tmpl::list<hydro::Tags::SpatialVelocity<DataType, 3>>{}));
  return {
      hydro::lorentz_factor(dot_product(spatial_velocity, spatial_velocity))};
}

template <typename DataType>
tuples::TaggedTuple<hydro::Tags::SpecificEnthalpy<DataType>> SlabJet::variables(
    const tnsr::I<DataType, 3>& x,
    tmpl::list<hydro::Tags::SpecificEnthalpy<DataType>> /*meta*/) const {
  using density_tag = hydro::Tags::RestMassDensity<DataType>;
  using energy_tag = hydro::Tags::SpecificInternalEnergy<DataType>;
  using pressure_tag = hydro::Tags::Pressure<DataType>;
  const auto data =
      variables(x, tmpl::list<density_tag, energy_tag, pressure_tag>{});
  return hydro::relativistic_specific_enthalpy(
      get<density_tag>(data), get<energy_tag>(data), get<pressure_tag>(data));
}

PUP::able::PUP_ID SlabJet::my_PUP_ID = 0;

bool operator==(const SlabJet& lhs, const SlabJet& rhs) {
  return lhs.equation_of_state_ == rhs.equation_of_state_ and
         lhs.ambient_density_ == rhs.ambient_density_ and
         lhs.ambient_pressure_ == rhs.ambient_pressure_ and
         lhs.ambient_electron_fraction_ == rhs.ambient_electron_fraction_ and
         lhs.jet_density_ == rhs.jet_density_ and
         lhs.jet_pressure_ == rhs.jet_pressure_ and
         lhs.jet_electron_fraction_ == rhs.jet_electron_fraction_ and
         lhs.jet_velocity_ == rhs.jet_velocity_ and
         lhs.inlet_radius_ == rhs.inlet_radius_ and
         lhs.magnetic_field_ == rhs.magnetic_field_;
}

bool operator!=(const SlabJet& lhs, const SlabJet& rhs) {
  return not(lhs == rhs);
}

#define DTYPE(data) BOOST_PP_TUPLE_ELEM(0, data)
#define TAG(data) BOOST_PP_TUPLE_ELEM(1, data)

#define INSTANTIATE_SCALARS(_, data)                        \
  template tuples::TaggedTuple < TAG(data) < DTYPE(data) >> \
      SlabJet::variables(const tnsr::I<DTYPE(data), 3>&,    \
                         tmpl::list < TAG(data) < DTYPE(data) >>) const;

GENERATE_INSTANTIATIONS(
    INSTANTIATE_SCALARS, (double, DataVector),
    (hydro::Tags::RestMassDensity, hydro::Tags::ElectronFraction,
     hydro::Tags::SpecificInternalEnergy, hydro::Tags::Pressure,
     hydro::Tags::DivergenceCleaningField, hydro::Tags::LorentzFactor,
     hydro::Tags::SpecificEnthalpy))

#define INSTANTIATE_VECTORS(_, data)                                   \
  template tuples::TaggedTuple < TAG(data) < DTYPE(data),              \
      3 >> SlabJet::variables(const tnsr::I<DTYPE(data), 3>&,          \
                              tmpl::list < TAG(data) < DTYPE(data), 3, \
                              Frame::Inertial >>) const;

GENERATE_INSTANTIATIONS(INSTANTIATE_VECTORS, (double, DataVector),
                        (hydro::Tags::SpatialVelocity,
                         hydro::Tags::MagneticField))

#undef DTYPE
#undef TAG
#undef INSTANTIATE_SCALARS
#undef INSTANTIATE_VECTORS
}  // namespace grmhd::AnalyticData
