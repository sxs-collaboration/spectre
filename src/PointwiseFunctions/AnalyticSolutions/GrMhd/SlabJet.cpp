// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "PointwiseFunctions/AnalyticSolutions/GrMhd/SlabJet.hpp"

#include <cmath>
#include <ostream>
#include <pup.h>

#include "DataStructures/DataBox/DataBoxTag.hpp"
#include "DataStructures/DataVector.hpp"  // IWYU pragma: keep
#include "Parallel/PupStlCpp11.hpp"
#include "PointwiseFunctions/Hydro/LorentzFactor.hpp"
#include "Utilities/ConstantExpressions.hpp"  // IWYU pragma: keep
#include "Utilities/GenerateInstantiations.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/MakeWithValue.hpp"
#include "Utilities/Math.hpp"  // IWYU pragma: keep

namespace {
template <typename DataType>
Scalar<DataType> compute_piecewise(
    const tnsr::I<DataType, 3, Frame::Inertial>& x, const double inlet_radius,
    const double& ambient_value, const double& jet_value) noexcept;

template <>
Scalar<double> compute_piecewise(const tnsr::I<double, 3, Frame::Inertial>& x,
                                 const double inlet_radius,
                                 const double& ambient_value,
                                 const double& jet_value) noexcept {
  auto piecewise_scalar = make_with_value<Scalar<double>>(x, ambient_value);
  if (get<0>(x) <= 0. and abs(get<1>(x)) <= inlet_radius) {
    get(piecewise_scalar) = jet_value;
  }
  return piecewise_scalar;
}

template <>
Scalar<DataVector> compute_piecewise(
    const tnsr::I<DataVector, 3, Frame::Inertial>& x, const double inlet_radius,
    const double& ambient_value, const double& jet_value) noexcept {
  auto piecewise_scalar = make_with_value<Scalar<DataVector>>(x, ambient_value);
  for (size_t i = 0; i < get(piecewise_scalar).size(); i++) {
    if (get<0>(x)[i] <= 0. and abs(get<1>(x)[i]) <= inlet_radius) {
      get(piecewise_scalar)[i] = jet_value;
    }
  }
  return piecewise_scalar;
}

template <typename DataType>
tnsr::I<DataType, 3, Frame::Inertial> compute_piecewise_vector(
    const tnsr::I<DataType, 3, Frame::Inertial>& x, const double inlet_radius,
    const std::array<double, 3>& ambient_value,
    const std::array<double, 3>& jet_value) noexcept;

template <>
tnsr::I<double, 3, Frame::Inertial> compute_piecewise_vector(
    const tnsr::I<double, 3, Frame::Inertial>& x, const double inlet_radius,
    const std::array<double, 3>& ambient_value,
    const std::array<double, 3>& jet_value) noexcept {
  auto piecewise_vector =
      make_with_value<tnsr::I<double, 3, Frame::Inertial>>(x, 0.);
  if (get<0>(x) <= 0. and abs(get<1>(x)) <= inlet_radius) {
    get<0>(piecewise_vector) = jet_value[0];
    get<1>(piecewise_vector) = jet_value[1];
    get<2>(piecewise_vector) = jet_value[2];
  } else {
    get<0>(piecewise_vector) = ambient_value[0];
    get<1>(piecewise_vector) = ambient_value[1];
    get<2>(piecewise_vector) = ambient_value[2];
  }
  return piecewise_vector;
}

template <>
tnsr::I<DataVector, 3, Frame::Inertial> compute_piecewise_vector(
    const tnsr::I<DataVector, 3, Frame::Inertial>& x, const double inlet_radius,
    const std::array<double, 3>& ambient_value,
    const std::array<double, 3>& jet_value) noexcept {
  auto piecewise_vector =
      make_with_value<tnsr::I<DataVector, 3, Frame::Inertial>>(x, 0.);
  for (size_t i = 0; i < get<0>(piecewise_vector).size(); i++) {
    if (get<0>(x)[i] <= 0. and abs(get<1>(x)[i]) <= inlet_radius) {
      get<0>(piecewise_vector)[i] = jet_value[0];
      get<1>(piecewise_vector)[i] = jet_value[1];
      get<2>(piecewise_vector)[i] = jet_value[2];
    } else {
      get<0>(piecewise_vector)[i] = ambient_value[0];
      get<1>(piecewise_vector)[i] = ambient_value[1];
      get<2>(piecewise_vector)[i] = ambient_value[2];
    }
  }
  return piecewise_vector;
}
}  // namespace

/// \cond
namespace grmhd {
namespace Solutions {

SlabJet::SlabJet(AmbientDensity::type ambient_density,
                 AmbientPressure::type ambient_pressure,
                 JetDensity::type jet_density, JetPressure::type jet_pressure,
                 JetVelocity::type jet_velocity, InletRadius::type inlet_radius,
                 MagneticField::type magnetic_field) noexcept
    : ambient_density_(ambient_density),
      ambient_pressure_(ambient_pressure),
      jet_density_(jet_density),
      jet_pressure_(jet_pressure),
      jet_velocity_(jet_velocity),
      inlet_radius_(inlet_radius),
      magnetic_field_(magnetic_field) {}

void SlabJet::pup(PUP::er& p) noexcept {
  p | ambient_density_;
  p | ambient_pressure_;
  p | jet_density_;
  p | jet_pressure_;
  p | jet_velocity_;
  p | inlet_radius_;
  p | magnetic_field_;
  p | equation_of_state_;
  p | background_spacetime_;
}

template <typename DataType>
tuples::TaggedTuple<hydro::Tags::RestMassDensity<DataType>> SlabJet::variables(
    const tnsr::I<DataType, 3, Frame::Inertial>& x, double /*t*/,
    tmpl::list<hydro::Tags::RestMassDensity<DataType>> /*meta*/) const
    noexcept {
  return compute_piecewise(x, inlet_radius_, ambient_density_, jet_density_);
}

template <typename DataType>
tuples::TaggedTuple<hydro::Tags::SpatialVelocity<DataType, 3>>
SlabJet::variables(
    const tnsr::I<DataType, 3, Frame::Inertial>& x, double /*t*/,
    tmpl::list<hydro::Tags::SpatialVelocity<DataType, 3>> /*meta*/) const
    noexcept {
  return compute_piecewise_vector(
      x, inlet_radius_, std::array<double, 3>{{0., 0., 0.}}, jet_velocity_);
}

template <typename DataType>
tuples::TaggedTuple<hydro::Tags::SpecificInternalEnergy<DataType>>
SlabJet::variables(
    const tnsr::I<DataType, 3, Frame::Inertial>& x, double t,
    tmpl::list<hydro::Tags::SpecificInternalEnergy<DataType>> /*meta*/) const
    noexcept {
  return equation_of_state_.specific_internal_energy_from_density_and_pressure(
      get<hydro::Tags::RestMassDensity<DataType>>(variables(
          x, t, tmpl::list<hydro::Tags::RestMassDensity<DataType>>{})),
      get<hydro::Tags::Pressure<DataType>>(
          variables(x, t, tmpl::list<hydro::Tags::Pressure<DataType>>{})));
}

template <typename DataType>
tuples::TaggedTuple<hydro::Tags::Pressure<DataType>> SlabJet::variables(
    const tnsr::I<DataType, 3, Frame::Inertial>& x, double /*t*/,
    tmpl::list<hydro::Tags::Pressure<DataType>> /*meta*/) const noexcept {
  return compute_piecewise(x, inlet_radius_, ambient_pressure_, jet_pressure_);
  ;
}

template <typename DataType>
tuples::TaggedTuple<hydro::Tags::MagneticField<DataType, 3>> SlabJet::variables(
    const tnsr::I<DataType, 3, Frame::Inertial>& x, double /*t*/,
    tmpl::list<hydro::Tags::MagneticField<DataType, 3>> /*meta*/) const
    noexcept {
  auto magnetic_field =
      make_with_value<db::item_type<hydro::Tags::MagneticField<DataType, 3>>>(
          x, 0.);
  get<0>(magnetic_field) = make_with_value<DataType>(x, magnetic_field_[0]);
  get<1>(magnetic_field) = make_with_value<DataType>(x, magnetic_field_[1]);
  get<2>(magnetic_field) = make_with_value<DataType>(x, magnetic_field_[2]);
  return {std::move(magnetic_field)};
}

template <typename DataType>
tuples::TaggedTuple<hydro::Tags::DivergenceCleaningField<DataType>>
SlabJet::variables(
    const tnsr::I<DataType, 3, Frame::Inertial>& x, double /*t*/,
    tmpl::list<hydro::Tags::DivergenceCleaningField<DataType>> /*meta*/) const
    noexcept {
  return {make_with_value<
      db::item_type<hydro::Tags::DivergenceCleaningField<DataType>>>(x, 0.0)};
}

template <typename DataType>
tuples::TaggedTuple<hydro::Tags::LorentzFactor<DataType>> SlabJet::variables(
    const tnsr::I<DataType, 3, Frame::Inertial>& x, double t,
    tmpl::list<hydro::Tags::LorentzFactor<DataType>> /*meta*/) const noexcept {
  const auto spatial_velocity = get<hydro::Tags::SpatialVelocity<DataType, 3>>(
      variables(x, t, tmpl::list<hydro::Tags::SpatialVelocity<DataType, 3>>{}));
  const Scalar<DataType> spatial_velocity_squared{
      square(get<0>(spatial_velocity)) + square(get<1>(spatial_velocity)) +
      square(get<2>(spatial_velocity))};
  return {hydro::lorentz_factor(spatial_velocity_squared)};
}

template <typename DataType>
tuples::TaggedTuple<hydro::Tags::SpecificEnthalpy<DataType>> SlabJet::variables(
    const tnsr::I<DataType, 3, Frame::Inertial>& x, double t,
    tmpl::list<hydro::Tags::SpecificEnthalpy<DataType>> /*meta*/) const
    noexcept {
  return equation_of_state_.specific_enthalpy_from_density_and_energy(
      get<hydro::Tags::RestMassDensity<DataType>>(variables(
          x, t, tmpl::list<hydro::Tags::RestMassDensity<DataType>>{})),
      get<hydro::Tags::SpecificInternalEnergy<DataType>>(variables(
          x, t, tmpl::list<hydro::Tags::SpecificInternalEnergy<DataType>>{})));
}

bool operator==(const SlabJet& lhs, const SlabJet& rhs) noexcept {
  return lhs.ambient_density_ == rhs.ambient_density_ and
         lhs.ambient_pressure_ == rhs.ambient_pressure_ and
         lhs.jet_density_ == rhs.jet_density_ and
         lhs.jet_pressure_ == rhs.jet_pressure_ and
         lhs.jet_velocity_ == rhs.jet_velocity_ and
         lhs.inlet_radius_ == rhs.inlet_radius_ and
         lhs.magnetic_field_ == rhs.magnetic_field_;
}

bool operator!=(const SlabJet& lhs, const SlabJet& rhs) noexcept {
  return not(lhs == rhs);
}

#define DTYPE(data) BOOST_PP_TUPLE_ELEM(0, data)
#define TAG(data) BOOST_PP_TUPLE_ELEM(1, data)

#define INSTANTIATE_SCALARS(_, data)                                      \
  template tuples::TaggedTuple<TAG(data) < DTYPE(data)>>                  \
      SlabJet::variables(const tnsr::I<DTYPE(data), 3, Frame::Inertial>&, \
                         double, tmpl::list<TAG(data) < DTYPE(data)>>)    \
          const noexcept;

GENERATE_INSTANTIATIONS(
    INSTANTIATE_SCALARS, (double, DataVector),
    (hydro::Tags::RestMassDensity, hydro::Tags::SpecificInternalEnergy,
     hydro::Tags::Pressure, hydro::Tags::DivergenceCleaningField,
     hydro::Tags::LorentzFactor, hydro::Tags::SpecificEnthalpy))

#define INSTANTIATE_VECTORS(_, data)                                \
  template tuples::TaggedTuple<TAG(data) < DTYPE(data), 3>>         \
      SlabJet::variables(                                           \
          const tnsr::I<DTYPE(data), 3, Frame::Inertial>&, double,  \
          tmpl::list<TAG(data) < DTYPE(data), 3, Frame::Inertial>>) \
          const noexcept;

GENERATE_INSTANTIATIONS(INSTANTIATE_VECTORS, (double, DataVector),
                        (hydro::Tags::SpatialVelocity,
                         hydro::Tags::MagneticField))

#undef DTYPE
#undef TAG
#undef INSTANTIATE_SCALARS
#undef INSTANTIATE_VECTORS
}  // namespace Solutions
}  // namespace grmhd
/// \endcond
