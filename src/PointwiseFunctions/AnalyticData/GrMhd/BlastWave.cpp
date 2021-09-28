// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "PointwiseFunctions/AnalyticData/GrMhd/BlastWave.hpp"

#include <cmath>
#include <ostream>
#include <pup.h>

#include "DataStructures/DataVector.hpp"  // IWYU pragma: keep
#include "DataStructures/Tensor/EagerMath/Magnitude.hpp"
#include "DataStructures/Tensor/Tensor.hpp"  // IWYU pragma: keep
#include "Options/ParseOptions.hpp"
#include "PointwiseFunctions/Hydro/Tags.hpp"
#include "Utilities/ConstantExpressions.hpp"  // IWYU pragma: keep
#include "Utilities/GenerateInstantiations.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/MakeWithValue.hpp"
#include "Utilities/Math.hpp"  // IWYU pragma: keep

namespace {
template <typename DataType>
Scalar<DataType> compute_piecewise(
    const tnsr::I<DataType, 3>& x, const double inner_radius,
    const double outer_radius, const double inner_value,
    const double outer_value,
    const grmhd::AnalyticData::BlastWave::Geometry geometry) {
  DataType radius{};
  if (geometry == grmhd::AnalyticData::BlastWave::Geometry::Cylindrical) {
    radius = sqrt(square(get<0>(x)) + square(get<1>(x)));
  } else {
    radius = get(magnitude(x));
  }
  auto piecewise_scalar = make_with_value<Scalar<DataType>>(x, 0.0);

  // piecewise_scalar should equal inner_value for r < inner_radius,
  // should equal outer_value for r > outer_radius, and should transition
  // in between. Here, use step_function() to compute the result
  // in each region separately.
  get(piecewise_scalar) += step_function(inner_radius - radius) * inner_value;
  get(piecewise_scalar) += step_function(radius - outer_radius) * outer_value;
  get(piecewise_scalar) +=
      // Blaze's step_function() at 0.0 is 1.0. So to get a mask that
      // is the inverse of the masks used above for inner_radius and
      // outer_radius without double-counting points on the boundary,
      // use 1.0 - stepfunction(x) instead of stepfunction(-x) here
      (1.0 - step_function(inner_radius - radius)) *
      (1.0 - step_function(radius - outer_radius)) *
      exp(((-1.0 * radius + inner_radius) * log(outer_value) +
           (radius - outer_radius) * log(inner_value)) /
          (inner_radius - outer_radius));
  return piecewise_scalar;
}
}  // namespace

namespace grmhd::AnalyticData {

BlastWave::BlastWave(const double inner_radius, const double outer_radius,
                     const double inner_density, const double outer_density,
                     const double inner_pressure, const double outer_pressure,
                     const std::array<double, 3>& magnetic_field,
                     const double adiabatic_index, const Geometry geometry,
                     const Options::Context& context)
    : inner_radius_(inner_radius),
      outer_radius_(outer_radius),
      inner_density_(inner_density),
      outer_density_(outer_density),
      inner_pressure_(inner_pressure),
      outer_pressure_(outer_pressure),
      magnetic_field_(magnetic_field),
      adiabatic_index_(adiabatic_index),
      geometry_(geometry),
      equation_of_state_(adiabatic_index) {
  if (inner_radius >= outer_radius) {
    PARSE_ERROR(context,
                "BlastWave expects InnerRadius < OuterRadius, "
                "but InnerRadius = "
                    << inner_radius << " and OuterRadius = " << outer_radius);
  }
}

void BlastWave::pup(PUP::er& p) {
  p | inner_radius_;
  p | outer_radius_;
  p | inner_density_;
  p | outer_density_;
  p | inner_pressure_;
  p | outer_pressure_;
  p | magnetic_field_;
  p | adiabatic_index_;
  p | geometry_;
  p | equation_of_state_;
  p | background_spacetime_;
}

template <typename DataType>
tuples::TaggedTuple<hydro::Tags::RestMassDensity<DataType>>
BlastWave::variables(
    const tnsr::I<DataType, 3>& x,
    tmpl::list<hydro::Tags::RestMassDensity<DataType>> /*meta*/) const {
  return compute_piecewise(x, inner_radius_, outer_radius_, inner_density_,
                           outer_density_, geometry_);
}

template <typename DataType>
tuples::TaggedTuple<hydro::Tags::SpatialVelocity<DataType, 3>>
BlastWave::variables(
    const tnsr::I<DataType, 3>& x,
    tmpl::list<hydro::Tags::SpatialVelocity<DataType, 3>> /*meta*/) const {
  return {make_with_value<tnsr::I<DataType, 3>>(x, 0.0)};
}

template <typename DataType>
tuples::TaggedTuple<hydro::Tags::SpecificInternalEnergy<DataType>>
BlastWave::variables(
    const tnsr::I<DataType, 3>& x,
    tmpl::list<hydro::Tags::SpecificInternalEnergy<DataType>> /*meta*/) const {
  return equation_of_state_.specific_internal_energy_from_density_and_pressure(
      get<hydro::Tags::RestMassDensity<DataType>>(
          variables(x, tmpl::list<hydro::Tags::RestMassDensity<DataType>>{})),
      get<hydro::Tags::Pressure<DataType>>(
          variables(x, tmpl::list<hydro::Tags::Pressure<DataType>>{})));
}

template <typename DataType>
tuples::TaggedTuple<hydro::Tags::Pressure<DataType>> BlastWave::variables(
    const tnsr::I<DataType, 3>& x,
    tmpl::list<hydro::Tags::Pressure<DataType>> /*meta*/) const {
  return compute_piecewise(x, inner_radius_, outer_radius_, inner_pressure_,
                           outer_pressure_, geometry_);
}

template <typename DataType>
tuples::TaggedTuple<hydro::Tags::MagneticField<DataType, 3>>
BlastWave::variables(
    const tnsr::I<DataType, 3>& x,
    tmpl::list<hydro::Tags::MagneticField<DataType, 3>> /*meta*/) const {
  auto magnetic_field = make_with_value<tnsr::I<DataType, 3>>(get<0>(x), 0.0);
  get<0>(magnetic_field) = gsl::at(magnetic_field_, 0);
  get<1>(magnetic_field) = gsl::at(magnetic_field_, 1);
  get<2>(magnetic_field) = gsl::at(magnetic_field_, 2);
  return magnetic_field;
}

template <typename DataType>
tuples::TaggedTuple<hydro::Tags::DivergenceCleaningField<DataType>>
BlastWave::variables(
    const tnsr::I<DataType, 3>& x,
    tmpl::list<hydro::Tags::DivergenceCleaningField<DataType>> /*meta*/) const {
  return {make_with_value<Scalar<DataType>>(x, 0.0)};
}

template <typename DataType>
tuples::TaggedTuple<hydro::Tags::LorentzFactor<DataType>> BlastWave::variables(
    const tnsr::I<DataType, 3>& x,
    tmpl::list<hydro::Tags::LorentzFactor<DataType>> /*meta*/) const {
  return {make_with_value<Scalar<DataType>>(x, 1.0)};
}

template <typename DataType>
tuples::TaggedTuple<hydro::Tags::SpecificEnthalpy<DataType>>
BlastWave::variables(
    const tnsr::I<DataType, 3>& x,
    tmpl::list<hydro::Tags::SpecificEnthalpy<DataType>> /*meta*/) const {
  return equation_of_state_.specific_enthalpy_from_density_and_energy(
      get<hydro::Tags::RestMassDensity<DataType>>(
          variables(x, tmpl::list<hydro::Tags::RestMassDensity<DataType>>{})),
      get<hydro::Tags::SpecificInternalEnergy<DataType>>(variables(
          x, tmpl::list<hydro::Tags::SpecificInternalEnergy<DataType>>{})));
}

bool operator==(const BlastWave& lhs, const BlastWave& rhs) {
  return lhs.inner_radius_ == rhs.inner_radius_ and
         lhs.outer_radius_ == rhs.outer_radius_ and
         lhs.inner_density_ == rhs.inner_density_ and
         lhs.outer_density_ == rhs.outer_density_ and
         lhs.inner_pressure_ == rhs.inner_pressure_ and
         lhs.outer_pressure_ == rhs.outer_pressure_ and
         lhs.magnetic_field_ == rhs.magnetic_field_ and
         lhs.adiabatic_index_ == rhs.adiabatic_index_ and
         lhs.geometry_ == rhs.geometry_;
}

bool operator!=(const BlastWave& lhs, const BlastWave& rhs) {
  return not(lhs == rhs);
}

#define DTYPE(data) BOOST_PP_TUPLE_ELEM(0, data)
#define TAG(data) BOOST_PP_TUPLE_ELEM(1, data)

#define INSTANTIATE_SCALARS(_, data)                                       \
  template tuples::TaggedTuple<TAG(data) < DTYPE(data)> >                  \
      BlastWave::variables(const tnsr::I<DTYPE(data), 3>& x,               \
                           tmpl::list<TAG(data) < DTYPE(data)> > /*meta*/) \
          const;

GENERATE_INSTANTIATIONS(
    INSTANTIATE_SCALARS, (double, DataVector),
    (hydro::Tags::RestMassDensity, hydro::Tags::SpecificInternalEnergy,
     hydro::Tags::Pressure, hydro::Tags::DivergenceCleaningField,
     hydro::Tags::LorentzFactor, hydro::Tags::SpecificEnthalpy))

#define INSTANTIATE_VECTORS(_, data)                                          \
  template tuples::TaggedTuple<TAG(data) < DTYPE(data), 3> >                  \
      BlastWave::variables(const tnsr::I<DTYPE(data), 3>& x,                  \
                           tmpl::list<TAG(data) < DTYPE(data), 3> > /*meta*/) \
          const;

GENERATE_INSTANTIATIONS(INSTANTIATE_VECTORS, (double, DataVector),
                        (hydro::Tags::SpatialVelocity,
                         hydro::Tags::MagneticField))

#undef DTYPE
#undef TAG
#undef INSTANTIATE_SCALARS
#undef INSTANTIATE_VECTORS
}  // namespace grmhd::AnalyticData

template <>
grmhd::AnalyticData::BlastWave::Geometry
Options::create_from_yaml<grmhd::AnalyticData::BlastWave::Geometry>::create<
    void>(const Options::Option& options) {
  const auto type_read = options.parse_as<std::string>();
  if ("Cylindrical" == type_read) {
    return grmhd::AnalyticData::BlastWave::Geometry::Cylindrical;
  } else if ("Spherical" == type_read) {
    return grmhd::AnalyticData::BlastWave::Geometry::Spherical;
  }
  PARSE_ERROR(
      options.context(),
      "Failed to convert \""
          << type_read
          << "\" to Geometry. Must be one of Cylindrical or Spherical.");
}
