// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "PointwiseFunctions/AnalyticData/GrMhd/MagneticFieldLoop.hpp"

#include <cmath>  // IWYU pragma: keep
#include <cstddef>
#include <ostream>
#include <pup.h>

#include "DataStructures/DataVector.hpp"  // IWYU pragma: keep
#include "DataStructures/Tensor/EagerMath/DotProduct.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Parallel/PupStlCpp11.hpp"
#include "PointwiseFunctions/Hydro/LorentzFactor.hpp"
#include "PointwiseFunctions/Hydro/Tags.hpp"
#include "Utilities/ConstantExpressions.hpp"  // IWYU pragma: keep
#include "Utilities/ContainerHelpers.hpp"
#include "Utilities/GenerateInstantiations.hpp"
#include "Utilities/MakeWithValue.hpp"
#include "Utilities/Math.hpp"  // IWYU pragma: keep
#include "Utilities/StdArrayHelpers.hpp"
#include "Utilities/StdHelpers.hpp"

// IWYU pragma: no_include <complex>

/// \cond
namespace grmhd {
namespace AnalyticData {

MagneticFieldLoop::MagneticFieldLoop(
    const double pressure, const double rest_mass_density,
    const double adiabatic_index,
    const std::array<double, 3>& advection_velocity,
    const double magnetic_field_magnitude, const double inner_radius,
    const double outer_radius, const OptionContext& context)
    : pressure_(pressure),
      rest_mass_density_(rest_mass_density),
      adiabatic_index_(adiabatic_index),
      advection_velocity_(advection_velocity),
      magnetic_field_magnitude_(magnetic_field_magnitude),
      inner_radius_(inner_radius),
      outer_radius_(outer_radius),
      equation_of_state_{adiabatic_index_} {
  if (magnitude(advection_velocity) >= 1.0) {
    PARSE_ERROR(context, "MagneticFieldLoop: superluminal AdvectionVelocity = "
                             << advection_velocity);
  }
  if (inner_radius >= outer_radius) {
    PARSE_ERROR(context, "MagneticFieldLoop: InnerRadius of "
                             << inner_radius
                             << " is not less than OuterRadius of "
                             << outer_radius);
  }
}

void MagneticFieldLoop::pup(PUP::er& p) noexcept {
  p | pressure_;
  p | rest_mass_density_;
  p | adiabatic_index_;
  p | advection_velocity_;
  p | magnetic_field_magnitude_;
  p | inner_radius_;
  p | outer_radius_;
  p | equation_of_state_;
  p | background_spacetime_;
}

template <typename DataType>
tuples::TaggedTuple<hydro::Tags::RestMassDensity<DataType>>
MagneticFieldLoop::variables(
    const tnsr::I<DataType, 3>& x,
    tmpl::list<hydro::Tags::RestMassDensity<DataType>> /*meta*/) const
    noexcept {
  return {make_with_value<Scalar<DataType>>(x, rest_mass_density_)};
}

template <typename DataType>
tuples::TaggedTuple<hydro::Tags::SpatialVelocity<DataType, 3>>
MagneticFieldLoop::variables(
    const tnsr::I<DataType, 3>& x,
    tmpl::list<hydro::Tags::SpatialVelocity<DataType, 3>> /*meta*/) const
    noexcept {
  auto result = make_with_value<tnsr::I<DataType, 3>>(x, 0.0);
  get<0>(result) = advection_velocity_[0];
  get<1>(result) = advection_velocity_[1];
  get<2>(result) = advection_velocity_[2];
  return {std::move(result)};
}

template <typename DataType>
tuples::TaggedTuple<hydro::Tags::SpecificInternalEnergy<DataType>>
MagneticFieldLoop::variables(
    const tnsr::I<DataType, 3>& x,
    tmpl::list<hydro::Tags::SpecificInternalEnergy<DataType>> /*meta*/) const
    noexcept {
  return equation_of_state_.specific_internal_energy_from_density_and_pressure(
      get<hydro::Tags::RestMassDensity<DataType>>(
          variables(x, tmpl::list<hydro::Tags::RestMassDensity<DataType>>{})),
      get<hydro::Tags::Pressure<DataType>>(
          variables(x, tmpl::list<hydro::Tags::Pressure<DataType>>{})));
}

template <typename DataType>
tuples::TaggedTuple<hydro::Tags::Pressure<DataType>>
MagneticFieldLoop::variables(
    const tnsr::I<DataType, 3>& x,
    tmpl::list<hydro::Tags::Pressure<DataType>> /*meta*/) const noexcept {
  return {make_with_value<Scalar<DataType>>(x, pressure_)};
}

template <typename DataType>
tuples::TaggedTuple<hydro::Tags::MagneticField<DataType, 3>>
MagneticFieldLoop::variables(
    const tnsr::I<DataType, 3>& x,
    tmpl::list<hydro::Tags::MagneticField<DataType, 3>> /*meta*/) const
    noexcept {
  auto result = make_with_value<tnsr::I<DataType, 3>>(x, 0.0);
  const DataType cylindrical_radius =
      sqrt(square(get<0>(x)) + square(get<1>(x)));
  for (size_t s = 0; s < get_size(cylindrical_radius); ++s) {
    const double current_radius = get_element(cylindrical_radius, s);
    // The data is ill-posed at the origin, so it is not clear what to do...
    if (current_radius <= outer_radius_ and current_radius > inner_radius_) {
      get_element(get<0>(result), s) = -magnetic_field_magnitude_ *
                                       get_element(get<1>(x), s) /
                                       current_radius;
      get_element(get<1>(result), s) = magnetic_field_magnitude_ *
                                       get_element(get<0>(x), s) /
                                       current_radius;
    }
    if (current_radius <= inner_radius_) {
      get_element(get<0>(result), s) = -magnetic_field_magnitude_ *
                                       get_element(get<1>(x), s) /
                                       inner_radius_;
      get_element(get<1>(result), s) =
          magnetic_field_magnitude_ * get_element(get<0>(x), s) / inner_radius_;
    }
  }
  return {std::move(result)};
}

template <typename DataType>
tuples::TaggedTuple<hydro::Tags::DivergenceCleaningField<DataType>>
MagneticFieldLoop::variables(
    const tnsr::I<DataType, 3>& x,
    tmpl::list<hydro::Tags::DivergenceCleaningField<DataType>> /*meta*/) const
    noexcept {
  return {make_with_value<Scalar<DataType>>(x, 0.0)};
}

template <typename DataType>
tuples::TaggedTuple<hydro::Tags::LorentzFactor<DataType>>
MagneticFieldLoop::variables(
    const tnsr::I<DataType, 3>& x,
    tmpl::list<hydro::Tags::LorentzFactor<DataType>> /*meta*/) const noexcept {
  using velocity_tag = hydro::Tags::SpatialVelocity<DataType, 3>;
  const auto velocity =
      get<velocity_tag>(variables(x, tmpl::list<velocity_tag>{}));
  return {hydro::lorentz_factor(dot_product(velocity, velocity))};
}

template <typename DataType>
tuples::TaggedTuple<hydro::Tags::SpecificEnthalpy<DataType>>
MagneticFieldLoop::variables(
    const tnsr::I<DataType, 3>& x,
    tmpl::list<hydro::Tags::SpecificEnthalpy<DataType>> /*meta*/) const
    noexcept {
  return equation_of_state_.specific_enthalpy_from_density_and_energy(
      get<hydro::Tags::RestMassDensity<DataType>>(
          variables(x, tmpl::list<hydro::Tags::RestMassDensity<DataType>>{})),
      get<hydro::Tags::SpecificInternalEnergy<DataType>>(variables(
          x, tmpl::list<hydro::Tags::SpecificInternalEnergy<DataType>>{})));
}

bool operator==(const MagneticFieldLoop& lhs,
                const MagneticFieldLoop& rhs) noexcept {
  // there is no comparison operator for the EoS, but should be okay as
  // the adiabatic_indexs are compared
  return lhs.pressure_ == rhs.pressure_ and
         lhs.rest_mass_density_ == rhs.rest_mass_density_ and
         lhs.adiabatic_index_ == rhs.adiabatic_index_ and
         lhs.advection_velocity_ == rhs.advection_velocity_ and
         lhs.magnetic_field_magnitude_ == rhs.magnetic_field_magnitude_ and
         lhs.inner_radius_ == rhs.inner_radius_ and
         lhs.outer_radius_ == rhs.outer_radius_ and
         lhs.background_spacetime_ == rhs.background_spacetime_;
}

bool operator!=(const MagneticFieldLoop& lhs,
                const MagneticFieldLoop& rhs) noexcept {
  return not(lhs == rhs);
}

#define DTYPE(data) BOOST_PP_TUPLE_ELEM(0, data)
#define TAG(data) BOOST_PP_TUPLE_ELEM(1, data)

#define INSTANTIATE_SCALARS(_, data)                     \
  template tuples::TaggedTuple<TAG(data) < DTYPE(data)>> \
      MagneticFieldLoop::variables(                      \
          const tnsr::I<DTYPE(data), 3>& x,              \
          tmpl::list<TAG(data) < DTYPE(data)>> /*meta*/) const noexcept;

GENERATE_INSTANTIATIONS(
    INSTANTIATE_SCALARS, (double, DataVector),
    (hydro::Tags::RestMassDensity, hydro::Tags::SpecificInternalEnergy,
     hydro::Tags::Pressure, hydro::Tags::DivergenceCleaningField,
     hydro::Tags::LorentzFactor, hydro::Tags::SpecificEnthalpy))

#define INSTANTIATE_VECTORS(_, data)                        \
  template tuples::TaggedTuple<TAG(data) < DTYPE(data), 3>> \
      MagneticFieldLoop::variables(                         \
          const tnsr::I<DTYPE(data), 3>& x,                 \
          tmpl::list<TAG(data) < DTYPE(data), 3>> /*meta*/) const noexcept;

GENERATE_INSTANTIATIONS(INSTANTIATE_VECTORS, (double, DataVector),
                        (hydro::Tags::SpatialVelocity,
                         hydro::Tags::MagneticField))

#undef DTYPE
#undef TAG
#undef INSTANTIATE_SCALARS
#undef INSTANTIATE_VECTORS
}  // namespace AnalyticData
}  // namespace grmhd
/// \endcond
