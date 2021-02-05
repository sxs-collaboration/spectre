// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "PointwiseFunctions/AnalyticData/NewtonianEuler/SodExplosion.hpp"

#include <cmath>
#include <cstddef>
#include <pup.h>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "PointwiseFunctions/Hydro/EquationsOfState/EquationOfState.hpp"
#include "PointwiseFunctions/Hydro/EquationsOfState/IdealFluid.hpp"
#include "Utilities/ContainerHelpers.hpp"
#include "Utilities/MakeWithValue.hpp"
#include "Utilities/Math.hpp"
#include "Utilities/StdArrayHelpers.hpp"

/// \cond
namespace NewtonianEuler::AnalyticData {
template <size_t Dim>
SodExplosion<Dim>::SodExplosion(const double initial_radius,
                                const double inner_mass_density,
                                const double inner_pressure,
                                const double outer_mass_density,
                                const double outer_pressure,
                                const Options::Context& context)
    : initial_radius_(initial_radius),
      inner_mass_density_(inner_mass_density),
      inner_pressure_(inner_pressure),
      outer_mass_density_(outer_mass_density),
      outer_pressure_(outer_pressure),
      equation_of_state_(1.4) {
  ASSERT(initial_radius_ > 0.0,
         "The initial radius must be positive but is " << initial_radius_);
  ASSERT(
      inner_mass_density_ > 0.0,
      "The inner mass density must be positive but is " << inner_mass_density_);
  ASSERT(inner_pressure_ > 0.0,
         "The inner pressure must be positive but is " << inner_pressure_);
  ASSERT(
      outer_mass_density_ > 0.0,
      "The outer mass density must be positive but is " << outer_mass_density_);
  ASSERT(outer_pressure_ > 0.0,
         "The outer pressure must be positive but is " << outer_pressure_);
  if (outer_mass_density_ > inner_mass_density_) {
    PARSE_ERROR(context, "The inner mass density ("
                             << inner_mass_density_
                             << ") must be larger than the outer mass density ("
                             << outer_mass_density_
                             << ") so the shock moves radially outward.");
  }
  if (outer_pressure_ > inner_pressure_) {
    PARSE_ERROR(context, "The inner pressure ("
                             << inner_pressure_
                             << ") must be larger than the outer pressure ("
                             << outer_pressure_
                             << ") so the shock moves radially outward.");
  }
}

template <size_t Dim>
void SodExplosion<Dim>::pup(PUP::er& p) noexcept {
  p | initial_radius_;
  p | inner_mass_density_;
  p | inner_pressure_;
  p | outer_mass_density_;
  p | outer_pressure_;
  p | equation_of_state_;
}

template <size_t Dim>
bool operator==(const SodExplosion<Dim>& lhs,
                const SodExplosion<Dim>& rhs) noexcept {
  // No comparison for equation_of_state_. The adiabatic index is hard-coded.
  return lhs.initial_radius_ == rhs.initial_radius_ and
         lhs.inner_mass_density_ == rhs.inner_mass_density_ and
         lhs.inner_pressure_ == rhs.inner_pressure_ and
         lhs.outer_mass_density_ == rhs.outer_mass_density_ and
         lhs.outer_pressure_ == rhs.outer_pressure_;
}

template <size_t Dim>
bool operator!=(const SodExplosion<Dim>& lhs,
                const SodExplosion<Dim>& rhs) noexcept {
  return not(lhs == rhs);
}

template <size_t Dim>
tuples::TaggedTuple<Tags::MassDensity<DataVector>> SodExplosion<Dim>::variables(
    const tnsr::I<DataVector, Dim, Frame::Inertial>& x,
    tmpl::list<Tags::MassDensity<DataVector>> /*meta*/) const noexcept {
  auto mass_density = make_with_value<Scalar<DataVector>>(x, 0.0);
  std::array<double, Dim> coords{};
  for (size_t s = 0; s < get_size(get<0>(x)); ++s) {
    for (size_t d = 0; d < Dim; ++d) {
      gsl::at(coords, d) = get_element(x.get(d), s);
    }
    const double radius = magnitude(coords);
    get_element(get(mass_density), s) =
        radius > initial_radius_ ? outer_mass_density_ : inner_mass_density_;
  }
  return mass_density;
}

template <size_t Dim>
tuples::TaggedTuple<Tags::Velocity<DataVector, Dim, Frame::Inertial>>
SodExplosion<Dim>::variables(
    const tnsr::I<DataVector, Dim, Frame::Inertial>& x,
    tmpl::list<Tags::Velocity<DataVector, Dim, Frame::Inertial>> /*meta*/)
    const noexcept {
  auto velocity = make_with_value<tnsr::I<DataVector, Dim, Frame::Inertial>>(
      get<0>(x), 0.0);
  return velocity;
}

template <size_t Dim>
tuples::TaggedTuple<Tags::Pressure<DataVector>> SodExplosion<Dim>::variables(
    const tnsr::I<DataVector, Dim, Frame::Inertial>& x,
    tmpl::list<Tags::Pressure<DataVector>> /*meta*/) const noexcept {
  auto pressure = make_with_value<Scalar<DataVector>>(x, 0.0);
  std::array<double, Dim> coords{};
  for (size_t s = 0; s < get_size(get<0>(x)); ++s) {
    for (size_t d = 0; d < Dim; ++d) {
      gsl::at(coords, d) = get_element(x.get(d), s);
    }
    const double radius = magnitude(coords);
    get_element(get(pressure), s) =
        radius > initial_radius_ ? outer_pressure_ : inner_pressure_;
  }
  return pressure;
}

template <size_t Dim>
tuples::TaggedTuple<Tags::SpecificInternalEnergy<DataVector>>
SodExplosion<Dim>::variables(
    const tnsr::I<DataVector, Dim, Frame::Inertial>& x,
    tmpl::list<Tags::SpecificInternalEnergy<DataVector>> /*meta*/)
    const noexcept {
  return equation_of_state_.specific_internal_energy_from_density_and_pressure(
      get<Tags::MassDensity<DataVector>>(
          variables(x, tmpl::list<Tags::MassDensity<DataVector>>{})),
      get<Tags::Pressure<DataVector>>(
          variables(x, tmpl::list<Tags::Pressure<DataVector>>{})));
}

template class SodExplosion<2>;
template class SodExplosion<3>;
template bool operator==(const SodExplosion<2>& lhs,
                         const SodExplosion<2>& rhs) noexcept;
template bool operator==(const SodExplosion<3>& lhs,
                         const SodExplosion<3>& rhs) noexcept;
template bool operator!=(const SodExplosion<2>& lhs,
                         const SodExplosion<2>& rhs) noexcept;
template bool operator!=(const SodExplosion<3>& lhs,
                         const SodExplosion<3>& rhs) noexcept;
}  // namespace NewtonianEuler::AnalyticData
/// \endcond
