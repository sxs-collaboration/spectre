// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "PointwiseFunctions/AnalyticData/GrMhd/BondiHoyleAccretion.hpp"

#include <cmath>
#include <pup.h>

#include "DataStructures/DataBox/DataBoxTag.hpp"
#include "DataStructures/DataVector.hpp"  // IWYU pragma: keep
#include "DataStructures/Tensor/Tensor.hpp"  // IWYU pragma: keep
#include "PointwiseFunctions/Hydro/EquationsOfState/EquationOfState.hpp"
#include "PointwiseFunctions/Hydro/EquationsOfState/PolytropicFluid.hpp"  // IWYU pragma: keep
#include "Utilities/ConstantExpressions.hpp"
#include "Utilities/GenerateInstantiations.hpp"
#include "Utilities/MakeWithValue.hpp"

// IWYU pragma:  no_include "DataStructures/Tensor/TypeAliases.hpp"

/// \cond
namespace grmhd {
namespace AnalyticData {

BondiHoyleAccretion::BondiHoyleAccretion(
    const double bh_mass, const double bh_dimless_spin,
    const double rest_mass_density, const double flow_speed,
    const double magnetic_field_strength, const double polytropic_constant,
    const double polytropic_exponent) noexcept
    : bh_mass_(bh_mass),
      bh_spin_a_(bh_mass * bh_dimless_spin),
      rest_mass_density_(rest_mass_density),
      flow_speed_(flow_speed),
      magnetic_field_strength_(magnetic_field_strength),
      polytropic_constant_(polytropic_constant),
      polytropic_exponent_(polytropic_exponent),
      equation_of_state_{polytropic_constant_, polytropic_exponent_},
      background_spacetime_{
          bh_mass, {{0.0, 0.0, bh_dimless_spin}}, {{0.0, 0.0, 0.0}}},
      kerr_schild_coords_{bh_mass, bh_dimless_spin} {}

void BondiHoyleAccretion::pup(PUP::er& p) noexcept {
  p | bh_mass_;
  p | bh_spin_a_;
  p | rest_mass_density_;
  p | flow_speed_;
  p | magnetic_field_strength_;
  p | polytropic_constant_;
  p | polytropic_exponent_;
  p | equation_of_state_;
  p | background_spacetime_;
  p | kerr_schild_coords_;
}

template <typename DataType>
typename hydro::Tags::SpatialVelocity<DataType, 3, Frame::NoFrame>::type
BondiHoyleAccretion::spatial_velocity(const DataType& r_squared,
                                      const DataType& cos_theta,
                                      const DataType& sin_theta) const
    noexcept {
  auto result = make_with_value<
      typename hydro::Tags::SpatialVelocity<DataType, 3, Frame::NoFrame>::type>(
      r_squared, 0.0);

  const DataType sigma = r_squared + square(bh_spin_a_) * square(cos_theta);
  get<0>(result) = flow_speed_ * cos_theta /
                   sqrt(1.0 + 2.0 * bh_mass_ * sqrt(r_squared) / sigma);
  get<1>(result) = -flow_speed_ * sin_theta / sqrt(sigma);
  // get<2>(result) is identically zero

  return result;
}

template <typename DataType>
typename hydro::Tags::MagneticField<DataType, 3, Frame::NoFrame>::type
BondiHoyleAccretion::magnetic_field(const DataType& r_squared,
                                    const DataType& cos_theta,
                                    const DataType& sin_theta) const noexcept {
  const double a_squared = bh_spin_a_ * bh_spin_a_;
  const DataType cos_theta_squared = square(cos_theta);
  const DataType two_m_r = 2.0 * bh_mass_ * sqrt(r_squared);

  // The square root of the determinant of the spatial metric is proportional to
  // sin(theta) squared. That factor cancels out after multiplying by Faraday.
  DataType sigma = r_squared + a_squared * cos_theta_squared;
  const DataType prefactor =
      magnetic_field_strength_ / sqrt(sigma * (sigma + two_m_r));

  auto result = make_with_value<
      typename hydro::Tags::MagneticField<DataType, 3, Frame::NoFrame>::type>(
      r_squared, 0.0);

  // Redefine sigma to save an allocation.
  sigma = 1.0 / square(sigma);
  get<0>(result) = prefactor *
                   (r_squared - two_m_r + a_squared +
                    sigma * two_m_r * (square(r_squared) - square(a_squared))) *
                   cos_theta;

  get<1>(result) = -prefactor *
                   (sqrt(r_squared) +
                    bh_mass_ * a_squared * sigma *
                        (r_squared - a_squared * cos_theta_squared) *
                        (1.0 + cos_theta_squared)) *
                   sin_theta;

  get<2>(result) = bh_spin_a_ * prefactor *
                   (1.0 + sigma * two_m_r * (r_squared - a_squared)) *
                   cos_theta;

  return result;
}

template <typename DataType>
tuples::TaggedTuple<hydro::Tags::RestMassDensity<DataType>>
BondiHoyleAccretion::variables(
    const tnsr::I<DataType, 3, Frame::Inertial>& x,
    tmpl::list<hydro::Tags::RestMassDensity<DataType>> /*meta*/) const
    noexcept {
  return {
      make_with_value<db::item_type<hydro::Tags::RestMassDensity<DataType>>>(
          x, rest_mass_density_)};
}

template <typename DataType>
tuples::TaggedTuple<hydro::Tags::SpatialVelocity<DataType, 3>>
BondiHoyleAccretion::variables(
    const tnsr::I<DataType, 3, Frame::Inertial>& x,
    tmpl::list<hydro::Tags::SpatialVelocity<DataType, 3>> /*meta*/) const
    noexcept {
  const DataType r_squared = get(kerr_schild_coords_.r_coord_squared(x));
  return kerr_schild_coords_.cartesian_from_spherical_ks(
      spatial_velocity(r_squared, DataType{get<2>(x) / sqrt(r_squared)},
                       DataType{sqrt((square(get<0>(x)) + square(get<1>(x))) /
                                     (r_squared + square(bh_spin_a_)))}),
      x);
}

template <typename DataType>
tuples::TaggedTuple<hydro::Tags::SpecificInternalEnergy<DataType>>
BondiHoyleAccretion::variables(
    const tnsr::I<DataType, 3, Frame::Inertial>& x,
    tmpl::list<hydro::Tags::SpecificInternalEnergy<DataType>> /*meta*/) const
    noexcept {
  return equation_of_state_.specific_internal_energy_from_density(
      get<hydro::Tags::RestMassDensity<DataType>>(
          variables(x, tmpl::list<hydro::Tags::RestMassDensity<DataType>>{})));
}

template <typename DataType>
tuples::TaggedTuple<hydro::Tags::Pressure<DataType>>
BondiHoyleAccretion::variables(
    const tnsr::I<DataType, 3, Frame::Inertial>& x,
    tmpl::list<hydro::Tags::Pressure<DataType>> /*meta*/) const noexcept {
  return equation_of_state_.pressure_from_density(
      get<hydro::Tags::RestMassDensity<DataType>>(
          variables(x, tmpl::list<hydro::Tags::RestMassDensity<DataType>>{})));
}

template <typename DataType>
tuples::TaggedTuple<hydro::Tags::MagneticField<DataType, 3>>
BondiHoyleAccretion::variables(
    const tnsr::I<DataType, 3, Frame::Inertial>& x,
    tmpl::list<hydro::Tags::MagneticField<DataType, 3>> /*meta*/) const
    noexcept {
  const DataType r_squared = get(kerr_schild_coords_.r_coord_squared(x));
  return kerr_schild_coords_.cartesian_from_spherical_ks(
      magnetic_field(r_squared, DataType{get<2>(x) / sqrt(r_squared)},
                     DataType{sqrt((square(get<0>(x)) + square(get<1>(x))) /
                                   (r_squared + square(bh_spin_a_)))}),
      x);
}

template <typename DataType>
tuples::TaggedTuple<hydro::Tags::DivergenceCleaningField<DataType>>
BondiHoyleAccretion::variables(
    const tnsr::I<DataType, 3, Frame::Inertial>& x,
    tmpl::list<hydro::Tags::DivergenceCleaningField<DataType>> /*meta*/) const
    noexcept {
  return {make_with_value<
      db::item_type<hydro::Tags::DivergenceCleaningField<DataType>>>(x, 0.0)};
}

template <typename DataType>
tuples::TaggedTuple<hydro::Tags::LorentzFactor<DataType>>
BondiHoyleAccretion::variables(
    const tnsr::I<DataType, 3, Frame::Inertial>& x,
    tmpl::list<hydro::Tags::LorentzFactor<DataType>> /*meta*/) const noexcept {
  return {make_with_value<db::item_type<hydro::Tags::LorentzFactor<DataType>>>(
      x, 1.0 / sqrt(1.0 - square(flow_speed_)))};
}

template <typename DataType>
tuples::TaggedTuple<hydro::Tags::SpecificEnthalpy<DataType>>
BondiHoyleAccretion::variables(
    const tnsr::I<DataType, 3, Frame::Inertial>& x,
    tmpl::list<hydro::Tags::SpecificEnthalpy<DataType>> /*meta*/) const
    noexcept {
  return equation_of_state_.specific_enthalpy_from_density(
      get<hydro::Tags::RestMassDensity<DataType>>(
          variables(x, tmpl::list<hydro::Tags::RestMassDensity<DataType>>{})));
}

bool operator==(const BondiHoyleAccretion& lhs,
                const BondiHoyleAccretion& rhs) noexcept {
  // There is no comparison operator for the `equation_of_state` and the
  // `background_spacetime`, but should be okay as the `bh_mass`s,
  // `bh_dimless_spin`s, `polytropic_exponent`s and `polytropic_constant`s are
  // compared.
  return lhs.bh_mass_ == rhs.bh_mass_ and lhs.bh_spin_a_ == rhs.bh_spin_a_ and
         lhs.rest_mass_density_ == rhs.rest_mass_density_ and
         lhs.flow_speed_ == rhs.flow_speed_ and
         lhs.magnetic_field_strength_ == rhs.magnetic_field_strength_ and
         lhs.polytropic_constant_ == rhs.polytropic_constant_ and
         lhs.polytropic_exponent_ == rhs.polytropic_exponent_;
}

bool operator!=(const BondiHoyleAccretion& lhs,
                const BondiHoyleAccretion& rhs) noexcept {
  return not(lhs == rhs);
}

#define DTYPE(data) BOOST_PP_TUPLE_ELEM(0, data)
#define TAG(data) BOOST_PP_TUPLE_ELEM(1, data)

#define INSTANTIATE_SCALARS(_, data)                         \
  template tuples::TaggedTuple<TAG(data) < DTYPE(data)>>     \
      BondiHoyleAccretion::variables(                        \
          const tnsr::I<DTYPE(data), 3, Frame::Inertial>& x, \
          tmpl::list<TAG(data) < DTYPE(data)>> /*meta*/) const noexcept;

GENERATE_INSTANTIATIONS(
    INSTANTIATE_SCALARS, (double, DataVector),
    (hydro::Tags::RestMassDensity, hydro::Tags::SpecificInternalEnergy,
     hydro::Tags::Pressure, hydro::Tags::DivergenceCleaningField,
     hydro::Tags::LorentzFactor, hydro::Tags::SpecificEnthalpy))

#define INSTANTIATE_VECTORS(_, data)                                         \
  template tuples::TaggedTuple<TAG(data) < DTYPE(data), 3>>                  \
      BondiHoyleAccretion::variables(                                        \
          const tnsr::I<DTYPE(data), 3, Frame::Inertial>& x,                 \
          tmpl::list<TAG(data) < DTYPE(data), 3, Frame::Inertial>> /*meta*/) \
          const noexcept;

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
