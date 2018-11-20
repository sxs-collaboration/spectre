// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "PointwiseFunctions/AnalyticData/GrMhd/OrszagTangVortex.hpp"

#include <cmath>
#include <pup.h>

#include "DataStructures/DataBox/DataBoxTag.hpp"
#include "DataStructures/DataVector.hpp"  // IWYU pragma: keep
#include "DataStructures/Tensor/EagerMath/DotProduct.hpp"
#include "PointwiseFunctions/Hydro/LorentzFactor.hpp"
#include "Utilities/GenerateInstantiations.hpp"
#include "Utilities/MakeWithValue.hpp"

/// \cond
namespace grmhd {
namespace AnalyticData {

OrszagTangVortex::OrszagTangVortex() = default;

template <typename DataType>
tuples::TaggedTuple<hydro::Tags::RestMassDensity<DataType>>
OrszagTangVortex::variables(
    const tnsr::I<DataType, 3, Frame::Inertial>& x,
    tmpl::list<hydro::Tags::RestMassDensity<DataType>> /*meta*/) const
    noexcept {
  return {
      make_with_value<db::item_type<hydro::Tags::RestMassDensity<DataType>>>(
          x, 25. / (36. * M_PI))};
}

template <typename DataType>
tuples::TaggedTuple<hydro::Tags::SpatialVelocity<DataType, 3>>
OrszagTangVortex::variables(
    const tnsr::I<DataType, 3, Frame::Inertial>& x,
    tmpl::list<hydro::Tags::SpatialVelocity<DataType, 3>> /*meta*/) const
    noexcept {
  return {tnsr::I<DataType, 3>{{{-0.5 * sin(2. * M_PI * get<1>(x)),
                                 0.5 * sin(2. * M_PI * get<0>(x)),
                                 make_with_value<DataType>(x, 0.)}}}};
}

template <typename DataType>
tuples::TaggedTuple<hydro::Tags::SpecificInternalEnergy<DataType>>
OrszagTangVortex::variables(
    const tnsr::I<DataType, 3, Frame::Inertial>& x,
    tmpl::list<hydro::Tags::SpecificInternalEnergy<DataType>> /*meta*/) const
    noexcept {
  using density_tag = hydro::Tags::RestMassDensity<DataType>;
  using pressure_tag = hydro::Tags::Pressure<DataType>;
  const auto data = variables(x, tmpl::list<density_tag, pressure_tag>{});
  return equation_of_state_.specific_internal_energy_from_density_and_pressure(
      get<density_tag>(data), get<pressure_tag>(data));
}

template <typename DataType>
tuples::TaggedTuple<hydro::Tags::Pressure<DataType>>
OrszagTangVortex::variables(
    const tnsr::I<DataType, 3, Frame::Inertial>& x,
    tmpl::list<hydro::Tags::Pressure<DataType>> /*meta*/) const noexcept {
  return {make_with_value<db::item_type<hydro::Tags::Pressure<DataType>>>(
      x, 5. / (12. * M_PI))};
}

template <typename DataType>
tuples::TaggedTuple<hydro::Tags::MagneticField<DataType, 3>>
OrszagTangVortex::variables(
    const tnsr::I<DataType, 3, Frame::Inertial>& x,
    tmpl::list<hydro::Tags::MagneticField<DataType, 3>> /*meta*/) const
    noexcept {
  return {
      tnsr::I<DataType, 3>{{{-1. / sqrt(4. * M_PI) * sin(2. * M_PI * get<1>(x)),
                             1. / sqrt(4. * M_PI) * sin(4. * M_PI * get<0>(x)),
                             make_with_value<DataType>(x, 0.)}}}};
}

template <typename DataType>
tuples::TaggedTuple<hydro::Tags::DivergenceCleaningField<DataType>>
OrszagTangVortex::variables(
    const tnsr::I<DataType, 3, Frame::Inertial>& x,
    tmpl::list<hydro::Tags::DivergenceCleaningField<DataType>> /*meta*/) const
    noexcept {
  return {make_with_value<
      db::item_type<hydro::Tags::DivergenceCleaningField<DataType>>>(x, 0.)};
}

template <typename DataType>
tuples::TaggedTuple<hydro::Tags::LorentzFactor<DataType>>
OrszagTangVortex::variables(
    const tnsr::I<DataType, 3, Frame::Inertial>& x,
    tmpl::list<hydro::Tags::LorentzFactor<DataType>> /*meta*/) const noexcept {
  using velocity_tag = hydro::Tags::SpatialVelocity<DataType, 3>;
  const auto velocity =
      get<velocity_tag>(variables(x, tmpl::list<velocity_tag>{}));
  return {hydro::lorentz_factor(dot_product(velocity, velocity))};
}

template <typename DataType>
tuples::TaggedTuple<hydro::Tags::SpecificEnthalpy<DataType>>
OrszagTangVortex::variables(
    const tnsr::I<DataType, 3, Frame::Inertial>& x,
    tmpl::list<hydro::Tags::SpecificEnthalpy<DataType>> /*meta*/) const
    noexcept {
  using density_tag = hydro::Tags::RestMassDensity<DataType>;
  using energy_tag = hydro::Tags::SpecificInternalEnergy<DataType>;
  const auto data = variables(x, tmpl::list<density_tag, energy_tag>{});
  return equation_of_state_.specific_enthalpy_from_density_and_energy(
      get<density_tag>(data), get<energy_tag>(data));
}

void OrszagTangVortex::pup(PUP::er& p) noexcept { p | equation_of_state_; }

bool operator==(const OrszagTangVortex& /*lhs*/,
                const OrszagTangVortex& /*rhs*/) noexcept {
  // there is no comparison operator for the EoS, but it's the same
  // for every instance of this class.
  return true;
}

bool operator!=(const OrszagTangVortex& lhs,
                const OrszagTangVortex& rhs) noexcept {
  return not(lhs == rhs);
}

#define DTYPE(data) BOOST_PP_TUPLE_ELEM(0, data)
#define TAG(data) BOOST_PP_TUPLE_ELEM(1, data)

#define INSTANTIATE_SCALARS(_, data)                         \
  template tuples::TaggedTuple<TAG(data) < DTYPE(data)>>     \
      OrszagTangVortex::variables(                           \
          const tnsr::I<DTYPE(data), 3, Frame::Inertial>& x, \
          tmpl::list<TAG(data) < DTYPE(data)>> /*meta*/) const noexcept;

GENERATE_INSTANTIATIONS(
    INSTANTIATE_SCALARS, (double, DataVector),
    (hydro::Tags::RestMassDensity, hydro::Tags::SpecificInternalEnergy,
     hydro::Tags::Pressure, hydro::Tags::DivergenceCleaningField,
     hydro::Tags::LorentzFactor, hydro::Tags::SpecificEnthalpy))

#define INSTANTIATE_VECTORS(_, data)                                         \
  template tuples::TaggedTuple<TAG(data) < DTYPE(data), 3>>                  \
      OrszagTangVortex::variables(                                           \
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
