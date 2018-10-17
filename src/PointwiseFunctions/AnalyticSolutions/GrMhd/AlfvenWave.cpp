// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "PointwiseFunctions/AnalyticSolutions/GrMhd/AlfvenWave.hpp"

#include <cmath>
#include <pup.h>

#include "DataStructures/DataBox/DataBoxTag.hpp"
#include "DataStructures/DataVector.hpp"                   // IWYU pragma: keep
#include "DataStructures/Tensor/Tensor.hpp"                // IWYU pragma: keep
#include "Utilities/ConstantExpressions.hpp"
#include "Utilities/GenerateInstantiations.hpp"
#include "Utilities/MakeWithValue.hpp"

// IWYU pragma:  no_include "DataStructures/Tensor/TypeAliases.hpp"

/// \cond
namespace grmhd {
namespace Solutions {

AlfvenWave::AlfvenWave(const WaveNumber::type wavenumber,
                       const Pressure::type pressure,
                       const RestMassDensity::type rest_mass_density,
                       const AdiabaticIndex::type adiabatic_index,
                       const BackgroundMagField::type background_mag_field,
                       const PerturbationSize::type perturbation_size) noexcept
    : wavenumber_(wavenumber),
      pressure_(pressure),
      rest_mass_density_(rest_mass_density),
      adiabatic_index_(adiabatic_index),
      background_mag_field_(background_mag_field),
      perturbation_size_(perturbation_size),
      equation_of_state_{adiabatic_index_} {
  alfven_speed_ = background_mag_field /
                  sqrt((rest_mass_density_ + pressure_ * adiabatic_index_ /
                                                 (adiabatic_index_ - 1.0)) +
                       square(background_mag_field));
  fluid_speed_ = -perturbation_size * alfven_speed_ / background_mag_field;
}

void AlfvenWave::pup(PUP::er& p) noexcept {
  p | wavenumber_;
  p | pressure_;
  p | rest_mass_density_;
  p | adiabatic_index_;
  p | background_mag_field_;
  p | perturbation_size_;
  p | alfven_speed_;
  p | fluid_speed_;
  p | equation_of_state_;
  p | background_spacetime_;
}

template <typename DataType>
DataType AlfvenWave::k_dot_x_minus_vt(const tnsr::I<DataType, 3>& x,
                                      const double t) const noexcept {
  auto result = make_with_value<DataType>(x, -wavenumber_ * alfven_speed_ * t);
  result += wavenumber_ * x.get(2);
  return result;
}

template <typename DataType>
tuples::TaggedTuple<hydro::Tags::RestMassDensity<DataType>>
AlfvenWave::variables(
    const tnsr::I<DataType, 3>& x, double /*t*/,
    tmpl::list<hydro::Tags::RestMassDensity<DataType>> /*meta*/) const
    noexcept {
  return {
      make_with_value<db::item_type<hydro::Tags::RestMassDensity<DataType>>>(
          x, rest_mass_density_)};
}

template <typename DataType>
tuples::TaggedTuple<hydro::Tags::SpecificInternalEnergy<DataType>>
AlfvenWave::variables(
    const tnsr::I<DataType, 3>& x, double /*t*/,
    tmpl::list<hydro::Tags::SpecificInternalEnergy<DataType>> /*meta*/) const
    noexcept {
  return {make_with_value<Scalar<DataType>>(
      x, pressure_ / ((adiabatic_index_ - 1.0) * rest_mass_density_))};
}

template <typename DataType>
tuples::TaggedTuple<hydro::Tags::Pressure<DataType>> AlfvenWave::variables(
    const tnsr::I<DataType, 3>& x, double /*t*/,
    tmpl::list<hydro::Tags::Pressure<DataType>> /*meta*/) const noexcept {
  return {make_with_value<Scalar<DataType>>(x, pressure_)};
}

template <typename DataType>
tuples::TaggedTuple<hydro::Tags::SpatialVelocity<DataType, 3, Frame::Inertial>>
AlfvenWave::variables(const tnsr::I<DataType, 3>& x, double t,
                      tmpl::list<hydro::Tags::SpatialVelocity<
                          DataType, 3, Frame::Inertial>> /*meta*/) const
    noexcept {
  const DataType phase = k_dot_x_minus_vt(x, t);
  auto result = make_with_value<db::item_type<
      hydro::Tags::SpatialVelocity<DataType, 3, Frame::Inertial>>>(x, 0.0);
  get<0>(result) = fluid_speed_ * cos(phase);
  get<1>(result) = fluid_speed_ * sin(phase);
  return {std::move(result)};
}

template <typename DataType>
tuples::TaggedTuple<hydro::Tags::MagneticField<DataType, 3, Frame::Inertial>>
AlfvenWave::variables(const tnsr::I<DataType, 3>& x, double t,
                      tmpl::list<hydro::Tags::MagneticField<
                          DataType, 3, Frame::Inertial>> /*meta*/) const
    noexcept {
  const DataType phase = k_dot_x_minus_vt(x, t);
  auto result = make_with_value<
      db::item_type<hydro::Tags::MagneticField<DataType, 3, Frame::Inertial>>>(
      x, background_mag_field_);
  get<0>(result) = perturbation_size_ * cos(phase);
  get<1>(result) = perturbation_size_ * sin(phase);
  return {std::move(result)};
}

template <typename DataType>
tuples::TaggedTuple<hydro::Tags::DivergenceCleaningField<DataType>>
AlfvenWave::variables(
    const tnsr::I<DataType, 3>& x, double /*t*/,
    tmpl::list<hydro::Tags::DivergenceCleaningField<DataType>> /*meta*/) const
    noexcept {
  return {make_with_value<
      db::item_type<hydro::Tags::DivergenceCleaningField<DataType>>>(x, 0.0)};
}

template <typename DataType>
tuples::TaggedTuple<hydro::Tags::LorentzFactor<DataType>> AlfvenWave::variables(
    const tnsr::I<DataType, 3>& x, double /*t*/,
    tmpl::list<hydro::Tags::LorentzFactor<DataType>> /*meta*/) const noexcept {
  return {make_with_value<db::item_type<hydro::Tags::LorentzFactor<DataType>>>(
      x, 1.0 / sqrt(1.0 - square(fluid_speed_)))};
}

template <typename DataType>
tuples::TaggedTuple<hydro::Tags::SpecificEnthalpy<DataType>>
AlfvenWave::variables(
    const tnsr::I<DataType, 3>& x, double t,
    tmpl::list<hydro::Tags::SpecificEnthalpy<DataType>> /*meta*/) const
    noexcept {
  Scalar<DataType> specific_internal_energy = std::move(
      get<hydro::Tags::SpecificInternalEnergy<DataType>>(variables<DataType>(
          x, t, tmpl::list<hydro::Tags::SpecificInternalEnergy<DataType>>{})));
  get(specific_internal_energy) *= adiabatic_index_;
  get(specific_internal_energy) += 1.0;
  return {std::move(specific_internal_energy)};
}

bool operator==(const AlfvenWave& lhs, const AlfvenWave& rhs) noexcept {
  // there is no comparison operator for the EoS, but should be okay as
  // the adiabatic_indexs are compared
  return lhs.wavenumber() == rhs.wavenumber() and
         lhs.pressure() == rhs.pressure() and
         lhs.rest_mass_density() == rhs.rest_mass_density() and
         lhs.adiabatic_index() == rhs.adiabatic_index() and
         lhs.background_mag_field() == rhs.background_mag_field() and
         lhs.perturbation_size() == rhs.perturbation_size() and
         lhs.alfven_speed() == rhs.alfven_speed() and
         lhs.fluid_speed() == rhs.fluid_speed() and
         lhs.background_spacetime() == rhs.background_spacetime();
}

bool operator!=(const AlfvenWave& lhs, const AlfvenWave& rhs) noexcept {
  return not(lhs == rhs);
}

#define DTYPE(data) BOOST_PP_TUPLE_ELEM(0, data)
#define TAG(data) BOOST_PP_TUPLE_ELEM(1, data)

#define INSTANTIATE_SCALARS(_, data)                                       \
  template tuples::TaggedTuple<TAG(data) < DTYPE(data)>>                   \
      AlfvenWave::variables(const tnsr::I<DTYPE(data), 3>& x, double t,    \
                            tmpl::list<TAG(data) < DTYPE(data)>> /*meta*/) \
          const noexcept;

GENERATE_INSTANTIATIONS(
    INSTANTIATE_SCALARS, (double, DataVector),
    (hydro::Tags::RestMassDensity, hydro::Tags::SpecificInternalEnergy,
     hydro::Tags::Pressure, hydro::Tags::DivergenceCleaningField,
     hydro::Tags::LorentzFactor, hydro::Tags::SpecificEnthalpy))

#define INSTANTIATE_VECTORS(_, data)                                         \
  template tuples::TaggedTuple<TAG(data) < DTYPE(data), 3, Frame::Inertial>> \
      AlfvenWave::variables(                                                 \
          const tnsr::I<DTYPE(data), 3>& x, double t,                        \
          tmpl::list<TAG(data) < DTYPE(data), 3, Frame::Inertial>> /*meta*/) \
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
