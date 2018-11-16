// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "PointwiseFunctions/AnalyticSolutions/GrMhd/AlfvenWave.hpp"

#include <cmath>
#include <cstddef>

#include "DataStructures/DataBox/DataBoxTag.hpp"
#include "DataStructures/DataVector.hpp"                   // IWYU pragma: keep
#include "DataStructures/Tensor/EagerMath/CrossProduct.hpp"
#include "DataStructures/Tensor/EagerMath/DotProduct.hpp"
#include "DataStructures/Tensor/EagerMath/Magnitude.hpp"
#include "DataStructures/Tensor/Tensor.hpp"                // IWYU pragma: keep
#include "ErrorHandling/Assert.hpp"
#include "Parallel/PupStlCpp11.hpp"
#include "Utilities/ConstantExpressions.hpp"
#include "Utilities/EqualWithinRoundoff.hpp"
#include "Utilities/GenerateInstantiations.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/MakeWithValue.hpp"

// IWYU pragma:  no_include "DataStructures/Tensor/TypeAliases.hpp"

/// \cond
namespace grmhd {
namespace Solutions {

AlfvenWave::AlfvenWave(
    const WaveNumber::type wavenumber, const Pressure::type pressure,
    const RestMassDensity::type rest_mass_density,
    const AdiabaticIndex::type adiabatic_index,
    const BackgroundMagneticField::type background_magnetic_field,
    const WaveMagneticField::type wave_magnetic_field) noexcept
    : wavenumber_(wavenumber),
      pressure_(pressure),
      rest_mass_density_(rest_mass_density),
      adiabatic_index_(adiabatic_index),
      background_magnetic_field_(background_magnetic_field),
      wave_magnetic_field_(wave_magnetic_field),
      equation_of_state_{adiabatic_index_},
      initial_unit_vector_along_background_magnetic_field_{
          background_magnetic_field},
      initial_unit_vector_along_wave_magnetic_field_{wave_magnetic_field},
      initial_unit_vector_along_wave_electric_field_{
          cross_product(initial_unit_vector_along_wave_magnetic_field_,
                        initial_unit_vector_along_background_magnetic_field_)} {
  magnitude_B0_ =
      magnitude(initial_unit_vector_along_background_magnetic_field_).get();
  magnitude_B1_ =
      magnitude(initial_unit_vector_along_wave_magnetic_field_).get();
  magnitude_E_ =
      magnitude(initial_unit_vector_along_wave_electric_field_).get();
  for (size_t d = 0; d < 3; d++) {
    initial_unit_vector_along_background_magnetic_field_.get(d) /=
        magnitude_B0_;
    initial_unit_vector_along_wave_magnetic_field_.get(d) /= magnitude_B1_;
    initial_unit_vector_along_wave_electric_field_.get(d) /= magnitude_E_;
  }
  ASSERT(equal_within_roundoff(
             dot_product(initial_unit_vector_along_background_magnetic_field_,
                         initial_unit_vector_along_wave_magnetic_field_)
                 .get(),
             0.0),
         "The background and wave magnetic fields must be perpendicular.");
  const double auxiliary_speed_b0 =
      magnitude_B0_ / sqrt((rest_mass_density_ + pressure_ * adiabatic_index_ /
                                                     (adiabatic_index_ - 1.0)) +
                           square(magnitude_B0_) + square(magnitude_B1_));
  const double auxiliary_speed_b1 =
      magnitude_B1_ * auxiliary_speed_b0 / magnitude_B0_;
  const double one_over_speed_denominator =
      1.0 / sqrt(0.5 * (1.0 + sqrt(1.0 - 4.0 * square(auxiliary_speed_b0 *
                                                      auxiliary_speed_b1))));
  alfven_speed_ = auxiliary_speed_b0 * one_over_speed_denominator;
  fluid_speed_ = -auxiliary_speed_b1 * one_over_speed_denominator;
}

void AlfvenWave::pup(PUP::er& p) noexcept {
  p | wavenumber_;
  p | pressure_;
  p | rest_mass_density_;
  p | adiabatic_index_;
  p | background_magnetic_field_;
  p | wave_magnetic_field_;
  p | alfven_speed_;
  p | fluid_speed_;
  p | initial_unit_vector_along_background_magnetic_field_;
  p | initial_unit_vector_along_wave_magnetic_field_;
  p | initial_unit_vector_along_wave_electric_field_;
  p | magnitude_B0_;
  p | magnitude_B1_;
  p | magnitude_E_;
  p | equation_of_state_;
  p | background_spacetime_;
}

template <typename DataType>
DataType AlfvenWave::k_dot_x_minus_vt(const tnsr::I<DataType, 3>& x,
                                      const double t) const noexcept {
  auto result = make_with_value<DataType>(x, -wavenumber_ * alfven_speed_ * t);
  for (size_t d = 0; d < 3; d++) {
    result += wavenumber_ * x.get(d) *
              initial_unit_vector_along_background_magnetic_field_.get(d);
  }
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
  for (size_t d = 0; d < 3; d++) {
    result.get(d) =
        fluid_speed_ *
        (cos(phase) * initial_unit_vector_along_wave_magnetic_field_[d] -
         sin(phase) * initial_unit_vector_along_wave_electric_field_[d]);
  }
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
      x, 0.0);
  for (size_t d = 0; d < 3; d++) {
    result.get(d) =
        gsl::at(background_magnetic_field_, d) +
        magnitude_B1_ *
            (cos(phase) * initial_unit_vector_along_wave_magnetic_field_[d] -
             sin(phase) * initial_unit_vector_along_wave_electric_field_[d]);
  }
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
  return lhs.wavenumber_ == rhs.wavenumber_ and
         lhs.pressure_ == rhs.pressure_ and
         lhs.rest_mass_density_ == rhs.rest_mass_density_ and
         lhs.adiabatic_index_ == rhs.adiabatic_index_ and
         lhs.background_magnetic_field_ == rhs.background_magnetic_field_ and
         lhs.wave_magnetic_field_ == rhs.wave_magnetic_field_ and
         lhs.initial_unit_vector_along_background_magnetic_field_ ==
             rhs.initial_unit_vector_along_background_magnetic_field_ and
         lhs.initial_unit_vector_along_wave_magnetic_field_ ==
             rhs.initial_unit_vector_along_wave_magnetic_field_ and
         lhs.initial_unit_vector_along_wave_electric_field_ ==
             rhs.initial_unit_vector_along_wave_electric_field_ and
         lhs.magnitude_B0_ == rhs.magnitude_B0_ and
         lhs.magnitude_B1_ == rhs.magnitude_B1_ and
         lhs.magnitude_E_ == rhs.magnitude_E_ and
         lhs.alfven_speed_ == rhs.alfven_speed_ and
         lhs.fluid_speed_ == rhs.fluid_speed_ and
         lhs.background_spacetime_ == rhs.background_spacetime_;
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
