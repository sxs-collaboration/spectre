// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "PointwiseFunctions/AnalyticSolutions/GrMhd/AlfvenWave.hpp"

#include <cmath>
#include <pup.h>

#include "DataStructures/DataBox/Prefixes.hpp"             // IWYU pragma: keep
#include "DataStructures/DataVector.hpp"                   // IWYU pragma: keep
#include "DataStructures/Tensor/EagerMath/DotProduct.hpp"  // IWYU pragma: keep
#include "DataStructures/Tensor/Tensor.hpp"
#include "Utilities/ConstantExpressions.hpp"
#include "Utilities/GenerateInstantiations.hpp"
#include "Utilities/MakeWithValue.hpp"
/// \cond
namespace grmhd {
namespace Solutions {

AlfvenWave::AlfvenWave(const WaveNumber::type wavenumber,
                       const Pressure::type pressure,
                       const RestMassDensity::type rest_mass_density,
                       const AdiabaticExponent::type adiabatic_exponent,
                       const BackgroundMagField::type background_mag_field,
                       const PerturbationSize::type perturbation_size) noexcept
    : wavenumber_(wavenumber),
      pressure_(pressure),
      rest_mass_density_(rest_mass_density),
      adiabatic_exponent_(adiabatic_exponent),
      background_mag_field_(background_mag_field),
      perturbation_size_(perturbation_size) {
  alfven_speed_ = background_mag_field /
                  sqrt((rest_mass_density_ + pressure_ * adiabatic_exponent_ /
                                                 (adiabatic_exponent_ - 1.0)) +
                       square(background_mag_field));
  fluid_speed_ = -perturbation_size * alfven_speed_ / background_mag_field;
}

void AlfvenWave::pup(PUP::er& p) noexcept {
  p | wavenumber_;
  p | pressure_;
  p | rest_mass_density_;
  p | adiabatic_exponent_;
  p | background_mag_field_;
  p | perturbation_size_;
  p | alfven_speed_;
  p | fluid_speed_;
}

template <typename DataType>
DataType AlfvenWave::k_dot_x_minus_vt(const tnsr::I<DataType, 3>& x,
                                      const double t) const noexcept {
  auto result = make_with_value<DataType>(x, -wavenumber_ * alfven_speed_ * t);
  result += wavenumber_ * x.get(2);
  return result;
}

template <typename DataType>
tuples::tagged_tuple_from_typelist<AlfvenWave::variables_t<DataType>>
AlfvenWave::variables(const tnsr::I<DataType, 3>& x, const double t,
                      AlfvenWave::variables_t<DataType> /*meta*/) const
    noexcept {
  // Explicitly set all variables to zero:
  auto result = make_with_value<
      tuples::tagged_tuple_from_typelist<AlfvenWave::variables_t<DataType>>>(
      x, 0.0);

  const DataType phase = k_dot_x_minus_vt(x, t);
  get(get<hydro::Tags::RestMassDensity<DataType>>(result)) = rest_mass_density_;
  get(get<hydro::Tags::SpecificInternalEnergy<DataType>>(result)) =
      pressure_ / ((adiabatic_exponent_ - 1.0) * rest_mass_density_);
  get<hydro::Tags::SpatialVelocity<DataType, 3, Frame::Inertial>>(result).get(
      0) = fluid_speed_ * cos(phase);
  get<hydro::Tags::SpatialVelocity<DataType, 3, Frame::Inertial>>(result).get(
      1) = fluid_speed_ * sin(phase);
  get<hydro::Tags::SpatialVelocity<DataType, 3, Frame::Inertial>>(result).get(
      2) = 0.0;

  get<hydro::Tags::MagneticField<DataType, 3, Frame::Inertial>>(result).get(0) =
      perturbation_size_ * cos(phase);
  get<hydro::Tags::MagneticField<DataType, 3, Frame::Inertial>>(result).get(1) =
      perturbation_size_ * sin(phase);
  get<hydro::Tags::MagneticField<DataType, 3, Frame::Inertial>>(result).get(2) =
      background_mag_field_;

  get(get<hydro::Tags::Pressure<DataType>>(result)) = pressure_;
  return result;
}

template <typename DataType>
tuples::tagged_tuple_from_typelist<AlfvenWave::dt_variables_t<DataType>>
AlfvenWave::variables(const tnsr::I<DataType, 3>& x, const double t,
                      AlfvenWave::dt_variables_t<DataType> /*meta*/) const
    noexcept {
  // Explicitly set all variables to zero:
  auto result = make_with_value<
      tuples::tagged_tuple_from_typelist<AlfvenWave::dt_variables_t<DataType>>>(
      x, 0.0);

  const DataType phase = k_dot_x_minus_vt(x, t);

  // Angular frequency:
  const double omega = alfven_speed_ * wavenumber_;

  get<Tags::dt<hydro::Tags::SpatialVelocity<DataType, 3, Frame::Inertial>>>(
      result)
      .get(0) = fluid_speed_ * omega * sin(phase);
  get<Tags::dt<hydro::Tags::SpatialVelocity<DataType, 3, Frame::Inertial>>>(
      result)
      .get(1) = -fluid_speed_ * omega * cos(phase);
  get<Tags::dt<hydro::Tags::SpatialVelocity<DataType, 3, Frame::Inertial>>>(
      result)
      .get(2) = 0.0;

  get<Tags::dt<hydro::Tags::MagneticField<DataType, 3, Frame::Inertial>>>(
      result)
      .get(0) = perturbation_size_ * omega * sin(phase);
  get<Tags::dt<hydro::Tags::MagneticField<DataType, 3, Frame::Inertial>>>(
      result)
      .get(1) = -perturbation_size_ * omega * cos(phase);
  get<Tags::dt<hydro::Tags::MagneticField<DataType, 3, Frame::Inertial>>>(
      result)
      .get(2) = 0.0;

  // The time derivatives of the rest mass density, pressure, and specific
  // internal energy are not set because they are identically zero, and
  // `result` is initialized with all time derivatives of primitive variables
  // equal to zero.

  return result;
}

bool operator==(const AlfvenWave& lhs, const AlfvenWave& rhs) noexcept {
  return lhs.wavenumber() == rhs.wavenumber() and
         lhs.pressure() == rhs.pressure() and
         lhs.rest_mass_density() == rhs.rest_mass_density() and
         lhs.adiabatic_exponent() == rhs.adiabatic_exponent() and
         lhs.background_mag_field() == rhs.background_mag_field() and
         lhs.perturbation_size() == rhs.perturbation_size() and
         lhs.alfven_speed() == rhs.alfven_speed() and
         lhs.fluid_speed() == rhs.fluid_speed();
}

bool operator!=(const AlfvenWave& lhs, const AlfvenWave& rhs) noexcept {
  return not(lhs == rhs);
}

}  // namespace Solutions
}  // namespace grmhd

#define DTYPE(data) BOOST_PP_TUPLE_ELEM(0, data)

#define INSTANTIATE(_, data)                                              \
  template tuples::tagged_tuple_from_typelist<                            \
      grmhd::Solutions::AlfvenWave::variables_t<DTYPE(data)>>             \
  grmhd::Solutions::AlfvenWave::variables(                                \
      const tnsr::I<DTYPE(data), 3>& x, const double t,                   \
      grmhd::Solutions::AlfvenWave::variables_t<DTYPE(data)> /*meta*/)    \
      const noexcept;                                                     \
  template tuples::tagged_tuple_from_typelist<                            \
      grmhd::Solutions::AlfvenWave::dt_variables_t<DTYPE(data)>>          \
  grmhd::Solutions::AlfvenWave::variables(                                \
      const tnsr::I<DTYPE(data), 3>& x, const double t,                   \
      grmhd::Solutions::AlfvenWave::dt_variables_t<DTYPE(data)> /*meta*/) \
      const noexcept;

GENERATE_INSTANTIATIONS(INSTANTIATE, (double, DataVector))

#undef DTYPE
#undef INSTANTIATE
/// \endcond
