// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "PointwiseFunctions/AnalyticData/GrMhd/KhInstability.hpp"

#include <cmath>
#include <cstddef>
#include <pup.h>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/EagerMath/DotProduct.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "PointwiseFunctions/Hydro/EquationsOfState/EquationOfState.hpp"
#include "PointwiseFunctions/Hydro/EquationsOfState/IdealFluid.hpp"
#include "PointwiseFunctions/Hydro/LorentzFactor.hpp"
#include "PointwiseFunctions/Hydro/SpecificEnthalpy.hpp"
#include "PointwiseFunctions/Hydro/Tags.hpp"
#include "Utilities/ConstantExpressions.hpp"
#include "Utilities/ContainerHelpers.hpp"
#include "Utilities/ErrorHandling/Assert.hpp"
#include "Utilities/GenerateInstantiations.hpp"
#include "Utilities/MakeWithValue.hpp"

namespace grmhd::AnalyticData {
KhInstability::KhInstability(
    const double adiabatic_index, const double strip_bimedian_height,
    const double strip_thickness, const double strip_density,
    const double strip_velocity, const double background_density,
    const double background_velocity, const double pressure,
    const double perturbation_amplitude, const double perturbation_width,
    const std::array<double, 3>& magnetic_field)
    : adiabatic_index_(adiabatic_index),
      strip_bimedian_height_(strip_bimedian_height),
      strip_half_thickness_(0.5 * strip_thickness),
      strip_density_(strip_density),
      strip_velocity_(strip_velocity),
      background_density_(background_density),
      background_velocity_(background_velocity),
      pressure_(pressure),
      perturbation_amplitude_(perturbation_amplitude),
      perturbation_width_(perturbation_width),
      magnetic_field_(magnetic_field),
      equation_of_state_(adiabatic_index) {
  ASSERT(strip_density_ > 0.0 and background_density_ > 0.0,
         "The mass density must be positive everywhere. Inner "
         "density: "
             << strip_density_ << ", Outer density: " << background_density_
             << ".");
  ASSERT(pressure_ > 0.0, "The pressure must be positive. The value given was "
                              << pressure_ << ".");
  ASSERT(perturbation_width_ > 0.0,
         "The damping factor must be positive. The value given was "
             << perturbation_width_ << ".");
  ASSERT(strip_thickness > 0.0,
         "The strip thickness must be positive. The value given was "
             << strip_thickness << ".");
}

KhInstability::KhInstability(CkMigrateMessage* msg) : InitialData(msg) {}

void KhInstability::pup(PUP::er& p) {
  InitialData::pup(p);
  p | adiabatic_index_;
  p | strip_bimedian_height_;
  p | strip_half_thickness_;
  p | strip_density_;
  p | strip_velocity_;
  p | background_density_;
  p | background_velocity_;
  p | pressure_;
  p | perturbation_amplitude_;
  p | perturbation_width_;
  p | magnetic_field_;
  p | equation_of_state_;
  p | background_spacetime_;
}

template <typename DataType>
tuples::TaggedTuple<hydro::Tags::RestMassDensity<DataType>>
KhInstability::variables(
    const tnsr::I<DataType, 3>& x,
    tmpl::list<hydro::Tags::RestMassDensity<DataType>> /*meta*/) const {
  auto result = make_with_value<Scalar<DataType>>(x, 0.0);
  const size_t n_pts = get_size(get<0>(x));
  for (size_t s = 0; s < n_pts; ++s) {
    get_element(get(result), s) =
        abs(get_element(get<1>(x), s) - strip_bimedian_height_) <
                strip_half_thickness_
            ? strip_density_
            : background_density_;
  }
  return {std::move(result)};
}

template <typename DataType>
tuples::TaggedTuple<hydro::Tags::SpecificInternalEnergy<DataType>>
KhInstability::variables(
    const tnsr::I<DataType, 3, Frame::Inertial>& x,
    tmpl::list<hydro::Tags::SpecificInternalEnergy<DataType>> /*meta*/) const {
  return equation_of_state_.specific_internal_energy_from_density_and_pressure(
      get<hydro::Tags::RestMassDensity<DataType>>(
          variables(x, tmpl::list<hydro::Tags::RestMassDensity<DataType>>{})),
      get<hydro::Tags::Pressure<DataType>>(
          variables(x, tmpl::list<hydro::Tags::Pressure<DataType>>{})));
}

template <typename DataType>
tuples::TaggedTuple<hydro::Tags::Pressure<DataType>> KhInstability::variables(
    const tnsr::I<DataType, 3, Frame::Inertial>& x,
    tmpl::list<hydro::Tags::Pressure<DataType>> /*meta*/) const {
  return make_with_value<Scalar<DataType>>(x, pressure_);
}

template <typename DataType>
tuples::TaggedTuple<hydro::Tags::SpatialVelocity<DataType, 3>>
KhInstability::variables(
    const tnsr::I<DataType, 3>& x,
    tmpl::list<hydro::Tags::SpatialVelocity<DataType, 3>> /*meta*/) const {
  auto result = make_with_value<tnsr::I<DataType, 3, Frame::Inertial>>(x, 0.0);

  const size_t n_pts = get_size(get<0>(x));
  for (size_t s = 0; s < n_pts; ++s) {
    get_element(get<0>(result), s) =
        abs(get_element(get<1>(x), s) - strip_bimedian_height_) <
                strip_half_thickness_
            ? strip_velocity_
            : background_velocity_;
  }

  const double one_over_two_sigma_squared = 0.5 / square(perturbation_width_);
  const double strip_lower_bound =
      strip_bimedian_height_ - strip_half_thickness_;
  const double strip_upper_bound =
      strip_bimedian_height_ + strip_half_thickness_;
  get<1>(result) =
      exp(-one_over_two_sigma_squared * square(get<1>(x) - strip_lower_bound)) +
      exp(-one_over_two_sigma_squared * square(get<1>(x) - strip_upper_bound));
  get<1>(result) *= perturbation_amplitude_ * sin(4.0 * M_PI * get<0>(x));

  return {std::move(result)};
}

template <typename DataType>
tuples::TaggedTuple<hydro::Tags::MagneticField<DataType, 3>>
KhInstability::variables(
    const tnsr::I<DataType, 3>& x,
    tmpl::list<hydro::Tags::MagneticField<DataType, 3>> /*meta*/) const {
  auto mag_field = make_with_value<tnsr::I<DataType, 3, Frame::Inertial>>(
      get<0>(x), magnetic_field_[0]);
  get<1>(mag_field) = magnetic_field_[1];
  get<2>(mag_field) = magnetic_field_[2];
  return mag_field;
}

template <typename DataType>
tuples::TaggedTuple<hydro::Tags::DivergenceCleaningField<DataType>>
KhInstability::variables(
    const tnsr::I<DataType, 3>& x,
    tmpl::list<hydro::Tags::DivergenceCleaningField<DataType>> /*meta*/) const {
  return make_with_value<Scalar<DataType>>(x, 0.0);
}

template <typename DataType>
tuples::TaggedTuple<hydro::Tags::LorentzFactor<DataType>>
KhInstability::variables(
    const tnsr::I<DataType, 3>& x,
    tmpl::list<hydro::Tags::LorentzFactor<DataType>> /*meta*/) const {
  using velocity_tag = hydro::Tags::SpatialVelocity<DataType, 3>;
  const auto velocity =
      get<velocity_tag>(variables(x, tmpl::list<velocity_tag>{}));
  return {hydro::lorentz_factor(dot_product(velocity, velocity))};
}

template <typename DataType>
tuples::TaggedTuple<hydro::Tags::SpecificEnthalpy<DataType>>
KhInstability::variables(
    const tnsr::I<DataType, 3>& x,
    tmpl::list<hydro::Tags::SpecificEnthalpy<DataType>> /*meta*/) const {
  return hydro::relativistic_specific_enthalpy(
      get<hydro::Tags::RestMassDensity<DataType>>(
          variables(x, tmpl::list<hydro::Tags::RestMassDensity<DataType>>{})),
      get<hydro::Tags::SpecificInternalEnergy<DataType>>(variables(
          x, tmpl::list<hydro::Tags::SpecificInternalEnergy<DataType>>{})),
      get<hydro::Tags::Pressure<DataType>>(
          variables(x, tmpl::list<hydro::Tags::Pressure<DataType>>{})));
}

PUP::able::PUP_ID KhInstability::my_PUP_ID = 0;

bool operator==(const KhInstability& lhs, const KhInstability& rhs) {
  // No comparison for equation_of_state_. Comparing adiabatic_index_ should
  // suffice.
  return lhs.adiabatic_index_ == rhs.adiabatic_index_ and
         lhs.strip_bimedian_height_ == rhs.strip_bimedian_height_ and
         lhs.strip_half_thickness_ == rhs.strip_half_thickness_ and
         lhs.strip_density_ == rhs.strip_density_ and
         lhs.strip_velocity_ == rhs.strip_velocity_ and
         lhs.background_density_ == rhs.background_density_ and
         lhs.background_velocity_ == rhs.background_velocity_ and
         lhs.pressure_ == rhs.pressure_ and
         lhs.perturbation_amplitude_ == rhs.perturbation_amplitude_ and
         lhs.perturbation_width_ == rhs.perturbation_width_ and
         lhs.magnetic_field_ == rhs.magnetic_field_;
}

bool operator!=(const KhInstability& lhs, const KhInstability& rhs) {
  return not(lhs == rhs);
}

#define DTYPE(data) BOOST_PP_TUPLE_ELEM(0, data)
#define TAG(data) BOOST_PP_TUPLE_ELEM(1, data)

#define INSTANTIATE_SCALARS(_, data)                         \
  template tuples::TaggedTuple<TAG(data) < DTYPE(data)> >    \
      KhInstability::variables(                              \
          const tnsr::I<DTYPE(data), 3, Frame::Inertial>& x, \
          tmpl::list<TAG(data) < DTYPE(data)> >) const;

GENERATE_INSTANTIATIONS(
    INSTANTIATE_SCALARS, (double, DataVector),
    (hydro::Tags::RestMassDensity, hydro::Tags::SpecificInternalEnergy,
     hydro::Tags::Pressure, hydro::Tags::DivergenceCleaningField,
     hydro::Tags::SpecificEnthalpy, hydro::Tags::LorentzFactor))

#define INSTANTIATE_VECTORS(_, data)                                          \
  template tuples::TaggedTuple<TAG(data) < DTYPE(data), 3, Frame::Inertial> > \
      KhInstability::variables(                                               \
          const tnsr::I<DTYPE(data), 3, Frame::Inertial>& x,                  \
          tmpl::list<TAG(data) < DTYPE(data), 3, Frame::Inertial> >) const;

GENERATE_INSTANTIATIONS(INSTANTIATE_VECTORS, (double, DataVector),
                        (hydro::Tags::SpatialVelocity,
                         hydro::Tags::MagneticField))

#undef INSTANTIATE_VECTORS
#undef INSTANTIATE_SCALARS
#undef TAG
#undef DTYPE
}  // namespace grmhd::AnalyticData
