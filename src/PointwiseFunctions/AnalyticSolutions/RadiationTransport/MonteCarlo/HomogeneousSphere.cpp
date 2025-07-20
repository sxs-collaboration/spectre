// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "PointwiseFunctions/AnalyticSolutions/RadiationTransport/MonteCarlo/HomogeneousSphere.hpp"

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/EagerMath/Magnitude.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "PointwiseFunctions/Hydro/EquationsOfState/EquationOfState.hpp"
#include "PointwiseFunctions/Hydro/Tags.hpp"
#include "Utilities/GenerateInstantiations.hpp"
#include "Utilities/MakeWithValue.hpp"

namespace RadiationTransport::MonteCarlo::Solutions {

namespace {
double get_mask(const tnsr::I<double, 3>& x, const double radius_bound) {
  const double radius = get(magnitude(x));
  return radius > radius_bound ? 1.0 : 0.0;
}

DataVector get_mask(const tnsr::I<DataVector, 3>& x,
                    const double radius_bound) {
  const DataVector radius = get(magnitude(x));
  DataVector mask = make_with_value<DataVector>(radius, 0.0);
  for (size_t i = 0; i < mask.size(); i++) {
    mask[i] = radius[i] > radius_bound ? 1.0 : 0.0;
  }
  return mask;
}
}  // namespace

HomogeneousSphere::HomogeneousSphere(
    const double& radius, const std::array<double, 2>& densities,
    const std::array<double, 2>& temperatures,
    const std::array<double, 2>& electron_fractions,
    std::unique_ptr<EquationsOfState::EquationOfState<IsRelativistic, 3>>
        local_eos)
    : radius_(radius),
      densities_(densities),
      temperatures_(temperatures),
      electron_fractions_(electron_fractions),
      equation_of_state_(std::move(local_eos)) {}

std::unique_ptr<evolution::initial_data::InitialData>
HomogeneousSphere::get_clone() const {
  return std::make_unique<HomogeneousSphere>(*this);
}

HomogeneousSphere::HomogeneousSphere(CkMigrateMessage* /*unused*/) {}

HomogeneousSphere::HomogeneousSphere(const HomogeneousSphere& rhs)
    : evolution::initial_data::InitialData(rhs),
      radius_(rhs.radius_),
      densities_(rhs.densities_),
      temperatures_(rhs.temperatures_),
      electron_fractions_(rhs.electron_fractions_),
      equation_of_state_(rhs.equation_of_state_->get_clone()) {}

HomogeneousSphere& HomogeneousSphere::operator=(const HomogeneousSphere& rhs) {
  radius_ = rhs.radius_;
  densities_ = rhs.densities_;
  temperatures_ = rhs.temperatures_;
  electron_fractions_ = rhs.electron_fractions_;
  equation_of_state_ = rhs.equation_of_state_->get_clone();
  return *this;
}

void HomogeneousSphere::pup(PUP::er& p) {
  p | radius_;
  p | densities_;
  p | temperatures_;
  p | electron_fractions_;
  p | equation_of_state_;
}

template <typename DataType>
tuples::TaggedTuple<hydro::Tags::ElectronFraction<DataType>>
HomogeneousSphere::variables(
    const tnsr::I<DataType, 3>& x, double /*t*/,
    tmpl::list<hydro::Tags::ElectronFraction<DataType>> /*meta*/) const {
  const DataType mask = get_mask(x, radius_);
  return {Scalar<DataType>{DataType{electron_fractions_[1] * mask +
                                    electron_fractions_[0] * (1.0 - mask)}}};
}

template <typename DataType>
tuples::TaggedTuple<hydro::Tags::RestMassDensity<DataType>>
HomogeneousSphere::variables(
    const tnsr::I<DataType, 3>& x, double /*t*/,
    tmpl::list<hydro::Tags::RestMassDensity<DataType>> /*meta*/) const {
  const DataType mask = get_mask(x, radius_);
  return {Scalar<DataType>{
      DataType{densities_[1] * mask + densities_[0] * (1.0 - mask)}}};
}

template <typename DataType>
tuples::TaggedTuple<hydro::Tags::Temperature<DataType>>
HomogeneousSphere::variables(
    const tnsr::I<DataType, 3>& x, double /*t*/,
    tmpl::list<hydro::Tags::Temperature<DataType>> /*meta*/) const {
  const DataType mask = get_mask(x, radius_);
  return {Scalar<DataType>{
      DataType{temperatures_[1] * mask + temperatures_[0] * (1.0 - mask)}}};
}

template <typename DataType>
tuples::TaggedTuple<hydro::Tags::LorentzFactor<DataType>>
HomogeneousSphere::variables(
    const tnsr::I<DataType, 3>& x, double /*t*/,
    tmpl::list<hydro::Tags::LorentzFactor<DataType>> /*meta*/) const {
  return {make_with_value<Scalar<DataType>>(x, 1.0)};
}

template <typename DataType>
tuples::TaggedTuple<hydro::Tags::SpecificInternalEnergy<DataType>>
HomogeneousSphere::variables(
    const tnsr::I<DataType, 3>& x, double t,
    tmpl::list<hydro::Tags::SpecificInternalEnergy<DataType>> /*meta*/) const {
  auto primitives = this->variables<DataType>(
      x, t,
      tmpl::list<hydro::Tags::RestMassDensity<DataType>,
                 hydro::Tags::Temperature<DataType>,
                 hydro::Tags::ElectronFraction<DataType>>{});
  return equation_of_state_
      ->specific_internal_energy_from_density_and_temperature(
          tuples::get<hydro::Tags::RestMassDensity<DataType>>(primitives),
          tuples::get<hydro::Tags::Temperature<DataType>>(primitives),
          tuples::get<hydro::Tags::ElectronFraction<DataType>>(primitives));
}

template <typename DataType>
tuples::TaggedTuple<hydro::Tags::Pressure<DataType>>
HomogeneousSphere::variables(
    const tnsr::I<DataType, 3>& x, double t,
    tmpl::list<hydro::Tags::Pressure<DataType>> /*meta*/) const {
  auto primitives = this->variables<DataType>(
      x, t,
      tmpl::list<hydro::Tags::RestMassDensity<DataType>,
                 hydro::Tags::Temperature<DataType>,
                 hydro::Tags::ElectronFraction<DataType>>{});
  return equation_of_state_->pressure_from_density_and_temperature(
      tuples::get<hydro::Tags::RestMassDensity<DataType>>(primitives),
      tuples::get<hydro::Tags::Temperature<DataType>>(primitives),
      tuples::get<hydro::Tags::ElectronFraction<DataType>>(primitives));
}

template <typename DataType>
tuples::TaggedTuple<hydro::Tags::SpecificEnthalpy<DataType>>
HomogeneousSphere::variables(
    const tnsr::I<DataType, 3>& x, double t,
    tmpl::list<hydro::Tags::SpecificEnthalpy<DataType>> /*meta*/) const {
  auto primitives = this->variables<DataType>(
      x, t,
      tmpl::list<hydro::Tags::RestMassDensity<DataType>,
                 hydro::Tags::Pressure<DataType>,
                 hydro::Tags::SpecificInternalEnergy<DataType>>{});
  return tuples::get<hydro::Tags::Pressure<DataType>>(primitives) /
             tuples::get<hydro::Tags::RestMassDensity<DataType>>(primitives) +
         tuples::get<hydro::Tags::SpecificInternalEnergy<DataType>>(
             primitives) +
         1.0;
}

template <typename DataType>
tuples::TaggedTuple<hydro::Tags::SpatialVelocity<DataType, 3>>
HomogeneousSphere::variables(
    const tnsr::I<DataType, 3>& x, double /*t*/,
    tmpl::list<hydro::Tags::SpatialVelocity<DataType, 3>> /*meta*/) const {
  return {make_with_value<tnsr::I<DataType, 3>>(x, 0.0)};
}

template <typename DataType>
tuples::TaggedTuple<hydro::Tags::MagneticField<DataType, 3>>
HomogeneousSphere::variables(
    const tnsr::I<DataType, 3>& x, double /*t*/,
    tmpl::list<hydro::Tags::MagneticField<DataType, 3>> /*meta*/) const {
  return {make_with_value<tnsr::I<DataType, 3>>(x, 0.0)};
}

template <typename DataType>
tuples::TaggedTuple<hydro::Tags::DivergenceCleaningField<DataType>>
HomogeneousSphere::variables(
    const tnsr::I<DataType, 3>& x, double /*t*/,
    tmpl::list<hydro::Tags::DivergenceCleaningField<DataType>> /*meta*/) const {
  return {make_with_value<Scalar<DataType>>(x, 0.0)};
}

#ifndef __CUDA_ARCH__
PUP::able::PUP_ID HomogeneousSphere::my_PUP_ID = 0;  // NOLINT
#endif                                               // __CUDA_ARCH__

bool operator==(const HomogeneousSphere& lhs, const HomogeneousSphere& rhs) {
  return (lhs.radius_ == rhs.radius_ && lhs.densities_ == rhs.densities_ &&
          lhs.temperatures_ == rhs.temperatures_ &&
          lhs.electron_fractions_ == rhs.electron_fractions_ &&
          *lhs.equation_of_state_ == *rhs.equation_of_state_);
}

bool operator!=(const HomogeneousSphere& lhs, const HomogeneousSphere& rhs) {
  return not(lhs == rhs);
}

#define DTYPE(data) BOOST_PP_TUPLE_ELEM(0, data)
#define TAG(data) BOOST_PP_TUPLE_ELEM(1, data)

#define INSTANTIATE_SCALARS(_, data)                      \
  template tuples::TaggedTuple<TAG(data) < DTYPE(data)> > \
      HomogeneousSphere::variables(                       \
          const tnsr::I<DTYPE(data), 3>& x, double t,     \
          tmpl::list<TAG(data) < DTYPE(data)> > /*meta*/) const;

GENERATE_INSTANTIATIONS(INSTANTIATE_SCALARS, (double, DataVector),
                        (hydro::Tags::DivergenceCleaningField,
                         hydro::Tags::RestMassDensity,
                         hydro::Tags::ElectronFraction,
                         hydro::Tags::Temperature, hydro::Tags::LorentzFactor,
                         hydro::Tags::SpecificInternalEnergy,
                         hydro::Tags::Pressure))

#define INSTANTIATE_VECTORS(_, data)                         \
  template tuples::TaggedTuple<TAG(data) < DTYPE(data), 3> > \
      HomogeneousSphere::variables(                          \
          const tnsr::I<DTYPE(data), 3>& x, double t,        \
          tmpl::list<TAG(data) < DTYPE(data), 3> > /*meta*/) const;

GENERATE_INSTANTIATIONS(INSTANTIATE_VECTORS, (double, DataVector),
                        (hydro::Tags::MagneticField,
                         hydro::Tags::SpatialVelocity))

#undef DTYPE
#undef TAG
#undef INSTANTIATE_SCALARS
#undef INSTANTIATE_VECTORS
}  // namespace RadiationTransport::MonteCarlo::Solutions
