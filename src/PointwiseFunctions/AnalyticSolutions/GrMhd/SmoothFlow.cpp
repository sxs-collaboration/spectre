// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "PointwiseFunctions/AnalyticSolutions/GrMhd/SmoothFlow.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <numeric>

#include "DataStructures/DataBox/DataBoxTag.hpp"
#include "DataStructures/DataVector.hpp"     // IWYU pragma: keep
#include "DataStructures/Tensor/Tensor.hpp"  // IWYU pragma: keep
#include "Parallel/PupStlCpp11.hpp"
#include "Utilities/Functional.hpp"
#include "Utilities/GenerateInstantiations.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/MakeWithValue.hpp"
#include "Utilities/Numeric.hpp"

// IWYU pragma:  no_include "DataStructures/Tensor/TypeAliases.hpp"

/// \cond
namespace grmhd {
namespace Solutions {

SmoothFlow::SmoothFlow(MeanVelocity::type mean_velocity,
                       WaveVector::type wavevector,
                       const Pressure::type pressure,
                       const AdiabaticIndex::type adiabatic_index,
                       const PerturbationSize::type perturbation_size) noexcept
    :  // clang-tidy: do not std::move trivial types.
      mean_velocity_(std::move(mean_velocity)),  // NOLINT
      // clang-tidy: do not std::move trivial types.
      wavevector_(std::move(wavevector)),  // NOLINT
      pressure_(pressure),
      adiabatic_index_(adiabatic_index),
      perturbation_size_(perturbation_size),
      k_dot_v_(std::inner_product(mean_velocity_.begin(), mean_velocity_.end(),
                                  wavevector_.begin(), 0.0)),
      equation_of_state_{adiabatic_index_} {}

void SmoothFlow::pup(PUP::er& p) noexcept {
  p | mean_velocity_;
  p | wavevector_;
  p | pressure_;
  p | adiabatic_index_;
  p | perturbation_size_;
  p | k_dot_v_;
  p | equation_of_state_;
  p | background_spacetime_;
}

template <typename DataType>
DataType SmoothFlow::k_dot_x_minus_vt(const tnsr::I<DataType, 3>& x,
                                      const double t) const noexcept {
  auto result = make_with_value<DataType>(x, -k_dot_v_ * t);
  for (size_t i = 0; i < 3; i++) {
    result += gsl::at(wavevector_, i) * x.get(i);
  }
  return result;
}

// Primitive variables.
template <typename DataType>
tuples::TaggedTuple<hydro::Tags::RestMassDensity<DataType>>
SmoothFlow::variables(
    const tnsr::I<DataType, 3>& x, double t,
    tmpl::list<hydro::Tags::RestMassDensity<DataType>> /*meta*/) const
    noexcept {
  const DataType phase = k_dot_x_minus_vt(x, t);
  return {Scalar<DataType>{DataType{1.0 + perturbation_size_ * sin(phase)}}};
}

template <typename DataType>
tuples::TaggedTuple<hydro::Tags::SpecificInternalEnergy<DataType>>
SmoothFlow::variables(
    const tnsr::I<DataType, 3>& x, double t,
    tmpl::list<hydro::Tags::SpecificInternalEnergy<DataType>> /*meta*/) const
    noexcept {
  const DataType phase = k_dot_x_minus_vt(x, t);
  return {
      Scalar<DataType>{pressure_ / ((adiabatic_index_ - 1.0) *
                                    (1.0 + perturbation_size_ * sin(phase)))}};
}

template <typename DataType>
tuples::TaggedTuple<hydro::Tags::Pressure<DataType>> SmoothFlow::variables(
    const tnsr::I<DataType, 3>& x, double /*t*/,
    tmpl::list<hydro::Tags::Pressure<DataType>> /*meta*/) const noexcept {
  return {make_with_value<Scalar<DataType>>(x, pressure_)};
}

template <typename DataType>
tuples::TaggedTuple<hydro::Tags::SpatialVelocity<DataType, 3, Frame::Inertial>>
SmoothFlow::variables(const tnsr::I<DataType, 3>& x, double /*t*/,
                      tmpl::list<hydro::Tags::SpatialVelocity<
                          DataType, 3, Frame::Inertial>> /*meta*/) const
    noexcept {
  auto result = make_with_value<db::item_type<
      hydro::Tags::SpatialVelocity<DataType, 3, Frame::Inertial>>>(
      x, mean_velocity_[0]);
  get<1>(result) = mean_velocity_[1];
  get<2>(result) = mean_velocity_[2];
  return {std::move(result)};
}

template <typename DataType>
tuples::TaggedTuple<hydro::Tags::MagneticField<DataType, 3, Frame::Inertial>>
SmoothFlow::variables(const tnsr::I<DataType, 3>& x, double /*t*/,
                      tmpl::list<hydro::Tags::MagneticField<
                          DataType, 3, Frame::Inertial>> /*meta*/) const
    noexcept {
  return {make_with_value<
      db::item_type<hydro::Tags::MagneticField<DataType, 3, Frame::Inertial>>>(
      x, 0.0)};
}

template <typename DataType>
tuples::TaggedTuple<hydro::Tags::DivergenceCleaningField<DataType>>
SmoothFlow::variables(
    const tnsr::I<DataType, 3>& x, double /*t*/,
    tmpl::list<hydro::Tags::DivergenceCleaningField<DataType>> /*meta*/) const
    noexcept {
  return {make_with_value<
      db::item_type<hydro::Tags::DivergenceCleaningField<DataType>>>(x, 0.0)};
}

template <typename DataType>
tuples::TaggedTuple<hydro::Tags::LorentzFactor<DataType>> SmoothFlow::variables(
    const tnsr::I<DataType, 3>& x, double /*t*/,
    tmpl::list<hydro::Tags::LorentzFactor<DataType>> /*meta*/) const noexcept {
  return {make_with_value<db::item_type<hydro::Tags::LorentzFactor<DataType>>>(
      x,
      1.0 / sqrt(1.0 - alg::accumulate(
                           mean_velocity_, 0.0,
                           funcl::Plus<funcl::Identity, funcl::Square<>>{})))};
}

template <typename DataType>
tuples::TaggedTuple<hydro::Tags::SpecificEnthalpy<DataType>>
SmoothFlow::variables(
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

bool operator==(const SmoothFlow& lhs, const SmoothFlow& rhs) noexcept {
  // there is no comparison operator for the EoS, but should be okay as
  // the adiabatic_indexs are compared
  return lhs.mean_velocity_ == rhs.mean_velocity_ and
         lhs.wavevector_ == rhs.wavevector_ and
         lhs.pressure_ == rhs.pressure_ and
         lhs.adiabatic_index_ == rhs.adiabatic_index_ and
         lhs.perturbation_size_ == rhs.perturbation_size_ and
         lhs.k_dot_v_ == rhs.k_dot_v_ and
         lhs.background_spacetime_ == rhs.background_spacetime_;
}

bool operator!=(const SmoothFlow& lhs, const SmoothFlow& rhs) noexcept {
  return not(lhs == rhs);
}

#define DTYPE(data) BOOST_PP_TUPLE_ELEM(0, data)
#define TAG(data) BOOST_PP_TUPLE_ELEM(1, data)

#define INSTANTIATE_SCALARS(_, data)                                       \
  template tuples::TaggedTuple<TAG(data) < DTYPE(data)>>                   \
      SmoothFlow::variables(const tnsr::I<DTYPE(data), 3>& x, double t,    \
                            tmpl::list<TAG(data) < DTYPE(data)>> /*meta*/) \
          const noexcept;

GENERATE_INSTANTIATIONS(
    INSTANTIATE_SCALARS, (double, DataVector),
    (hydro::Tags::RestMassDensity, hydro::Tags::SpecificInternalEnergy,
     hydro::Tags::Pressure, hydro::Tags::DivergenceCleaningField,
     hydro::Tags::LorentzFactor, hydro::Tags::SpecificEnthalpy))

#define INSTANTIATE_VECTORS(_, data)                                         \
  template tuples::TaggedTuple<TAG(data) < DTYPE(data), 3, Frame::Inertial>> \
      SmoothFlow::variables(                                                 \
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
