// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "PointwiseFunctions/AnalyticSolutions/RelativisticEuler/SmoothFlow.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <numeric>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Parallel/PupStlCpp11.hpp"
#include "PointwiseFunctions/Hydro/Tags.hpp"
#include "Utilities/Functional.hpp"
#include "Utilities/GenerateInstantiations.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/MakeWithValue.hpp"
#include "Utilities/Numeric.hpp"

/// \cond
namespace RelativisticEuler {
namespace Solutions {

template <size_t Dim>
SmoothFlow<Dim>::SmoothFlow(const std::array<double, Dim> mean_velocity,
                            const std::array<double, Dim> wavevector,
                            const double pressure, const double adiabatic_index,
                            const double perturbation_size) noexcept
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

template <size_t Dim>
void SmoothFlow<Dim>::pup(PUP::er& p) noexcept {
  p | mean_velocity_;
  p | wavevector_;
  p | pressure_;
  p | adiabatic_index_;
  p | perturbation_size_;
  p | k_dot_v_;
  p | equation_of_state_;
  p | background_spacetime_;
}

template <size_t Dim>
template <typename DataType>
DataType SmoothFlow<Dim>::k_dot_x_minus_vt(const tnsr::I<DataType, Dim>& x,
                                           const double t) const noexcept {
  auto result = make_with_value<DataType>(x, -k_dot_v_ * t);
  for (size_t i = 0; i < Dim; i++) {
    result += gsl::at(wavevector_, i) * x.get(i);
  }
  return result;
}

// Primitive variables.
template <size_t Dim>
template <typename DataType>
tuples::TaggedTuple<hydro::Tags::RestMassDensity<DataType>>
SmoothFlow<Dim>::variables(
    const tnsr::I<DataType, Dim>& x, double t,
    tmpl::list<hydro::Tags::RestMassDensity<DataType>> /*meta*/) const
    noexcept {
  const DataType phase = k_dot_x_minus_vt(x, t);
  return {Scalar<DataType>{DataType{1.0 + perturbation_size_ * sin(phase)}}};
}

template <size_t Dim>
template <typename DataType>
tuples::TaggedTuple<hydro::Tags::SpecificInternalEnergy<DataType>>
SmoothFlow<Dim>::variables(
    const tnsr::I<DataType, Dim>& x, double t,
    tmpl::list<hydro::Tags::SpecificInternalEnergy<DataType>> /*meta*/) const
    noexcept {
  const DataType phase = k_dot_x_minus_vt(x, t);
  return {
      Scalar<DataType>{pressure_ / ((adiabatic_index_ - 1.0) *
                                    (1.0 + perturbation_size_ * sin(phase)))}};
}

template <size_t Dim>
template <typename DataType>
tuples::TaggedTuple<hydro::Tags::Pressure<DataType>> SmoothFlow<Dim>::variables(
    const tnsr::I<DataType, Dim>& x, double /*t*/,
    tmpl::list<hydro::Tags::Pressure<DataType>> /*meta*/) const noexcept {
  return {make_with_value<Scalar<DataType>>(x, pressure_)};
}

template <size_t Dim>
template <typename DataType>
tuples::TaggedTuple<hydro::Tags::SpatialVelocity<DataType, Dim>>
SmoothFlow<Dim>::variables(
    const tnsr::I<DataType, Dim>& x, double /*t*/,
    tmpl::list<hydro::Tags::SpatialVelocity<DataType, Dim>> /*meta*/) const
    noexcept {
  auto result = make_with_value<tnsr::I<DataType, Dim>>(x, 0.0);
  for (size_t i = 0; i < Dim; ++i) {
    result.get(i) = gsl::at(mean_velocity_, i);
  }
  return {std::move(result)};
}

template <size_t Dim>
template <typename DataType>
tuples::TaggedTuple<hydro::Tags::LorentzFactor<DataType>>
SmoothFlow<Dim>::variables(
    const tnsr::I<DataType, Dim>& x, double /*t*/,
    tmpl::list<hydro::Tags::LorentzFactor<DataType>> /*meta*/) const noexcept {
  return {make_with_value<Scalar<DataType>>(
      x,
      1.0 / sqrt(1.0 - alg::accumulate(
                           mean_velocity_, 0.0,
                           funcl::Plus<funcl::Identity, funcl::Square<>>{})))};
}

template <size_t Dim>
template <typename DataType>
tuples::TaggedTuple<hydro::Tags::SpecificEnthalpy<DataType>>
SmoothFlow<Dim>::variables(
    const tnsr::I<DataType, Dim>& x, double t,
    tmpl::list<hydro::Tags::SpecificEnthalpy<DataType>> /*meta*/) const
    noexcept {
  Scalar<DataType> specific_internal_energy = std::move(
      get<hydro::Tags::SpecificInternalEnergy<DataType>>(variables<DataType>(
          x, t, tmpl::list<hydro::Tags::SpecificInternalEnergy<DataType>>{})));
  get(specific_internal_energy) *= adiabatic_index_;
  get(specific_internal_energy) += 1.0;
  return {std::move(specific_internal_energy)};
}

template <size_t Dim>
bool operator==(const SmoothFlow<Dim>& lhs,
                const SmoothFlow<Dim>& rhs) noexcept {
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

template <size_t Dim>
bool operator!=(const SmoothFlow<Dim>& lhs,
                const SmoothFlow<Dim>& rhs) noexcept {
  return not(lhs == rhs);
}

#define DIM(data) BOOST_PP_TUPLE_ELEM(0, data)
#define DTYPE(data) BOOST_PP_TUPLE_ELEM(1, data)
#define TAG(data) BOOST_PP_TUPLE_ELEM(2, data)

#define INSTANTIATE_CLASS(_, data)                                 \
  template class SmoothFlow<DIM(data)>;                            \
  template bool operator==(const SmoothFlow<DIM(data)>&,           \
                           const SmoothFlow<DIM(data)>&) noexcept; \
  template bool operator!=(const SmoothFlow<DIM(data)>&,           \
                           const SmoothFlow<DIM(data)>&) noexcept;

GENERATE_INSTANTIATIONS(INSTANTIATE_CLASS, (1, 2, 3))

#define INSTANTIATE_SCALARS(_, data)                          \
  template tuples::TaggedTuple<TAG(data) < DTYPE(data)>>      \
      SmoothFlow<DIM(data)>::variables(                       \
          const tnsr::I<DTYPE(data), DIM(data)>& x, double t, \
          tmpl::list<TAG(data) < DTYPE(data)>> /*meta*/) const noexcept;

GENERATE_INSTANTIATIONS(INSTANTIATE_SCALARS, (1, 2, 3), (double, DataVector),
                        (hydro::Tags::RestMassDensity,
                         hydro::Tags::SpecificInternalEnergy,
                         hydro::Tags::Pressure, hydro::Tags::LorentzFactor,
                         hydro::Tags::SpecificEnthalpy))

#define INSTANTIATE_VECTORS(_, data)                                \
  template tuples::TaggedTuple<TAG(data) < DTYPE(data), DIM(data),  \
                               Frame::Inertial>>                    \
      SmoothFlow<DIM(data)>::variables(                             \
          const tnsr::I<DTYPE(data), DIM(data)>& x, double t,       \
          tmpl::list<TAG(data) < DTYPE(data), DIM(data)>> /*meta*/) \
          const noexcept;

GENERATE_INSTANTIATIONS(INSTANTIATE_VECTORS, (1, 2, 3), (double, DataVector),
                        (hydro::Tags::SpatialVelocity))

#undef DIM
#undef DTYPE
#undef TAG
#undef INSTANTIATE_CLASS
#undef INSTANTIATE_SCALARS
#undef INSTANTIATE_VECTORS
}  // namespace Solutions
}  // namespace RelativisticEuler
/// \endcond
