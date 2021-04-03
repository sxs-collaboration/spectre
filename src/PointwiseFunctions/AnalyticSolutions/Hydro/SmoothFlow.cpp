// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "PointwiseFunctions/AnalyticSolutions/Hydro/SmoothFlow.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <numeric>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "PointwiseFunctions/Hydro/Tags.hpp"
#include "Utilities/Functional.hpp"
#include "Utilities/GenerateInstantiations.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/MakeWithValue.hpp"
#include "Utilities/Numeric.hpp"

namespace hydro::Solutions {

template <size_t Dim, bool IsRelativistic>
SmoothFlow<Dim, IsRelativistic>::SmoothFlow(
    const std::array<double, Dim>& mean_velocity,
    const std::array<double, Dim>& wavevector, const double pressure,
    const double adiabatic_index, const double perturbation_size) noexcept
    : mean_velocity_(mean_velocity),
      wavevector_(wavevector),
      pressure_(pressure),
      adiabatic_index_(adiabatic_index),
      perturbation_size_(perturbation_size),
      k_dot_v_(std::inner_product(mean_velocity_.begin(), mean_velocity_.end(),
                                  wavevector_.begin(), 0.0)),
      equation_of_state_{adiabatic_index_} {}

template <size_t Dim, bool IsRelativistic>
SmoothFlow<Dim, IsRelativistic>::SmoothFlow(
    CkMigrateMessage* /*unused*/) noexcept {}

template <size_t Dim, bool IsRelativistic>
void SmoothFlow<Dim, IsRelativistic>::pup(PUP::er& p) noexcept {
  p | mean_velocity_;
  p | wavevector_;
  p | pressure_;
  p | adiabatic_index_;
  p | perturbation_size_;
  p | k_dot_v_;
  p | equation_of_state_;
}

template <size_t Dim, bool IsRelativistic>
template <typename DataType>
DataType SmoothFlow<Dim, IsRelativistic>::k_dot_x_minus_vt(
    const tnsr::I<DataType, Dim>& x, const double t) const noexcept {
  auto result = make_with_value<DataType>(x, -k_dot_v_ * t);
  for (size_t i = 0; i < Dim; i++) {
    result += gsl::at(wavevector_, i) * x.get(i);
  }
  return result;
}

// Primitive variables.
template <size_t Dim, bool IsRelativistic>
template <typename DataType>
tuples::TaggedTuple<hydro::Tags::RestMassDensity<DataType>>
SmoothFlow<Dim, IsRelativistic>::variables(
    const tnsr::I<DataType, Dim>& x, double t,
    tmpl::list<hydro::Tags::RestMassDensity<DataType>> /*meta*/)
    const noexcept {
  const DataType phase = k_dot_x_minus_vt(x, t);
  return {Scalar<DataType>{DataType{1.0 + perturbation_size_ * sin(phase)}}};
}

template <size_t Dim, bool IsRelativistic>
template <typename DataType>
tuples::TaggedTuple<hydro::Tags::SpecificInternalEnergy<DataType>>
SmoothFlow<Dim, IsRelativistic>::variables(
    const tnsr::I<DataType, Dim>& x, double t,
    tmpl::list<hydro::Tags::SpecificInternalEnergy<DataType>> /*meta*/)
    const noexcept {
  const DataType phase = k_dot_x_minus_vt(x, t);
  return {
      Scalar<DataType>{pressure_ / ((adiabatic_index_ - 1.0) *
                                    (1.0 + perturbation_size_ * sin(phase)))}};
}

template <size_t Dim, bool IsRelativistic>
template <typename DataType>
tuples::TaggedTuple<hydro::Tags::Pressure<DataType>>
SmoothFlow<Dim, IsRelativistic>::variables(
    const tnsr::I<DataType, Dim>& x, double /*t*/,
    tmpl::list<hydro::Tags::Pressure<DataType>> /*meta*/) const noexcept {
  return {make_with_value<Scalar<DataType>>(x, pressure_)};
}

template <size_t Dim, bool IsRelativistic>
template <typename DataType>
tuples::TaggedTuple<hydro::Tags::SpatialVelocity<DataType, Dim>>
SmoothFlow<Dim, IsRelativistic>::variables(
    const tnsr::I<DataType, Dim>& x, double /*t*/,
    tmpl::list<hydro::Tags::SpatialVelocity<DataType, Dim>> /*meta*/)
    const noexcept {
  auto result = make_with_value<tnsr::I<DataType, Dim>>(x, 0.0);
  for (size_t i = 0; i < Dim; ++i) {
    result.get(i) = gsl::at(mean_velocity_, i);
  }
  return {std::move(result)};
}

template <size_t Dim, bool IsRelativistic>
template <typename DataType, bool LocalIsRelativistic,
          Requires<IsRelativistic and IsRelativistic == LocalIsRelativistic>>
tuples::TaggedTuple<hydro::Tags::LorentzFactor<DataType>>
SmoothFlow<Dim, IsRelativistic>::variables(
    const tnsr::I<DataType, Dim>& x, double /*t*/,
    tmpl::list<hydro::Tags::LorentzFactor<DataType>> /*meta*/) const noexcept {
  return {make_with_value<Scalar<DataType>>(
      x,
      1.0 / sqrt(1.0 - alg::accumulate(
                           mean_velocity_, 0.0,
                           funcl::Plus<funcl::Identity, funcl::Square<>>{})))};
}

template <size_t Dim, bool IsRelativistic>
template <typename DataType>
tuples::TaggedTuple<hydro::Tags::SpecificEnthalpy<DataType>>
SmoothFlow<Dim, IsRelativistic>::variables(
    const tnsr::I<DataType, Dim>& x, double t,
    tmpl::list<hydro::Tags::SpecificEnthalpy<DataType>> /*meta*/)
    const noexcept {
  Scalar<DataType> specific_internal_energy = std::move(
      get<hydro::Tags::SpecificInternalEnergy<DataType>>(variables<DataType>(
          x, t, tmpl::list<hydro::Tags::SpecificInternalEnergy<DataType>>{})));
  get(specific_internal_energy) *= adiabatic_index_;
  if constexpr (IsRelativistic) {
    get(specific_internal_energy) += 1.0;
  }
  return {std::move(specific_internal_energy)};
}

template <size_t Dim, bool IsRelativistic>
bool operator==(const SmoothFlow<Dim, IsRelativistic>& lhs,
                const SmoothFlow<Dim, IsRelativistic>& rhs) noexcept {
  // there is no comparison operator for the EoS, but should be okay as
  // the adiabatic_indexs are compared
  return lhs.mean_velocity_ == rhs.mean_velocity_ and
         lhs.wavevector_ == rhs.wavevector_ and
         lhs.pressure_ == rhs.pressure_ and
         lhs.adiabatic_index_ == rhs.adiabatic_index_ and
         lhs.perturbation_size_ == rhs.perturbation_size_ and
         lhs.k_dot_v_ == rhs.k_dot_v_;
}

template <size_t Dim, bool IsRelativistic>
bool operator!=(const SmoothFlow<Dim, IsRelativistic>& lhs,
                const SmoothFlow<Dim, IsRelativistic>& rhs) noexcept {
  return not(lhs == rhs);
}

#define DIM(data) BOOST_PP_TUPLE_ELEM(0, data)
#define IS_RELATIVISTIC(data) BOOST_PP_TUPLE_ELEM(1, data)
#define DTYPE(data) BOOST_PP_TUPLE_ELEM(2, data)
#define TAG(data) BOOST_PP_TUPLE_ELEM(3, data)

#define INSTANTIATE_CLASS(_, data)                                   \
  template class SmoothFlow<DIM(data), IS_RELATIVISTIC(data)>;       \
  template bool operator==(                                          \
      const SmoothFlow<DIM(data), IS_RELATIVISTIC(data)>&,           \
      const SmoothFlow<DIM(data), IS_RELATIVISTIC(data)>&) noexcept; \
  template bool operator!=(                                          \
      const SmoothFlow<DIM(data), IS_RELATIVISTIC(data)>&,           \
      const SmoothFlow<DIM(data), IS_RELATIVISTIC(data)>&) noexcept;

GENERATE_INSTANTIATIONS(INSTANTIATE_CLASS, (1, 2, 3), (true, false))

#define INSTANTIATE_SCALARS(_, data)                           \
  template tuples::TaggedTuple<TAG(data) < DTYPE(data)> >      \
      SmoothFlow<DIM(data), IS_RELATIVISTIC(data)>::variables( \
          const tnsr::I<DTYPE(data), DIM(data)>& x, double t,  \
          tmpl::list<TAG(data) < DTYPE(data)> > /*meta*/) const noexcept;

GENERATE_INSTANTIATIONS(INSTANTIATE_SCALARS, (1, 2, 3), (true, false),
                        (double, DataVector),
                        (hydro::Tags::RestMassDensity,
                         hydro::Tags::SpecificInternalEnergy,
                         hydro::Tags::Pressure, hydro::Tags::SpecificEnthalpy))

#define INSTANTIATE_LORENTZ_FACTOR(_, data)                             \
  template tuples::TaggedTuple<hydro::Tags::LorentzFactor<DTYPE(data)>> \
  SmoothFlow<DIM(data), IS_RELATIVISTIC(data)>::variables(              \
      const tnsr::I<DTYPE(data), DIM(data)>& x, double t,               \
      tmpl::list<hydro::Tags::LorentzFactor<DTYPE(data)>> /*meta*/)     \
      const noexcept;

GENERATE_INSTANTIATIONS(INSTANTIATE_LORENTZ_FACTOR, (1, 2, 3), (true),
                        (double, DataVector))

#define INSTANTIATE_VECTORS(_, data)                                 \
  template tuples::TaggedTuple<TAG(data) < DTYPE(data), DIM(data),   \
                               Frame::Inertial> >                    \
      SmoothFlow<DIM(data), IS_RELATIVISTIC(data)>::variables(       \
          const tnsr::I<DTYPE(data), DIM(data)>& x, double t,        \
          tmpl::list<TAG(data) < DTYPE(data), DIM(data)> > /*meta*/) \
          const noexcept;

GENERATE_INSTANTIATIONS(INSTANTIATE_VECTORS, (1, 2, 3), (true, false),
                        (double, DataVector), (hydro::Tags::SpatialVelocity))

#undef DIM
#undef IS_RELATIVISTIC
#undef DTYPE
#undef TAG
#undef INSTANTIATE_CLASS
#undef INSTANTIATE_SCALARS
#undef INSTANTIATE_LORENTZ_FACTOR
#undef INSTANTIATE_VECTORS
}  // namespace hydro::Solutions
