// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "PointwiseFunctions/AnalyticSolutions/NewtonianEuler/IsentropicVortex.hpp"

#include <cmath>
#include <cstddef>
#include <limits>
#include <pup.h>  // IWYU pragma: keep

#include "DataStructures/DataVector.hpp"  // IWYU pragma: keep
#include "DataStructures/Tensor/Tensor.hpp"
#include "Evolution/Systems/NewtonianEuler/Sources/VortexPerturbation.hpp"
#include "PointwiseFunctions/Hydro/EquationsOfState/EquationOfState.hpp"  // IWYU pragma: keep
#include "PointwiseFunctions/Hydro/EquationsOfState/PolytropicFluid.hpp"  // IWYU pragma: keep
#include "Utilities/ConstantExpressions.hpp"
#include "Utilities/ErrorHandling/Assert.hpp"
#include "Utilities/GenerateInstantiations.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/MakeWithValue.hpp"

// IWYU pragma: no_include <complex>

namespace NewtonianEuler::Solutions {

template <size_t Dim>
IsentropicVortex<Dim>::IsentropicVortex(
    const double adiabatic_index, const std::array<double, Dim>& center,
    const std::array<double, Dim>& mean_velocity, const double strength,
    const double perturbation_amplitude)
    : adiabatic_index_(adiabatic_index),
      center_(center),
      mean_velocity_(mean_velocity),
      perturbation_amplitude_(perturbation_amplitude),
      strength_(strength),
      // Polytropic constant is set equal to 1.0
      equation_of_state_(1.0, adiabatic_index) {
  if (Dim == 2) {
    ASSERT(
        abs(perturbation_amplitude_) < std::numeric_limits<double>::epsilon(),
        "A nonzero perturbation amplitude only makes sense in 3 dimensions. "
        "The value given was "
            << perturbation_amplitude);
  }

  ASSERT(strength_ >= 0.0,
         "The strength must be non-negative. The value given "
         "was "
             << strength_ << ".");
}

template <size_t Dim>
void IsentropicVortex<Dim>::pup(PUP::er& p) {
  p | adiabatic_index_;
  p | center_;
  p | mean_velocity_;
  p | perturbation_amplitude_;
  p | strength_;
  p | equation_of_state_;
  p | source_term_;
}

// Can be any smooth function of z. For testing purposes, we choose sin(z).
template <size_t Dim>
template <typename DataType>
DataType IsentropicVortex<Dim>::perturbation_profile(const DataType& z) const {
  return sin(z);
}

template <size_t Dim>
template <typename DataType>
DataType IsentropicVortex<Dim>::deriv_of_perturbation_profile(
    const DataType& z) const {
  return cos(z);
}

template <size_t Dim>
template <typename DataType>
IsentropicVortex<Dim>::IntermediateVariables<DataType>::IntermediateVariables(
    const tnsr::I<DataType, Dim, Frame::Inertial>& x, const double t,
    const std::array<double, Dim>& center,
    const std::array<double, Dim>& mean_velocity, const double strength) {
  x_tilde = get<0>(x) - center[0] - t * mean_velocity[0];
  y_tilde = get<1>(x) - center[1] - t * mean_velocity[1];
  profile = 0.5 * strength *
            exp(0.5 - 0.5 * (square(x_tilde) + square(y_tilde))) / M_PI;
  if (Dim == 3) {
    z_coord = get<Dim - 1>(x);
  }
}

template <size_t Dim>
template <typename DataType>
tuples::TaggedTuple<Tags::MassDensity<DataType>>
IsentropicVortex<Dim>::variables(
    tmpl::list<Tags::MassDensity<DataType>> /*meta*/,
    const IntermediateVariables<DataType>& vars) const {
  const double adiabatic_index_minus_one = adiabatic_index_ - 1.0;
  return Scalar<DataType>(pow(1.0 - 0.5 * adiabatic_index_minus_one *
                                        square(vars.profile) / adiabatic_index_,
                              1.0 / adiabatic_index_minus_one));
}

template <size_t Dim>
template <typename DataType>
tuples::TaggedTuple<Tags::Velocity<DataType, Dim>>
IsentropicVortex<Dim>::variables(
    tmpl::list<Tags::Velocity<DataType, Dim>> /*meta*/,
    const IntermediateVariables<DataType>& vars) const {
  auto velocity = make_with_value<tnsr::I<DataType, Dim, Frame::Inertial>>(
      vars.y_tilde, 0.0);
  for (size_t i = 0; i < Dim; ++i) {
    velocity.get(i) = gsl::at(mean_velocity_, i);
  }
  get<0>(velocity) -= vars.y_tilde * vars.profile;
  get<1>(velocity) += vars.x_tilde * vars.profile;
  if (Dim == 3) {
    get<Dim - 1>(velocity) +=
        perturbation_amplitude_ * perturbation_profile(vars.z_coord);
  }
  return velocity;
}

template <size_t Dim>
template <typename DataType>
tuples::TaggedTuple<Tags::SpecificInternalEnergy<DataType>>
IsentropicVortex<Dim>::variables(
    tmpl::list<Tags::SpecificInternalEnergy<DataType>> /*meta*/,
    const IntermediateVariables<DataType>& vars) const {
  return equation_of_state_.specific_internal_energy_from_density(
      get<Tags::MassDensity<DataType>>(
          variables(tmpl::list<Tags::MassDensity<DataType>>{}, vars)));
}

template <size_t Dim>
template <typename DataType>
tuples::TaggedTuple<Tags::Pressure<DataType>> IsentropicVortex<Dim>::variables(
    tmpl::list<Tags::Pressure<DataType>> /*meta*/,
    const IntermediateVariables<DataType>& vars) const {
  return equation_of_state_.pressure_from_density(
      get<Tags::MassDensity<DataType>>(
          variables(tmpl::list<Tags::MassDensity<DataType>>{}, vars)));
}

template <size_t Dim>
bool operator==(const IsentropicVortex<Dim>& lhs,
                const IsentropicVortex<Dim>& rhs) {
  // No comparison for equation_of_state_ or source_term_. Comparing individual
  // members should suffice.
  return lhs.adiabatic_index_ == rhs.adiabatic_index_ and
         lhs.center_ == rhs.center_ and
         lhs.mean_velocity_ == rhs.mean_velocity_ and
         lhs.perturbation_amplitude_ == rhs.perturbation_amplitude_ and
         lhs.strength_ == rhs.strength_;
}

template <size_t Dim>
bool operator!=(const IsentropicVortex<Dim>& lhs,
                const IsentropicVortex<Dim>& rhs) {
  return not(lhs == rhs);
}

#define DIM(data) BOOST_PP_TUPLE_ELEM(0, data)
#define DTYPE(data) BOOST_PP_TUPLE_ELEM(1, data)
#define TAG(data) BOOST_PP_TUPLE_ELEM(2, data)

#define INSTANTIATE_CLASS(_, data)                              \
  template class IsentropicVortex<DIM(data)>;                   \
  template bool operator==(const IsentropicVortex<DIM(data)>&,  \
                           const IsentropicVortex<DIM(data)>&); \
  template bool operator!=(const IsentropicVortex<DIM(data)>&,  \
                           const IsentropicVortex<DIM(data)>&);

GENERATE_INSTANTIATIONS(INSTANTIATE_CLASS, (2, 3))

#define INSTANTIATE_MEMBERS(_, data)                                           \
  template struct IsentropicVortex<DIM(data)>::IntermediateVariables<DTYPE(    \
      data)>;                                                                  \
  template DTYPE(data)                                                         \
      IsentropicVortex<DIM(data)>::perturbation_profile(const DTYPE(data) & z) \
          const;                                                               \
  template DTYPE(data)                                                         \
      IsentropicVortex<DIM(data)>::deriv_of_perturbation_profile(              \
          const DTYPE(data) & z) const;

GENERATE_INSTANTIATIONS(INSTANTIATE_MEMBERS, (2, 3), (double, DataVector))

#define INSTANTIATE_SCALARS(_, data)                      \
  template tuples::TaggedTuple<TAG(data) < DTYPE(data)> > \
      IsentropicVortex<DIM(data)>::variables(             \
          tmpl::list<TAG(data) < DTYPE(data)> >,          \
          const IntermediateVariables<DTYPE(data)>&) const;

GENERATE_INSTANTIATIONS(INSTANTIATE_SCALARS, (2, 3), (double, DataVector),
                        (Tags::MassDensity, Tags::SpecificInternalEnergy,
                         Tags::Pressure))

#define INSTANTIATE_VELOCITY(_, data)                                \
  template tuples::TaggedTuple<TAG(data) < DTYPE(data), DIM(data)> > \
      IsentropicVortex<DIM(data)>::variables(                        \
          tmpl::list<TAG(data) < DTYPE(data), DIM(data)> >,          \
          const IntermediateVariables<DTYPE(data)>&) const;

GENERATE_INSTANTIATIONS(INSTANTIATE_VELOCITY, (2, 3), (double, DataVector),
                        (Tags::Velocity))

#undef DIM
#undef DTYPE
#undef TAG
#undef INSTANTIATE_CLASS
#undef INSTANTIATE_SCALARS
#undef INSTANTIATE_VELOCITY

}  // namespace NewtonianEuler::Solutions
