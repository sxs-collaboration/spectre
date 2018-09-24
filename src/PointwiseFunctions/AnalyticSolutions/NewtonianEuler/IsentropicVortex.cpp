// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "PointwiseFunctions/AnalyticSolutions/NewtonianEuler/IsentropicVortex.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <ostream>

#include "DataStructures/DataVector.hpp"                   // IWYU pragma: keep
#include "DataStructures/Tensor/EagerMath/DotProduct.hpp"  // IWYU pragma: keep
#include "DataStructures/Tensor/Tensor.hpp"
#include "ErrorHandling/Assert.hpp"
#include "Evolution/Systems/NewtonianEuler/ConservativeFromPrimitive.hpp"
#include "Parallel/PupStlCpp11.hpp"
#include "Utilities/GenerateInstantiations.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/MakeWithValue.hpp"

/// \cond
namespace NewtonianEuler {
namespace Solutions {

IsentropicVortex::IsentropicVortex(
    const double adiabatic_index, IsentropicVortex::Center::type center,
    IsentropicVortex::MeanVelocity::type mean_velocity,
    const double perturbation_amplitude, const double strength)
    : adiabatic_index_(adiabatic_index),
      // clang-tidy: do not std::move trivial types.
      center_(std::move(center)),  // NOLINT
      // clang-tidy: do not std::move trivial types.
      mean_velocity_(std::move(mean_velocity)),  // NOLINT
      perturbation_amplitude_(perturbation_amplitude),
      strength_(strength) {
  ASSERT(adiabatic_index_ > 1.0 and adiabatic_index_ < 2.0,
         "The adiabatic index must be in the range (1, 2). The value given "
         "was "
             << adiabatic_index_ << ".");
  ASSERT(strength_ >= 0.0,
         "The strength must be non-negative. The value given "
         "was "
             << strength_ << ".");
}

void IsentropicVortex::pup(PUP::er& p) noexcept {
  p | adiabatic_index_;
  p | center_;
  p | mean_velocity_;
  p | perturbation_amplitude_;
  p | strength_;
}

template <typename DataType>
Scalar<DataType> IsentropicVortex::perturbation(const DataType& coord_z) const
    noexcept {
  return Scalar<DataType>{perturbation_amplitude_ * sin(coord_z)};
}

template <typename DataType>
tuples::tagged_tuple_from_typelist<IsentropicVortex::primitive_t<DataType>>
IsentropicVortex::primitive_variables(const tnsr::I<DataType, 3>& x,
                                      const double t) const noexcept {
  const auto adiabatic_index_minus_one = adiabatic_index_ - 1.0;

  const auto x_tilde = [&x, &t, this ]() noexcept {
    auto l_x_tilde = make_with_value<tnsr::I<DataType, 2>>(x, 0.0);
    // Note: x_tilde has only 2 components as it is used to
    // compute a distance on a plane perpendicular to the z-axis.
    for (size_t i = 0; i < 2; ++i) {
      l_x_tilde.get(i) =
          x.get(i) - gsl::at(center_, i) - t * gsl::at(mean_velocity_, i);
    }
    return l_x_tilde;
  }
  ();

  auto result = make_with_value<tuples::tagged_tuple_from_typelist<
      IsentropicVortex::primitive_t<DataType>>>(x, 0.0);

  const DataType temp = 0.5 * strength_ *
                        exp(0.5 - 0.5 * get(dot_product(x_tilde, x_tilde))) /
                        M_PI;

  get<Tags::MassDensity<DataType>>(result) = Scalar<DataType>(pow(
      1.0 - 0.5 * adiabatic_index_minus_one * temp * temp / adiabatic_index_,
      1.0 / adiabatic_index_minus_one));

  auto velocity = make_with_value<tnsr::I<DataType, 3>>(x, 0.0);
  for (size_t i = 0; i < 3; ++i) {
    velocity.get(i) = gsl::at(mean_velocity_, i);
  }
  velocity.get(0) -= x_tilde.get(1) * temp;
  velocity.get(1) += x_tilde.get(0) * temp;
  velocity.get(2) += get(perturbation(x.get(2)));

  get<Tags::Velocity<DataType, 3>>(result) = std::move(velocity);

  get<Tags::SpecificInternalEnergy<DataType>>(result) =
      Scalar<DataType>(pow(get(get<Tags::MassDensity<DataType>>(result)),
                           adiabatic_index_minus_one) /
                       adiabatic_index_minus_one);

  return result;
}

template <typename DataType>
tuples::tagged_tuple_from_typelist<IsentropicVortex::conservative_t<DataType>>
IsentropicVortex::conservative_variables(const tnsr::I<DataType, 3>& x,
                                         const double t) const noexcept {
  const auto primitives = primitive_variables(x, t);

  auto result = make_with_value<tuples::tagged_tuple_from_typelist<
      IsentropicVortex::conservative_t<DataType>>>(x, 0.0);

  get<Tags::MassDensity<DataType>>(result) =
      get<Tags::MassDensity<DataType>>(primitives);

  conservative_from_primitive(
      make_not_null(&get<Tags::MomentumDensity<DataType, 3>>(result)),
      make_not_null(&get<Tags::EnergyDensity<DataType>>(result)),
      get<Tags::MassDensity<DataType>>(primitives),
      get<Tags::Velocity<DataType, 3>>(primitives),
      get<Tags::SpecificInternalEnergy<DataType>>(primitives));

  return result;
}

}  // namespace Solutions
}  // namespace NewtonianEuler

#define DTYPE(data) BOOST_PP_TUPLE_ELEM(0, data)

#define INSTANTIATE(_, data)                                                 \
  template Scalar<DTYPE(data)>                                               \
  NewtonianEuler::Solutions::IsentropicVortex::perturbation(                 \
      const DTYPE(data) & coord_z) const noexcept;                           \
  template tuples::tagged_tuple_from_typelist<                               \
      NewtonianEuler::Solutions::IsentropicVortex::primitive_t<DTYPE(data)>> \
  NewtonianEuler::Solutions::IsentropicVortex::primitive_variables(          \
      const tnsr::I<DTYPE(data), 3>& x, const double t) const noexcept;      \
  template tuples::tagged_tuple_from_typelist<                               \
      NewtonianEuler::Solutions::IsentropicVortex::conservative_t<DTYPE(     \
          data)>>                                                            \
  NewtonianEuler::Solutions::IsentropicVortex::conservative_variables(       \
      const tnsr::I<DTYPE(data), 3>& x, const double t) const noexcept;

GENERATE_INSTANTIATIONS(INSTANTIATE, (double, DataVector))

#undef DTYPE
#undef INSTANTIATE
/// \endcond
