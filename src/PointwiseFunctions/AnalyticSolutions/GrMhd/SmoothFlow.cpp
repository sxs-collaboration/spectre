// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "PointwiseFunctions/AnalyticSolutions/GrMhd/SmoothFlow.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <numeric>

#include "DataStructures/DataVector.hpp"                   // IWYU pragma: keep
#include "DataStructures/Tensor/Tensor.hpp"                // IWYU pragma: keep
#include "Parallel/PupStlCpp11.hpp"
#include "Utilities/GenerateInstantiations.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/MakeWithValue.hpp"

// IWYU pragma:  no_include "DataStructures/Tensor/TypeAliases.hpp"

/// \cond
namespace grmhd {
namespace Solutions {

SmoothFlow::SmoothFlow(MeanVelocity::type mean_velocity,
                       WaveVector::type wavevector,
                       const Pressure::type pressure,
                       const AdiabaticExponent::type adiabatic_exponent,
                       const PerturbationSize::type perturbation_size) noexcept
    :  // clang-tidy: do not std::move trivial types.
      mean_velocity_(std::move(mean_velocity)),  // NOLINT
      // clang-tidy: do not std::move trivial types.
      wavevector_(std::move(wavevector)),  // NOLINT
      pressure_(pressure),
      adiabatic_exponent_(adiabatic_exponent),
      perturbation_size_(perturbation_size),
      k_dot_v_(std::inner_product(mean_velocity_.begin(), mean_velocity_.end(),
                                  wavevector_.begin(), 0.0)),
      equation_of_state_{adiabatic_exponent_} {}

void SmoothFlow::pup(PUP::er& p) noexcept {
  p | mean_velocity_;
  p | wavevector_;
  p | pressure_;
  p | adiabatic_exponent_;
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

template <typename DataType>
tuples::tagged_tuple_from_typelist<SmoothFlow::variables_tags<DataType>>
SmoothFlow::variables(const tnsr::I<DataType, 3>& x, const double t,
                      SmoothFlow::variables_tags<DataType> /*meta*/) const
    noexcept {
  // Explicitly set all variables to zero:
  auto result = make_with_value<
      tuples::tagged_tuple_from_typelist<SmoothFlow::variables_tags<DataType>>>(
      x, 0.0);

  const DataType phase = k_dot_x_minus_vt(x, t);
  get(get<hydro::Tags::RestMassDensity<DataType>>(result)) =
      1.0 + perturbation_size_ * sin(phase);
  get(get<hydro::Tags::SpecificInternalEnergy<DataType>>(result)) =
      pressure_ / ((adiabatic_exponent_ - 1.0) *
                   get(get<hydro::Tags::RestMassDensity<DataType>>(result)));
  for (size_t i = 0; i < 3; ++i) {
    get<hydro::Tags::SpatialVelocity<DataType, 3, Frame::Inertial>>(result).get(
        i) = gsl::at(mean_velocity_, i);
  }

  // Magnetic field is not set because it is identically zero,
  // and `result` is initialized with all primitive variables equal to zero.

  get(get<hydro::Tags::Pressure<DataType>>(result)) = pressure_;
  return result;
}

#define DTYPE(data) BOOST_PP_TUPLE_ELEM(0, data)

#define INSTANTIATE(_, data)                                              \
  template tuples::tagged_tuple_from_typelist<                            \
      SmoothFlow::variables_tags<DTYPE(data)>>                            \
  SmoothFlow::variables(const tnsr::I<DTYPE(data), 3>& x, const double t, \
                        SmoothFlow::variables_tags<DTYPE(data)> /*meta*/) \
      const noexcept;

GENERATE_INSTANTIATIONS(INSTANTIATE, (double, DataVector))

#undef DTYPE
#undef INSTANTIATE
}  // namespace Solutions
}  // namespace grmhd
/// \endcond
