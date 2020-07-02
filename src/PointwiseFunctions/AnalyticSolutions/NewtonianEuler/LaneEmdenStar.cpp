// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "PointwiseFunctions/AnalyticSolutions/NewtonianEuler/LaneEmdenStar.hpp"

#include <cmath>
#include <cstddef>

#include "DataStructures/DataVector.hpp"                  // IWYU pragma: keep
#include "DataStructures/Tensor/EagerMath/Magnitude.hpp"  // IWYU pragma: keep
#include "DataStructures/Tensor/TypeAliases.hpp"          // IWYU pragma: keep
#include "ErrorHandling/Assert.hpp"
#include "Utilities/ConstantExpressions.hpp"
#include "Utilities/ContainerHelpers.hpp"
#include "Utilities/GenerateInstantiations.hpp"
#include "Utilities/MakeWithValue.hpp"

// IWYU pragma: no_forward_declare Tensor

namespace NewtonianEuler::Solutions {

LaneEmdenStar::LaneEmdenStar(const double central_mass_density,
                             const double polytropic_constant) noexcept
    : central_mass_density_(central_mass_density),
      polytropic_constant_(polytropic_constant),
      equation_of_state_{polytropic_constant_, 2.0} {
  ASSERT(central_mass_density > 0.0,
         "central_mass_density = " << central_mass_density);
  ASSERT(polytropic_constant > 0.0,
         "polytropic_constant = " << polytropic_constant);
}

void LaneEmdenStar::pup(PUP::er& p) noexcept {
  p | central_mass_density_;
  p | polytropic_constant_;
  p | equation_of_state_;
  p | source_term_;
}

template <typename DataType>
tnsr::I<DataType, 3> LaneEmdenStar::gravitational_field(
    const tnsr::I<DataType, 3>& x) const noexcept {
  // Compute alpha for polytrope n==1, units G==1
  const double alpha = sqrt(0.5 * polytropic_constant_ / M_PI);
  const double outer_radius = alpha * M_PI;
  const double mass_scale = 4.0 * M_PI * cube(alpha) * central_mass_density_;
  // Add tiny offset to avoid divisons by zero
  const DataType radius = get(magnitude(x)) + 1.e-30 * outer_radius;

  auto enclosed_mass = make_with_value<DataType>(get_size(radius), mass_scale);
  for (size_t s = 0; s < get_size(radius); ++s) {
    if (get_element(radius, s) < outer_radius) {
      const double xi = get_element(radius, s) / alpha;
      get_element(enclosed_mass, s) *= sin(xi) - xi * cos(xi);
    } else {
      get_element(enclosed_mass, s) *= M_PI;
    }
  }

  auto gravitational_field_result = x;
  for (size_t i = 0; i < 3; ++i) {
    gravitational_field_result.get(i) *= -enclosed_mass / cube(radius);
  }
  return gravitational_field_result;
}

template <typename DataType>
Scalar<DataType> LaneEmdenStar::precompute_mass_density(
    const tnsr::I<DataType, 3>& x) const noexcept {
  // Compute alpha for polytrope n==1, units G==1
  const double alpha = sqrt(0.5 * polytropic_constant_ / M_PI);
  const double outer_radius = alpha * M_PI;
  // Add tiny offset to avoid divisons by zero
  const DataType radius = get(magnitude(x)) + 1.e-30 * outer_radius;

  Scalar<DataType> mass_density(get_size(radius));
  for (size_t s = 0; s < get_size(radius); ++s) {
    if (get_element(radius, s) < outer_radius) {
      const double xi = get_element(radius, s) / alpha;
      get_element(get(mass_density), s) = central_mass_density_ * sin(xi) / xi;
    } else {
      get_element(get(mass_density), s) = 0.0;
    }
  }
  return mass_density;
}

template <typename DataType>
tuples::TaggedTuple<Tags::MassDensity<DataType>> LaneEmdenStar::variables(
    tmpl::list<Tags::MassDensity<DataType>> /*meta*/,
    const Scalar<DataType>& mass_density) const noexcept {
  return mass_density;
}

template <typename DataType>
tuples::TaggedTuple<Tags::Velocity<DataType, 3>> LaneEmdenStar::variables(
    tmpl::list<Tags::Velocity<DataType, 3>> /*meta*/,
    const Scalar<DataType>& mass_density) const noexcept {
  return make_with_value<tnsr::I<DataType, 3>>(get(mass_density), 0.0);
}

template <typename DataType>
tuples::TaggedTuple<Tags::Pressure<DataType>> LaneEmdenStar::variables(
    tmpl::list<Tags::Pressure<DataType>> /*meta*/,
    const Scalar<DataType>& mass_density) const noexcept {
  return equation_of_state_.pressure_from_density(mass_density);
}

template <typename DataType>
tuples::TaggedTuple<Tags::SpecificInternalEnergy<DataType>>
LaneEmdenStar::variables(
    tmpl::list<Tags::SpecificInternalEnergy<DataType>> /*meta*/,
    const Scalar<DataType>& mass_density) const noexcept {
  return equation_of_state_.specific_internal_energy_from_density(mass_density);
}

bool operator==(const LaneEmdenStar& lhs, const LaneEmdenStar& rhs) noexcept {
  // There is no comparison operator for the EoS, but should be okay as
  // the `polytropic_constant`s are compared.
  // There is no comparison operator for the LaneEmdenGravitationalField source
  // term, but this should also be okay as it holds no data.
  return lhs.central_mass_density_ == rhs.central_mass_density_ and
         lhs.polytropic_constant_ == rhs.polytropic_constant_;
}

bool operator!=(const LaneEmdenStar& lhs, const LaneEmdenStar& rhs) noexcept {
  return not(lhs == rhs);
}

#define DTYPE(data) BOOST_PP_TUPLE_ELEM(0, data)

#define INSTANTIATE(_, data)                                                 \
  template tnsr::I<DTYPE(data), 3> LaneEmdenStar::gravitational_field(       \
      const tnsr::I<DTYPE(data), 3>& x) const noexcept;                      \
  template Scalar<DTYPE(data)> LaneEmdenStar::precompute_mass_density(       \
      const tnsr::I<DTYPE(data), 3>& x) const noexcept;                      \
  template tuples::TaggedTuple<Tags::MassDensity<DTYPE(data)>>               \
  LaneEmdenStar::variables(                                                  \
      tmpl::list<Tags::MassDensity<DTYPE(data)>> /*meta*/,                   \
      const Scalar<DTYPE(data)>& mass_density) const noexcept;               \
  template tuples::TaggedTuple<Tags::Velocity<DTYPE(data), 3>>               \
  LaneEmdenStar::variables(                                                  \
      tmpl::list<Tags::Velocity<DTYPE(data), 3>> /*meta*/,                   \
      const Scalar<DTYPE(data)>& mass_density) const noexcept;               \
  template tuples::TaggedTuple<Tags::Pressure<DTYPE(data)>>                  \
  LaneEmdenStar::variables(tmpl::list<Tags::Pressure<DTYPE(data)>> /*meta*/, \
                           const Scalar<DTYPE(data)>& mass_density)          \
      const noexcept;                                                        \
  template tuples::TaggedTuple<Tags::SpecificInternalEnergy<DTYPE(data)>>    \
  LaneEmdenStar::variables(                                                  \
      tmpl::list<Tags::SpecificInternalEnergy<DTYPE(data)>> /*meta*/,        \
      const Scalar<DTYPE(data)>& mass_density) const noexcept;

GENERATE_INSTANTIATIONS(INSTANTIATE, (double, DataVector))

#undef DTYPE
#undef INSTANTIATE

}  // namespace NewtonianEuler::Solutions
