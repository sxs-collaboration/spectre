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
    : RelativisticEuler::Solutions::SmoothFlow<3>(mean_velocity, wavevector,
                                                  pressure, adiabatic_index,
                                                  perturbation_size) {}

void SmoothFlow::pup(PUP::er& p) noexcept {
  RelativisticEuler::Solutions::SmoothFlow<3>::pup(p);
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

bool operator==(const SmoothFlow& lhs, const SmoothFlow& rhs) noexcept {
  using smooth_flow = RelativisticEuler::Solutions::SmoothFlow<3>;
  return *static_cast<const smooth_flow*>(&lhs) ==
         *static_cast<const smooth_flow*>(&rhs);
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

GENERATE_INSTANTIATIONS(INSTANTIATE_SCALARS, (double, DataVector),
                        (hydro::Tags::DivergenceCleaningField))

#define INSTANTIATE_VECTORS(_, data)                                         \
  template tuples::TaggedTuple<TAG(data) < DTYPE(data), 3, Frame::Inertial>> \
      SmoothFlow::variables(                                                 \
          const tnsr::I<DTYPE(data), 3>& x, double t,                        \
          tmpl::list<TAG(data) < DTYPE(data), 3, Frame::Inertial>> /*meta*/) \
          const noexcept;

GENERATE_INSTANTIATIONS(INSTANTIATE_VECTORS, (double, DataVector),
                        (hydro::Tags::MagneticField))

#undef DTYPE
#undef TAG
#undef INSTANTIATE_SCALARS
#undef INSTANTIATE_VECTORS
}  // namespace Solutions
}  // namespace grmhd
/// \endcond
