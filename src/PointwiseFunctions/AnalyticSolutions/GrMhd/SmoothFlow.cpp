// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "PointwiseFunctions/AnalyticSolutions/GrMhd/SmoothFlow.hpp"

#include "DataStructures/DataVector.hpp"     // IWYU pragma: keep
#include "DataStructures/Tensor/Tensor.hpp"  // IWYU pragma: keep
#include "PointwiseFunctions/Hydro/Tags.hpp"
#include "Utilities/GenerateInstantiations.hpp"
#include "Utilities/MakeWithValue.hpp"

// IWYU pragma:  no_include "DataStructures/Tensor/TypeAliases.hpp"

namespace grmhd::Solutions {

SmoothFlow::SmoothFlow(const std::array<double, 3>& mean_velocity,
                       const std::array<double, 3>& wavevector,
                       const double pressure, const double adiabatic_index,
                       const double perturbation_size)
    : RelativisticEuler::Solutions::SmoothFlow<3>(mean_velocity, wavevector,
                                                  pressure, adiabatic_index,
                                                  perturbation_size) {}

void SmoothFlow::pup(PUP::er& p) {
  RelativisticEuler::Solutions::SmoothFlow<3>::pup(p);
}

template <typename DataType>
tuples::TaggedTuple<hydro::Tags::MagneticField<DataType, 3>>
SmoothFlow::variables(
    const tnsr::I<DataType, 3>& x, double /*t*/,
    tmpl::list<hydro::Tags::MagneticField<DataType, 3>> /*meta*/) const {
  return {make_with_value<tnsr::I<DataType, 3>>(x, 0.0)};
}

template <typename DataType>
tuples::TaggedTuple<hydro::Tags::DivergenceCleaningField<DataType>>
SmoothFlow::variables(
    const tnsr::I<DataType, 3>& x, double /*t*/,
    tmpl::list<hydro::Tags::DivergenceCleaningField<DataType>> /*meta*/) const {
  return {make_with_value<Scalar<DataType>>(x, 0.0)};
}

bool operator==(const SmoothFlow& lhs, const SmoothFlow& rhs) {
  using smooth_flow = RelativisticEuler::Solutions::SmoothFlow<3>;
  return *static_cast<const smooth_flow*>(&lhs) ==
         *static_cast<const smooth_flow*>(&rhs);
}

bool operator!=(const SmoothFlow& lhs, const SmoothFlow& rhs) {
  return not(lhs == rhs);
}

#define DTYPE(data) BOOST_PP_TUPLE_ELEM(0, data)
#define TAG(data) BOOST_PP_TUPLE_ELEM(1, data)

#define INSTANTIATE_SCALARS(_, data)                                        \
  template tuples::TaggedTuple<TAG(data) < DTYPE(data)> >                   \
      SmoothFlow::variables(const tnsr::I<DTYPE(data), 3>& x, double t,     \
                            tmpl::list<TAG(data) < DTYPE(data)> > /*meta*/) \
          const;

GENERATE_INSTANTIATIONS(INSTANTIATE_SCALARS, (double, DataVector),
                        (hydro::Tags::DivergenceCleaningField))

#define INSTANTIATE_VECTORS(_, data)                                           \
  template tuples::TaggedTuple<TAG(data) < DTYPE(data), 3> >                   \
      SmoothFlow::variables(const tnsr::I<DTYPE(data), 3>& x, double t,        \
                            tmpl::list<TAG(data) < DTYPE(data), 3> > /*meta*/) \
          const;

GENERATE_INSTANTIATIONS(INSTANTIATE_VECTORS, (double, DataVector),
                        (hydro::Tags::MagneticField))

#undef DTYPE
#undef TAG
#undef INSTANTIATE_SCALARS
#undef INSTANTIATE_VECTORS
}  // namespace grmhd::Solutions
