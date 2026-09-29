// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Evolution/DiscontinuousGalerkin/Initialization/SpectralFilters.tpp"
#include "Evolution/Systems/SecondOrderScalarWave/Tags.hpp"
#include "NumericalAlgorithms/LinearOperators/Filters/FilledCylinder.tpp"
#include "NumericalAlgorithms/LinearOperators/Filters/HollowCylinder.tpp"
#include "NumericalAlgorithms/LinearOperators/Filters/Hypercube.tpp"
#include "NumericalAlgorithms/LinearOperators/Filters/None.tpp"
#include "Utilities/GenerateInstantiations.hpp"
#include "Utilities/TMPL.hpp"  // IWYU pragma: keep

namespace {
using tags_for_filter = tmpl::list<SecondOrderScalarWave::Tags::Psi,
                                   SecondOrderScalarWave::Tags::Pi>;
}  // namespace

#define DIM(data) BOOST_PP_TUPLE_ELEM(0, data)

#define INSTANTIATE(_, data)                                      \
  template class Filters::Hypercube<DIM(data), tags_for_filter>;  \
  template class Filters::None<DIM(data), tags_for_filter>;       \
  template struct evolution::dg::Initialization::SpectralFilters< \
      DIM(data), tags_for_filter>;

GENERATE_INSTANTIATIONS(INSTANTIATE, (1, 2, 3))

template class Filters::HollowCylinder<tags_for_filter>;
template class Filters::FilledCylinder<tags_for_filter>;

#undef DIM
#undef INSTANTIATE
