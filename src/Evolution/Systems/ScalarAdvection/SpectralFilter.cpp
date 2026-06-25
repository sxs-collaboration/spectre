// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Evolution/DiscontinuousGalerkin/Initialization/SpectralFilters.tpp"
#include "Evolution/Systems/ScalarAdvection/Tags.hpp"
#include "NumericalAlgorithms/LinearOperators/Filters/HollowCylinder.tpp"
#include "NumericalAlgorithms/LinearOperators/Filters/Hypercube.tpp"
#include "NumericalAlgorithms/LinearOperators/Filters/None.tpp"
#include "Utilities/GenerateInstantiations.hpp"
#include "Utilities/TMPL.hpp"

namespace {
using scalar_advection_tags = tmpl::list<ScalarAdvection::Tags::U>;
}  // namespace

#define DIM(data) BOOST_PP_TUPLE_ELEM(0, data)

#define INSTANTIATE(_, data)                                           \
  template class Filters::Hypercube<DIM(data), scalar_advection_tags>; \
  template class Filters::None<DIM(data), scalar_advection_tags>;      \
  template struct evolution::dg::Initialization::SpectralFilters<      \
      DIM(data), scalar_advection_tags>;

GENERATE_INSTANTIATIONS(INSTANTIATE, (1, 2, 3))

template class Filters::HollowCylinder<scalar_advection_tags>;

#undef DIM
#undef INSTANTIATE
