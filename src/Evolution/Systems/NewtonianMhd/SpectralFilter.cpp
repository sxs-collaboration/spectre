// Distributed under the MIT License.
// See LICENSE.txt for details.

#include <cstddef>

#include "Evolution/DiscontinuousGalerkin/Initialization/SpectralFilters.tpp"
#include "Evolution/Systems/NewtonianMhd/Tags.hpp"
#include "NumericalAlgorithms/LinearOperators/Filters/FilledCylinder.tpp"
#include "NumericalAlgorithms/LinearOperators/Filters/HollowCylinder.tpp"
#include "NumericalAlgorithms/LinearOperators/Filters/Hypercube.tpp"
#include "NumericalAlgorithms/LinearOperators/Filters/None.tpp"
#include "Utilities/GenerateInstantiations.hpp"
#include "Utilities/TMPL.hpp"

namespace {
template <size_t Dim>
using newtonian_mhd_tags = tmpl::list<NewtonianMhd::Tags::MassDensityCons,
                                      NewtonianMhd::Tags::MomentumDensity<Dim>,
                                      NewtonianMhd::Tags::EnergyDensity>;
}  // namespace

#define DIM(data) BOOST_PP_TUPLE_ELEM(0, data)

#define INSTANTIATE(_, data)                                                   \
  template class Filters::Hypercube<DIM(data), newtonian_mhd_tags<DIM(data)>>; \
  template class Filters::None<DIM(data), newtonian_mhd_tags<DIM(data)>>;      \
  template struct evolution::dg::Initialization::SpectralFilters<              \
      DIM(data), newtonian_mhd_tags<DIM(data)>>;

GENERATE_INSTANTIATIONS(INSTANTIATE, (1, 2, 3))

template class Filters::HollowCylinder<newtonian_mhd_tags<3>>;
template class Filters::FilledCylinder<newtonian_mhd_tags<3>>;

#undef DIM
#undef INSTANTIATE
