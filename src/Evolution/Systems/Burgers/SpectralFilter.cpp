// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Evolution/DiscontinuousGalerkin/Initialization/SpectralFilters.tpp"
#include "Evolution/Systems/Burgers/Tags.hpp"
#include "NumericalAlgorithms/LinearOperators/Filters/Hypercube.tpp"
#include "NumericalAlgorithms/LinearOperators/Filters/None.tpp"
#include "Utilities/TMPL.hpp"

namespace {
using burgers_tags = tmpl::list<Burgers::Tags::U>;
}  // namespace

template class Filters::Hypercube<1, burgers_tags>;
template class Filters::None<1, burgers_tags>;
template struct evolution::dg::Initialization::SpectralFilters<1, burgers_tags>;
