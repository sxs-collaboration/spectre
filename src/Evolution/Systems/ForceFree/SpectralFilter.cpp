// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Evolution/DiscontinuousGalerkin/Initialization/SpectralFilters.tpp"
#include "Evolution/Systems/ForceFree/Tags.hpp"
#include "NumericalAlgorithms/LinearOperators/Filters/FilledCylinder.tpp"
#include "NumericalAlgorithms/LinearOperators/Filters/HollowCylinder.tpp"
#include "NumericalAlgorithms/LinearOperators/Filters/Hypercube.tpp"
#include "NumericalAlgorithms/LinearOperators/Filters/None.tpp"
#include "Utilities/TMPL.hpp"

namespace {
using ff_tags = tmpl::list<ForceFree::Tags::TildeE, ForceFree::Tags::TildeB,
                           ForceFree::Tags::TildePsi, ForceFree::Tags::TildePhi,
                           ForceFree::Tags::TildeQ>;
}  // namespace

template class Filters::Hypercube<3, ff_tags>;
template class Filters::None<3, ff_tags>;
template class Filters::HollowCylinder<ff_tags>;
template class Filters::FilledCylinder<ff_tags>;
template struct evolution::dg::Initialization::SpectralFilters<3, ff_tags>;
