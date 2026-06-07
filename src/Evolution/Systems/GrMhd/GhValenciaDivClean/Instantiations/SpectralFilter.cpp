// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Evolution/DiscontinuousGalerkin/Initialization/SpectralFilters.tpp"
#include "Evolution/Systems/GrMhd/GhValenciaDivClean/System.hpp"
#include "Evolution/Systems/RadiationTransport/NoNeutrinos/System.hpp"
#include "NumericalAlgorithms/LinearOperators/Filters/Hypercube.tpp"
#include "NumericalAlgorithms/LinearOperators/Filters/None.tpp"
#include "NumericalAlgorithms/LinearOperators/Filters/SphericalShell.tpp"

namespace {
using ghmhd_system =
    grmhd::GhValenciaDivClean::System<RadiationTransport::NoNeutrinos::System>;
using ghmhd_tags = typename ghmhd_system::variables_tag::tags_list;
}  // namespace

template class Filters::Hypercube<3, ghmhd_tags>;
template class Filters::None<3, ghmhd_tags>;
template class Filters::SphericalShell<ghmhd_tags>;
template struct evolution::dg::Initialization::SpectralFilters<3, ghmhd_tags>;
