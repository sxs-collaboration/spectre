// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "DataStructures/DataVector.hpp"
#include "Evolution/DiscontinuousGalerkin/Initialization/SpectralFilters.tpp"
#include "Evolution/Systems/CurvedScalarWave/Tags.hpp"
#include "Evolution/Systems/GeneralizedHarmonic/Tags.hpp"
#include "NumericalAlgorithms/LinearOperators/Filters/FilledCylinder.tpp"
#include "NumericalAlgorithms/LinearOperators/Filters/HollowCylinder.tpp"
#include "NumericalAlgorithms/LinearOperators/Filters/Hypercube.tpp"
#include "NumericalAlgorithms/LinearOperators/Filters/None.tpp"
#include "PointwiseFunctions/GeneralRelativity/Tags.hpp"
#include "Utilities/TMPL.hpp"

namespace {
using st_tags =
    tmpl::list<gr::Tags::SpacetimeMetric<DataVector, 3>,
               gh::Tags::Pi<DataVector, 3>, gh::Tags::Phi<DataVector, 3>,
               CurvedScalarWave::Tags::Psi, CurvedScalarWave::Tags::Pi,
               CurvedScalarWave::Tags::Phi<3>>;
}  // namespace

template class Filters::Hypercube<3, st_tags>;
template class Filters::None<3, st_tags>;
template class Filters::HollowCylinder<st_tags>;
template class Filters::FilledCylinder<st_tags>;
template struct evolution::dg::Initialization::SpectralFilters<3, st_tags>;
