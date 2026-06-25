// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "DataStructures/Tensor/IndexType.hpp"
#include "Evolution/DiscontinuousGalerkin/Initialization/SpectralFilters.tpp"
#include "Evolution/Systems/GrMhd/ValenciaDivClean/Tags.hpp"
#include "NumericalAlgorithms/LinearOperators/Filters/HollowCylinder.tpp"
#include "NumericalAlgorithms/LinearOperators/Filters/Hypercube.tpp"
#include "NumericalAlgorithms/LinearOperators/Filters/None.tpp"
#include "Utilities/TMPL.hpp"

namespace {
using valencia_tags =
    tmpl::list<grmhd::ValenciaDivClean::Tags::TildeD,
               grmhd::ValenciaDivClean::Tags::TildeYe,
               grmhd::ValenciaDivClean::Tags::TildeTau,
               grmhd::ValenciaDivClean::Tags::TildeS<Frame::Inertial>,
               grmhd::ValenciaDivClean::Tags::TildeB<Frame::Inertial>,
               grmhd::ValenciaDivClean::Tags::TildePhi>;
}  // namespace

template class Filters::Hypercube<3, valencia_tags>;
template class Filters::None<3, valencia_tags>;
template class Filters::HollowCylinder<valencia_tags>;
template struct evolution::dg::Initialization::SpectralFilters<3,
                                                               valencia_tags>;
