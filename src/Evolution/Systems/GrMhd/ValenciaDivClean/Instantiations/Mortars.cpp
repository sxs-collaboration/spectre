// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Evolution/DiscontinuousGalerkin/CleanMortarHistory.tpp"
#include "Evolution/Systems/GrMhd/ValenciaDivClean/System.hpp"

template class evolution::dg::CleanMortarHistory<
    grmhd::ValenciaDivClean::System>;
