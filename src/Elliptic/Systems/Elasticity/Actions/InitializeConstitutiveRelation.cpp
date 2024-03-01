// Distributed under the MIT License.
// See LICENSE.txt for details.

// Instantiations of domain::ExpandOverBlocks for Elasticity

#include <cstddef>
#include <memory>

#include "Domain/Creators/ExpandOverBlocks.tpp"
#include "PointwiseFunctions/Elasticity/ConstitutiveRelations/ConstitutiveRelation.hpp"

template <size_t Dim>
using ConstRelPtr = std::unique_ptr<
    Elasticity::ConstitutiveRelations::ConstitutiveRelation<Dim>>;

template class domain::ExpandOverBlocks<ConstRelPtr<2>>;
template class domain::ExpandOverBlocks<ConstRelPtr<3>>;
