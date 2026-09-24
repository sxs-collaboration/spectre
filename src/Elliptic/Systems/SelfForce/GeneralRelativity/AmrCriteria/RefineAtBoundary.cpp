// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Elliptic/Systems/SelfForce/GeneralRelativity/AmrCriteria/RefineAtBoundary.hpp"

namespace GrSelfForce::AmrCriteria {

// template <size_t Dim, size_t DimToRefine>
// RefineAtBoundary<Dim, DimToRefine>::RefineAtBoundary(CkMigrateMessage* msg)
//     : Criterion(msg) {}

template class RefineAtBoundary<2, 0>;
template class RefineAtBoundary<2, 1>;

}  // namespace GrSelfForce::AmrCriteria
