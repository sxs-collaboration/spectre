// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "ParallelAlgorithms/Amr/Criteria/RefineAtBoundary.hpp"

namespace amr::Criteria {

template class RefineAtBoundary<1, 0>;
template class RefineAtBoundary<2, 0>;
template class RefineAtBoundary<2, 1>;
template class RefineAtBoundary<3, 0>;
template class RefineAtBoundary<3, 1>;
template class RefineAtBoundary<3, 2>;

}  // namespace amr::Criteria
