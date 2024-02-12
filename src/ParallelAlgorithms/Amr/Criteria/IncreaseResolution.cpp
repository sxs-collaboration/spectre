// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "ParallelAlgorithms/Amr/Criteria/IncreaseResolution.hpp"

namespace amr::Criteria {

template <size_t Dim>
IncreaseResolution<Dim>::IncreaseResolution(CkMigrateMessage* msg)
    : Criterion(msg) {}

template class IncreaseResolution<1>;
template class IncreaseResolution<2>;
template class IncreaseResolution<3>;

}  // namespace amr::Criteria
