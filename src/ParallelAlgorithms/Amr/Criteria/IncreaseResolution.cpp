// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "ParallelAlgorithms/Amr/Criteria/IncreaseResolution.hpp"

namespace amr::Criteria {

template class IncreaseResolution<1>;
template class IncreaseResolution<2>;
template class IncreaseResolution<3>;

}  // namespace amr::Criteria
