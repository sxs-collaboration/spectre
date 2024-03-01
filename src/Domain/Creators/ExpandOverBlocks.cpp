// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Domain/Creators/ExpandOverBlocks.tpp"

#include <array>
#include <cstddef>

namespace domain {

template class ExpandOverBlocks<std::array<size_t, 1>>;
template class ExpandOverBlocks<std::array<size_t, 2>>;
template class ExpandOverBlocks<std::array<size_t, 3>>;

}  // namespace domain
