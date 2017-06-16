// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "DataStructures/Index.hpp"

/// \cond HIDDEN_SYMBOLS
template <size_t Dim>
bool operator==(const Index<Dim>& lhs, const Index<Dim>& rhs) {
  return std::equal(lhs.begin(), lhs.end(), rhs.begin());
}

template <size_t Dim>
bool operator!=(const Index<Dim>& lhs, const Index<Dim>& rhs) {
  return not(lhs == rhs);
}

template class Index<0>;
template class Index<1>;
template class Index<2>;
template class Index<3>;

template bool operator==(const Index<0>& lhs, const Index<0>& rhs);
template bool operator==(const Index<1>& lhs, const Index<1>& rhs);
template bool operator==(const Index<2>& lhs, const Index<2>& rhs);
template bool operator==(const Index<3>& lhs, const Index<3>& rhs);

template bool operator!=(const Index<0>& lhs, const Index<0>& rhs);
template bool operator!=(const Index<1>& lhs, const Index<1>& rhs);
template bool operator!=(const Index<2>& lhs, const Index<2>& rhs);
template bool operator!=(const Index<3>& lhs, const Index<3>& rhs);
/// \endcond
