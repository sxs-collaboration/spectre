// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Evolution/DiscontinuousGalerkin/MortarDataHolder.hpp"

#include <cstddef>
#include <ostream>
#include <pup.h>

#include "Utilities/GenerateInstantiations.hpp"

namespace evolution::dg {
template <size_t Dim>
void MortarDataHolder<Dim>::pup(PUP::er& p) {
  p | local_data_;
  p | neighbor_data_;
}

template <size_t Dim>
bool operator==(const MortarDataHolder<Dim>& lhs,
                const MortarDataHolder<Dim>& rhs) {
  return lhs.local() == rhs.local() and lhs.neighbor() == rhs.neighbor();
}

template <size_t Dim>
bool operator!=(const MortarDataHolder<Dim>& lhs,
                const MortarDataHolder<Dim>& rhs) {
  return not(lhs == rhs);
}

template <size_t Dim>
std::ostream& operator<<(std::ostream& os,
                         const MortarDataHolder<Dim>& mortar_data_holder) {
  os << "Local mortar data:\n" << mortar_data_holder.local() << "\n";
  os << "Neighbor mortar data:\n" << mortar_data_holder.neighbor() << "\n";
  return os;
}

#define DIM(data) BOOST_PP_TUPLE_ELEM(0, data)

#define INSTANTIATION(r, data)                                      \
  template class MortarDataHolder<DIM(data)>;                       \
  template bool operator==(const MortarDataHolder<DIM(data)>& lhs,  \
                           const MortarDataHolder<DIM(data)>& rhs); \
  template bool operator!=(const MortarDataHolder<DIM(data)>& lhs,  \
                           const MortarDataHolder<DIM(data)>& rhs); \
  template std::ostream& operator<<(                                \
      std::ostream& os,                                             \
      const MortarDataHolder<DIM(data)>& mortar_data_holder);

GENERATE_INSTANTIATIONS(INSTANTIATION, (1, 2, 3))

#undef INSTANTIATION
#undef DIM
}  // namespace evolution::dg
