// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Domain/Structure/DirectionalId.hpp"

#include <boost/functional/hash.hpp>
#include <ostream>
#include <pup.h>

#include "Utilities/GenerateInstantiations.hpp"

static_assert(sizeof(DirectionalId<1>) == sizeof(ElementId<1>));
static_assert(sizeof(DirectionalId<2>) == sizeof(ElementId<2>));
static_assert(sizeof(DirectionalId<3>) == sizeof(ElementId<3>));

template <size_t VolumeDim>
size_t hash_value(const DirectionalId<VolumeDim>& id) {
  return hash_value(id.id()) bitor static_cast<uint64_t>(id.direction().bits())
                                       << ElementId<VolumeDim>::direction_shift;
}

// NOLINTNEXTLINE(cert-dcl58-cpp)
namespace std {
template <size_t VolumeDim>
size_t hash<DirectionalId<VolumeDim>>::operator()(
    const DirectionalId<VolumeDim>& id) const {
  return boost::hash<DirectionalId<VolumeDim>>{}(id);
}
}  // namespace std

template <size_t VolumeDim>
std::ostream& operator<<(std::ostream& os,
                         const DirectionalId<VolumeDim>& direction_id) {
  os << "(" << direction_id.direction() << ", " << direction_id.id() << ")";
  return os;
}

template <size_t VolumeDim>
bool operator==(const DirectionalId<VolumeDim>& lhs,
                const DirectionalId<VolumeDim>& rhs) {
  return lhs.direction() == rhs.direction() and lhs.id() == rhs.id();
}

template <size_t VolumeDim>
bool operator<(const DirectionalId<VolumeDim>& lhs,
               const DirectionalId<VolumeDim>& rhs) {
  if (lhs.direction() != rhs.direction()) {
    return lhs.direction() < rhs.direction();
  }
  return lhs.id() < rhs.id();
}

#define GET_DIM(data) BOOST_PP_TUPLE_ELEM(0, data)

#define INSTANTIATION(r, data)                                            \
  template std::ostream& operator<<(std::ostream&,                        \
                                    const DirectionalId<GET_DIM(data)>&); \
  template bool operator==(const DirectionalId<GET_DIM(data)>& lhs,       \
                           const DirectionalId<GET_DIM(data)>& rhs);      \
  template bool operator<(const DirectionalId<GET_DIM(data)>& lhs,        \
                          const DirectionalId<GET_DIM(data)>& rhs);       \
  template size_t hash_value(const DirectionalId<GET_DIM(data)>&);        \
  namespace std { /* NOLINT */                                            \
  template struct hash<DirectionalId<GET_DIM(data)>>;                     \
  }

GENERATE_INSTANTIATIONS(INSTANTIATION, (1, 2, 3))

#undef GET_DIM
#undef INSTANTIATION
