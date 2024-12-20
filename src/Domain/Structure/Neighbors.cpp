// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Domain/Structure/Neighbors.hpp"

#include <ostream>
#include <pup.h>
#include <pup_stl.h>

#include "Domain/Structure/ElementId.hpp"
#include "Domain/Structure/MaxNumberOfNeighbors.hpp"
#include "Utilities/ErrorHandling/Assert.hpp"
#include "Utilities/GenerateInstantiations.hpp"
#include "Utilities/StdHelpers.hpp"

template <size_t VolumeDim>
Neighbors<VolumeDim>::Neighbors(std::unordered_set<ElementId<VolumeDim>> ids,
                                OrientationMap<VolumeDim> orientation,
                                domain::BlockGeometry geometry)
    : ids_(std::move(ids)),
      orientation_(std::move(orientation)),
      geometry_(std::move(geometry)) {
  ASSERT(orientation_ != OrientationMap<VolumeDim>{},
         "Cannot use a default-constructed OrientationMap in Neighbors.");
  if (geometry_ == domain::BlockGeometry::SphericalShell) {
    ASSERT(ids_.size() == 1, "Only one spherical shell neighbor is possible.");
  }
}

template <size_t VolumeDim>
void Neighbors<VolumeDim>::add_ids(
    const std::unordered_set<ElementId<VolumeDim>>& additional_ids) {
  for (const auto& id : additional_ids) {
    ids_.insert(id);
  }
  if (geometry_ == domain::BlockGeometry::SphericalShell) {
    ASSERT(ids_.size() == 1, "Only one spherical shell neighbor is possible.");
  }
}

template <size_t VolumeDim>
std::ostream& operator<<(std::ostream& os, const Neighbors<VolumeDim>& n) {
  os << "Ids = " << n.ids() << "; orientation = " << n.orientation();
  return os;
}

template <size_t VolumeDim>
bool operator==(const Neighbors<VolumeDim>& lhs,
                const Neighbors<VolumeDim>& rhs) {
  return (lhs.ids() == rhs.ids() and lhs.orientation() == rhs.orientation() and
          lhs.geometry() == rhs.geometry());
}

template <size_t VolumeDim>
bool operator!=(const Neighbors<VolumeDim>& lhs,
                const Neighbors<VolumeDim>& rhs) {
  return not(lhs == rhs);
}

template <size_t VolumeDim>
void Neighbors<VolumeDim>::pup(PUP::er& p) {
  p | ids_;
  p | orientation_;
  p | geometry_;
}

#define GET_DIM(data) BOOST_PP_TUPLE_ELEM(0, data)

#define INSTANTIATION(r, data)                                              \
  template class Neighbors<GET_DIM(data)>;                                  \
  template std::ostream& operator<<(std::ostream& os,                       \
                                    const Neighbors<GET_DIM(data)>& block); \
  template bool operator==(const Neighbors<GET_DIM(data)>& lhs,             \
                           const Neighbors<GET_DIM(data)>& rhs);            \
  template bool operator!=(const Neighbors<GET_DIM(data)>& lhs,             \
                           const Neighbors<GET_DIM(data)>& rhs);

GENERATE_INSTANTIATIONS(INSTANTIATION, (1, 2, 3))

#undef GET_DIM
#undef INSTANTIATION
