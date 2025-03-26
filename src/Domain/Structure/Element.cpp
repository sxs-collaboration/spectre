// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Domain/Structure/Element.hpp"

#include <array>
#include <ostream>
#include <pup.h>

#include "Domain/Structure/HasBoundary.hpp"
#include "Domain/Structure/MaxNumberOfNeighbors.hpp"
#include "Domain/Structure/Topology.hpp"
#include "Utilities/ErrorHandling/Assert.hpp"
#include "Utilities/GenerateInstantiations.hpp"
#include "Utilities/StdHelpers.hpp"

template <size_t VolumeDim>
Element<VolumeDim>::Element(ElementId<VolumeDim> id, Neighbors_t neighbors,
                            std::array<domain::Topology, VolumeDim> topologies)
    : id_(std::move(id)),
      neighbors_(std::move(neighbors)),
      number_of_neighbors_([this]() {
        size_t number_of_neighbors = 0;
        for (const auto& p : neighbors_) {
          number_of_neighbors += p.second.size();
        }
        return number_of_neighbors;
      }()),
      topologies_(std::move(topologies)) {
  for (const auto& direction : Direction<VolumeDim>::all_directions()) {
    const bool has_boundary_in_this_direction = has_boundary(
        gsl::at(topologies_, direction.dimension()), direction.side());
    if (neighbors_.find(direction) == neighbors_.end()) {
      if (has_boundary_in_this_direction) {
        external_boundaries_.emplace(direction);
      }
    } else {
      ASSERT(has_boundary_in_this_direction,
             "Cannot specify a neighbor in a direction with no boundary.  "
             "Topologies: "
                 << topologies_ << "; Direction: " << direction);
      internal_boundaries_.emplace(direction);
    }
  }
  // Assuming a maximum 2-to-1 refinement between neighboring elements:
  ASSERT(number_of_neighbors_ <= maximum_number_of_neighbors(VolumeDim),
         "Can't have " << number_of_neighbors_ << " neighbors in " << VolumeDim
                       << " dimensions");
}

template <size_t VolumeDim>
void Element<VolumeDim>::pup(PUP::er& p) {
  p | id_;
  p | neighbors_;
  p | number_of_neighbors_;
  p | external_boundaries_;
  p | internal_boundaries_;
  p | topologies_;
}

template <size_t VolumeDim>
bool operator==(const Element<VolumeDim>& lhs, const Element<VolumeDim>& rhs) {
  return lhs.id() == rhs.id() and lhs.neighbors() == rhs.neighbors() and
         lhs.number_of_neighbors() == rhs.number_of_neighbors() and
         lhs.external_boundaries() == rhs.external_boundaries() and
         lhs.internal_boundaries() == rhs.internal_boundaries() and
         lhs.topologies() == rhs.topologies();
}

template <size_t VolumeDim>
bool operator!=(const Element<VolumeDim>& lhs, const Element<VolumeDim>& rhs) {
  return not(lhs == rhs);
}

template <size_t VolumeDim>
std::ostream& operator<<(std::ostream& os, const Element<VolumeDim>& element) {
  os << "Element " << element.id() << ":\n";
  os << "  Topology: " << element.topologies() << '\n';
  os << "  Neighbors: " << element.neighbors() << "\n";
  os << "  External boundaries: " << element.external_boundaries() << "\n";
  return os;
}

#define GET_DIM(data) BOOST_PP_TUPLE_ELEM(0, data)

#define INSTANTIATION(r, data)                             \
  template class Element<GET_DIM(data)>;                   \
  template bool operator==(const Element<GET_DIM(data)>&,  \
                           const Element<GET_DIM(data)>&); \
  template bool operator!=(const Element<GET_DIM(data)>&,  \
                           const Element<GET_DIM(data)>&); \
  template std::ostream& operator<<(std::ostream&,         \
                                    const Element<GET_DIM(data)>&);

GENERATE_INSTANTIATIONS(INSTANTIATION, (1, 2, 3))

#undef GET_DIM
#undef INSTANTIATION
