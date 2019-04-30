// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Domain/Element.hpp"

#include <ostream>
#include <pup.h>  // IWYU pragma: keep

#include "Domain/MaxNumberOfNeighbors.hpp"
#include "ErrorHandling/Assert.hpp"
#include "Parallel/PupStlCpp11.hpp"  // IWYU pragma: keep
#include "Utilities/GenerateInstantiations.hpp"
#include "Utilities/StdHelpers.hpp"  // IWYU pragma: keep

template <size_t VolumeDim>
Element<VolumeDim>::Element(ElementId<VolumeDim> id,
                            Neighbors_t neighbors) noexcept
    : id_(std::move(id)),
      neighbors_(std::move(neighbors)),
      number_of_neighbors_([this](){
        size_t number_of_neighbors = 0;
        for (const auto& p : neighbors_) {
          number_of_neighbors += p.second.size();
        }
        return number_of_neighbors;
      }()),
      external_boundaries_([this](){
        std::unordered_set<Direction<VolumeDim>> external_boundaries(
            Direction<VolumeDim>::all_directions().begin(),
            Direction<VolumeDim>::all_directions().end());
        for (const auto& neighbor_direction : neighbors_) {
          external_boundaries.erase(neighbor_direction.first);
        }
        return external_boundaries;
      }()) {
  // Assuming a maximum 2-to-1 refinement between neighboring elements:
  ASSERT(number_of_neighbors_ <= maximum_number_of_neighbors(VolumeDim),
         "Can't have " << number_of_neighbors_ << " neighbors in " << VolumeDim
                       << " dimensions");
}

template <size_t VolumeDim>
void Element<VolumeDim>::pup(PUP::er& p) noexcept {
  p | id_;
  p | neighbors_;
  p | number_of_neighbors_;
  p | external_boundaries_;
}

template <size_t VolumeDim>
bool operator==(const Element<VolumeDim>& lhs,
                const Element<VolumeDim>& rhs) noexcept {
  return lhs.id() == rhs.id() and
         lhs.neighbors() == rhs.neighbors() and
         lhs.number_of_neighbors() == rhs.number_of_neighbors() and
         lhs.external_boundaries() == rhs.external_boundaries();
}

template <size_t VolumeDim>
bool operator!=(const Element<VolumeDim>& lhs,
                const Element<VolumeDim>& rhs) noexcept {
  return not(lhs == rhs);
}

template <size_t VolumeDim>
std::ostream& operator<<(std::ostream& os,
                         const Element<VolumeDim>& element) noexcept {
  os << "Element " << element.id() << ":\n";
  os << "  Neighbors: " << element.neighbors() << "\n";
  os << "  External boundaries: " << element.external_boundaries() << "\n";
  return os;
}

#define GET_DIM(data) BOOST_PP_TUPLE_ELEM(0, data)

#define INSTANTIATION(r, data)                                      \
  template class Element<GET_DIM(data)>;                            \
  template bool operator==(const Element<GET_DIM(data)>&,           \
                           const Element<GET_DIM(data)>&) noexcept; \
  template bool operator!=(const Element<GET_DIM(data)>&,           \
                           const Element<GET_DIM(data)>&) noexcept; \
  template std::ostream& operator<<(std::ostream&,                  \
                                    const Element<GET_DIM(data)>&) noexcept;

GENERATE_INSTANTIATIONS(INSTANTIATION, (1, 2, 3))

#undef GET_DIM
#undef INSTANTIATION
