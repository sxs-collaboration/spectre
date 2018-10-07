// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Domain/OrientationMap.hpp"

#include <ostream>
#include <set>

#include "Domain/SegmentId.hpp"  // IWYU pragma: keep
#include "ErrorHandling/Assert.hpp"
#include "Parallel/PupStlCpp11.hpp"  // IWYU pragma: keep

namespace {
template <size_t VolumeDim>
std::set<size_t> set_of_dimensions(
    const std::array<Direction<VolumeDim>, VolumeDim>& directions) noexcept {
  std::set<size_t> set_of_dims;
  for (size_t j = 0; j < VolumeDim; j++) {
    set_of_dims.insert(gsl::at(directions, j).dimension());
  }
  return set_of_dims;
}
}  // namespace

template <size_t VolumeDim>
OrientationMap<VolumeDim>::OrientationMap() noexcept {
  for (size_t j = 0; j < VolumeDim; j++) {
    gsl::at(mapped_directions_, j) = Direction<VolumeDim>(j, Side::Upper);
  }
}

template <size_t VolumeDim>
OrientationMap<VolumeDim>::OrientationMap(
    std::array<Direction<VolumeDim>, VolumeDim> mapped_directions) noexcept
    : mapped_directions_(std::move(mapped_directions)) {
  for (size_t j = 0; j < VolumeDim; j++) {
    if (gsl::at(mapped_directions_, j).dimension() != j or
        gsl::at(mapped_directions_, j).side() != Side::Upper) {
      is_aligned_ = false;
    }
  }
  ASSERT(set_of_dimensions(mapped_directions).size() == VolumeDim,
         "This OrientationMap fails to map Directions one-to-one.");
}

template <size_t VolumeDim>
OrientationMap<VolumeDim>::OrientationMap(
    const std::array<Direction<VolumeDim>, VolumeDim>& directions_in_host,
    const std::array<Direction<VolumeDim>, VolumeDim>&
        directions_in_neighbor) noexcept {
  for (size_t j = 0; j < VolumeDim; j++) {
    gsl::at(mapped_directions_, gsl::at(directions_in_host, j).dimension()) =
        (gsl::at(directions_in_host, j).side() == Side::Upper
             ? gsl::at(directions_in_neighbor, j)
             : gsl::at(directions_in_neighbor, j).opposite());
    if (gsl::at(directions_in_host, j) != gsl::at(directions_in_neighbor, j)) {
      is_aligned_ = false;
    }
  }
  ASSERT(set_of_dimensions(directions_in_host).size() == VolumeDim,
         "This OrientationMap fails to map Directions one-to-one.");
  ASSERT(set_of_dimensions(directions_in_neighbor).size() == VolumeDim,
         "This OrientationMap fails to map Directions one-to-one.");
}

template <size_t VolumeDim>
std::array<SegmentId, VolumeDim> OrientationMap<VolumeDim>::operator()(
    const std::array<SegmentId, VolumeDim>& segmentIds) const noexcept {
  std::array<SegmentId, VolumeDim> result = segmentIds;
  for (size_t d = 0; d < VolumeDim; d++) {
    gsl::at(result, gsl::at(mapped_directions_, d).dimension()) =
        gsl::at(mapped_directions_, d).side() == Side::Upper
            ? gsl::at(segmentIds, d)
            : gsl::at(segmentIds, d).id_if_flipped();
  }
  return result;
}

template <size_t VolumeDim>
Mesh<VolumeDim> OrientationMap<VolumeDim>::operator()(
    const Mesh<VolumeDim>& mesh) const noexcept {
  return Mesh<VolumeDim>(this->permute_from_neighbor(mesh.extents().indices()),
                         this->permute_from_neighbor(mesh.basis()),
                         this->permute_from_neighbor(mesh.quadrature()));
}

template <size_t VolumeDim>
OrientationMap<VolumeDim> OrientationMap<VolumeDim>::inverse_map() const
    noexcept {
  std::array<Direction<VolumeDim>, VolumeDim> result;
  for (size_t i = 0; i < VolumeDim; i++) {
    gsl::at(result, gsl::at(mapped_directions_, i).dimension()) =
        Direction<VolumeDim>(i, gsl::at(mapped_directions_, i).side());
  }
  return OrientationMap<VolumeDim>{result};
}

template <size_t VolumeDim>
void OrientationMap<VolumeDim>::pup(PUP::er& p) noexcept {
  p | mapped_directions_;
  p | is_aligned_;
}

template <>
std::ostream& operator<<(std::ostream& os,
                         const OrientationMap<1>& orientation) noexcept {
  os << "(" << orientation(Direction<1>::upper_xi()) << ")";
  return os;
}

template <>
std::ostream& operator<<(std::ostream& os,
                         const OrientationMap<2>& orientation) noexcept {
  os << "(" << orientation(Direction<2>::upper_xi()) << ", "
     << orientation(Direction<2>::upper_eta()) << ")";
  return os;
}

template <>
std::ostream& operator<<(std::ostream& os,
                         const OrientationMap<3>& orientation) noexcept {
  os << "(" << orientation(Direction<3>::upper_xi()) << ", "
     << orientation(Direction<3>::upper_eta()) << ", "
     << orientation(Direction<3>::upper_zeta()) << ")";
  return os;
}

template class OrientationMap<1>;
template class OrientationMap<2>;
template class OrientationMap<3>;
