// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Domain/Structure/OrientationMap.hpp"

#include <functional>
#include <ostream>
#include <pup.h>
#include <pup_stl.h>
#include <set>

#include "DataStructures/DataVector.hpp"
#include "Domain/Structure/SegmentId.hpp"  // IWYU pragma: keep
#include "Utilities/ErrorHandling/Assert.hpp"
#include "Utilities/GenerateInstantiations.hpp"

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
  ASSERT(set_of_dimensions(mapped_directions_).size() == VolumeDim,
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

template <size_t VolumeDim, typename T>
std::array<tt::remove_cvref_wrap_t<T>, VolumeDim> discrete_rotation(
    const OrientationMap<VolumeDim>& rotation,
    std::array<T, VolumeDim> source_coords) noexcept {
  using ReturnType = tt::remove_cvref_wrap_t<T>;
  std::array<ReturnType, VolumeDim> new_coords{};
  for (size_t i = 0; i < VolumeDim; i++) {
    const auto new_direction = rotation(Direction<VolumeDim>(i, Side::Upper));
    gsl::at(new_coords, i) =
        std::move(gsl::at(source_coords, new_direction.dimension()));
    if (new_direction.side() != Side::Upper) {
      gsl::at(new_coords, i) *= -1.0;
    }
  }
  return new_coords;
}

#define DIM(data) BOOST_PP_TUPLE_ELEM(0, data)
#define DTYPE(data) BOOST_PP_TUPLE_ELEM(1, data)

#define INSTANTIATION(r, data)                                         \
  template std::array<tt::remove_cvref_wrap_t<DTYPE(data)>, DIM(data)> \
  discrete_rotation<DIM(data), DTYPE(data)>(                           \
      const OrientationMap<DIM(data)>& rotation,                       \
      std::array<DTYPE(data), DIM(data)> source_coords) noexcept;

GENERATE_INSTANTIATIONS(INSTANTIATION, (1, 2, 3),
                        (double, DataVector, const double, const DataVector,
                         std::reference_wrapper<double>,
                         std::reference_wrapper<DataVector>,
                         std::reference_wrapper<const double>,
                         std::reference_wrapper<const DataVector>,
                         std::reference_wrapper<double> const,
                         std::reference_wrapper<DataVector> const,
                         std::reference_wrapper<const double> const,
                         std::reference_wrapper<const DataVector> const))

#undef INSTANTIATION
#undef DTYPE
#undef DIM
