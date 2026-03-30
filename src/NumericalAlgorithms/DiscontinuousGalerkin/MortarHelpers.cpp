// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "NumericalAlgorithms/DiscontinuousGalerkin/MortarHelpers.hpp"

#include <algorithm>

#include "DataStructures/Index.hpp"
#include "Domain/Structure/ElementId.hpp"
#include "Domain/Structure/OrientationMap.hpp"
#include "Domain/Structure/Side.hpp"
#include "NumericalAlgorithms/Spectral/Basis.hpp"
#include "NumericalAlgorithms/Spectral/Mesh.hpp"
#include "NumericalAlgorithms/Spectral/Quadrature.hpp"
#include "NumericalAlgorithms/Spectral/SegmentSize.hpp"
#include "Utilities/ErrorHandling/Assert.hpp"
#include "Utilities/GenerateInstantiations.hpp"
#include "Utilities/StdArrayHelpers.hpp"

namespace dg {
template <size_t Dim>
Mesh<Dim> mortar_mesh(const Mesh<Dim>& face_mesh1,
                      const Mesh<Dim>& face_mesh2) {
  if constexpr (Dim == 0) {
    return {{{}}, {{}}, {{}}};
  } else {
    Index<Dim> mortar_extents{};
    for (size_t i = 0; i < Dim; ++i) {
      // From the point of view of mortars, ZernikeB2 with Equiangular
      // quadrature are identical to Fourier with Equiangular quadrature
      ASSERT(
          face_mesh1.basis(i) == face_mesh2.basis(i) or
              (face_mesh1.quadrature(i) == Spectral::Quadrature::Equiangular and
               ((face_mesh1.basis(i) == Spectral::Basis::ZernikeB2 and
                 face_mesh2.basis(0) == Spectral::Basis::Fourier) or
                (face_mesh1.basis(i) == Spectral::Basis::Fourier and
                 face_mesh2.basis(0) == Spectral::Basis::ZernikeB2))),
          "The quadrature on face_mesh1 and face_mesh2 must be equal in "
          "direction "
              << i << " but face_mesh1 is " << face_mesh1.basis(i)
              << " while face_mesh2 is " << face_mesh2.basis(i));
      ASSERT(face_mesh1.quadrature(i) == face_mesh2.quadrature(i),
             "The quadrature on face_mesh1 and face_mesh2 must be equal in "
             "direction "
                 << i << " but face_mesh1 is " << face_mesh1.quadrature(i)
                 << " while face_mesh2 is " << face_mesh2.quadrature(i));
      ASSERT(
          face_mesh1.basis(i) != Spectral::Basis::FiniteDifference,
          "Cannot construct a mortar_mesh when the basis is FiniteDifference.");
      mortar_extents[i] =
          std::max(face_mesh1.extents(i), face_mesh2.extents(i));
    }
    return {mortar_extents.indices(), face_mesh1.basis(),
            face_mesh1.quadrature()};
  }
}

template <size_t Dim>
std::array<Spectral::SegmentSize, Dim - 1> mortar_size(
    const ElementId<Dim>& self, const ElementId<Dim>& neighbor,
    const size_t dimension, const OrientationMap<Dim>& orientation) {
  const auto self_segments =
      all_but_specified_element_of(self.segment_ids(), dimension);
  const auto neighbor_segments = all_but_specified_element_of(
      orientation.inverse_map()(neighbor.segment_ids()), dimension);

  std::array<Spectral::SegmentSize, Dim - 1> result{};
  for (size_t d = 0; d < Dim - 1; ++d) {
    const auto& self_segment = gsl::at(self_segments, d);
    const auto& neighbor_segment = gsl::at(neighbor_segments, d);
    if (neighbor_segment == self_segment.id_of_child(Side::Lower)) {
      gsl::at(result, d) = Spectral::SegmentSize::LowerHalf;
    } else if (neighbor_segment == self_segment.id_of_child(Side::Upper)) {
      gsl::at(result, d) = Spectral::SegmentSize::UpperHalf;
    } else {
      ASSERT(neighbor_segment == self_segment or
             neighbor_segment == self_segment.id_of_parent(),
             "Neighbor elements do not overlap 1:1 or 2:1: " << self_segment
             << " " << neighbor_segment);
      gsl::at(result, d) = Spectral::SegmentSize::Full;
    }
  }
  return result;
}

template <size_t Dim>
std::array<SegmentId, Dim - 1> mortar_segments(
    const ElementId<Dim>& self, const ElementId<Dim>& neighbor,
    const size_t dimension, const OrientationMap<Dim>& orientation) {
  const auto self_segments =
      all_but_specified_element_of(self.segment_ids(), dimension);
  const auto neighbor_segments = all_but_specified_element_of(
      orientation.inverse_map()(neighbor.segment_ids()), dimension);

  std::array<SegmentId, Dim - 1> result{};
  for (size_t d = 0; d < Dim - 1; ++d) {
    const auto& self_segment = gsl::at(self_segments, d);
    const auto& neighbor_segment = gsl::at(neighbor_segments, d);
    if (self_segment.refinement_level() ==
        neighbor_segment.refinement_level()) {
      ASSERT(self_segment == neighbor_segment,
             "Neighbor elements do not overlap: "
                 << self_segment << " " << neighbor_segment);
      gsl::at(result, d) = self_segment;
    } else if (self_segment.refinement_level() <
               neighbor_segment.refinement_level()) {
      ASSERT(self_segment == neighbor_segment.id_of_parent(),
             "Neighbor elements do not overlap 1:1 or 2:1: "
                 << self_segment << " " << neighbor_segment);
      gsl::at(result, d) = neighbor_segment;
    } else {
      ASSERT(neighbor_segment == self_segment.id_of_parent(),
             "Neighbor elements do not overlap 1:1 or 2:1: "
                 << self_segment << " " << neighbor_segment);
      gsl::at(result, d) = self_segment;
    }
  }
  return result;
}

#define DIM(data) BOOST_PP_TUPLE_ELEM(0, data)
#define INSTANTIATE(_, data)                                               \
  template Mesh<DIM(data)> mortar_mesh(const Mesh<DIM(data)>& face_mesh1,  \
                                       const Mesh<DIM(data)>& face_mesh2); \
  template std::array<Spectral::SegmentSize, DIM(data)> mortar_size(       \
      const ElementId<DIM(data) + 1>& self,                                \
      const ElementId<DIM(data) + 1>& neighbor, size_t dimension,          \
      const OrientationMap<DIM(data) + 1>& orientation);                   \
  template std::array<SegmentId, DIM(data)> mortar_segments(               \
      const ElementId<DIM(data) + 1>& self,                                \
      const ElementId<DIM(data) + 1>& neighbor, size_t dimension,          \
      const OrientationMap<DIM(data) + 1>& orientation);

GENERATE_INSTANTIATIONS(INSTANTIATE, (0, 1, 2))

#undef INSTANTIATE
#undef DIM
}  // namespace dg
