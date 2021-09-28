// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "NumericalAlgorithms/DiscontinuousGalerkin/MortarHelpers.hpp"

#include <algorithm>  // IWYU pragma: keep // for std::max

#include "DataStructures/Index.hpp"
#include "Domain/Structure/ElementId.hpp"       // IWYU pragma: keep
#include "Domain/Structure/OrientationMap.hpp"  // IWYU pragma: keep
#include "Domain/Structure/Side.hpp"
#include "NumericalAlgorithms/Spectral/Spectral.hpp"
#include "Utilities/ErrorHandling/Assert.hpp"
#include "Utilities/GenerateInstantiations.hpp"
#include "Utilities/StdArrayHelpers.hpp"

namespace dg {
template <size_t Dim>
Mesh<Dim> mortar_mesh(const Mesh<Dim>& face_mesh1,
                      const Mesh<Dim>& face_mesh2) {
  Index<Dim> mortar_extents{};
  for (size_t i = 0; i < Dim; ++i) {
    ASSERT(
        face_mesh1.basis(i) == Spectral::Basis::Legendre and
            face_mesh2.basis(i) == Spectral::Basis::Legendre,
        "Only Legendre basis meshes are supported for element faces so far.");
    ASSERT(face_mesh1.quadrature(i) == face_mesh2.quadrature(i),
           "The quadrature on face_mesh1 and face_mesh2 must be equal in "
           "direction "
               << i << " but face_mesh1 is " << face_mesh1.quadrature(i)
               << " while face_mesh2 is " << face_mesh2.quadrature(i));
    ASSERT(face_mesh1.quadrature(i) == Spectral::Quadrature::Gauss or
               face_mesh1.quadrature(i) == Spectral::Quadrature::GaussLobatto,
           "The quadrature on the faces must be Gauss or GaussLobatto, not "
               << face_mesh1.quadrature(i) << ". The direction is " << i);
    mortar_extents[i] = std::max(face_mesh1.extents(i), face_mesh2.extents(i));
  }
  // In 0-d we don't care about basis or quadrature so just specify GaussLobatto
  return {
      mortar_extents.indices(), Spectral::Basis::Legendre,
      Dim != 0 ? face_mesh1.quadrature(0) : Spectral::Quadrature::GaussLobatto};
}

template <size_t Dim>
std::array<Spectral::MortarSize, Dim - 1> mortar_size(
    const ElementId<Dim>& self, const ElementId<Dim>& neighbor,
    const size_t dimension, const OrientationMap<Dim>& orientation) {
  const auto self_segments =
      all_but_specified_element_of(self.segment_ids(), dimension);
  const auto neighbor_segments = all_but_specified_element_of(
      orientation.inverse_map()(neighbor.segment_ids()), dimension);

  std::array<Spectral::MortarSize, Dim - 1> result{};
  for (size_t d = 0; d < Dim - 1; ++d) {
    const auto& self_segment = gsl::at(self_segments, d);
    const auto& neighbor_segment = gsl::at(neighbor_segments, d);
    if (neighbor_segment == self_segment.id_of_child(Side::Lower)) {
      gsl::at(result, d) = Spectral::MortarSize::LowerHalf;
    } else if (neighbor_segment == self_segment.id_of_child(Side::Upper)) {
      gsl::at(result, d) = Spectral::MortarSize::UpperHalf;
    } else {
      ASSERT(neighbor_segment == self_segment or
             neighbor_segment == self_segment.id_of_parent(),
             "Neighbor elements do not overlap 1:1 or 2:1: " << self_segment
             << " " << neighbor_segment);
      gsl::at(result, d) = Spectral::MortarSize::Full;
    }
  }
  return result;
}

#define DIM(data) BOOST_PP_TUPLE_ELEM(0, data)
#define INSTANTIATE(_, data)                                               \
  template Mesh<DIM(data)> mortar_mesh(const Mesh<DIM(data)>& face_mesh1,  \
                                       const Mesh<DIM(data)>& face_mesh2); \
  template std::array<Spectral::MortarSize, DIM(data)> mortar_size(        \
      const ElementId<DIM(data) + 1>& self,                                \
      const ElementId<DIM(data) + 1>& neighbor, size_t dimension,          \
      const OrientationMap<DIM(data) + 1>& orientation);

GENERATE_INSTANTIATIONS(INSTANTIATE, (0, 1, 2))

#undef INSTANTIATE
#undef DIM
}  // namespace dg
