// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "NumericalAlgorithms/DiscontinuousGalerkin/MortarHelpers.hpp"

#include <algorithm>  // IWYU pragma: keep // for std::max

#include "DataStructures/Index.hpp"
#include "Domain/ElementId.hpp"  // IWYU pragma: keep
#include "Domain/OrientationMap.hpp"  // IWYU pragma: keep
#include "Domain/Side.hpp"
#include "ErrorHandling/Assert.hpp"
#include "NumericalAlgorithms/Spectral/Spectral.hpp"
#include "Utilities/GenerateInstantiations.hpp"
#include "Utilities/StdArrayHelpers.hpp"

namespace dg {
template <size_t Dim>
domain::Mesh<Dim> mortar_mesh(const domain::Mesh<Dim>& face_mesh1,
                              const domain::Mesh<Dim>& face_mesh2) noexcept {
  Index<Dim> mortar_extents{};
  for (size_t i = 0; i < Dim; ++i) {
    mortar_extents[i] = std::max(face_mesh1.extents(i), face_mesh2.extents(i));
  }
  return {mortar_extents.indices(), Spectral::Basis::Legendre,
          Spectral::Quadrature::GaussLobatto};
}

template <size_t Dim>
std::array<Spectral::MortarSize, Dim - 1> mortar_size(
    const domain::ElementId<Dim>& self, const domain::ElementId<Dim>& neighbor,
    const size_t dimension,
    const domain::OrientationMap<Dim>& orientation) noexcept {
  const auto self_segments =
      all_but_specified_element_of(self.segment_ids(), dimension);
  const auto neighbor_segments = all_but_specified_element_of(
      orientation.inverse_map()(neighbor.segment_ids()), dimension);

  std::array<Spectral::MortarSize, Dim - 1> result{};
  for (size_t d = 0; d < Dim - 1; ++d) {
    const auto& self_segment = gsl::at(self_segments, d);
    const auto& neighbor_segment = gsl::at(neighbor_segments, d);
    if (neighbor_segment == self_segment.id_of_child(domain::Side::Lower)) {
      gsl::at(result, d) = Spectral::MortarSize::LowerHalf;
    } else if (neighbor_segment ==
               self_segment.id_of_child(domain::Side::Upper)) {
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
#define INSTANTIATE(_, data)                                              \
  template domain::Mesh<DIM(data)> mortar_mesh(                           \
      const domain::Mesh<DIM(data)>& face_mesh1,                          \
      const domain::Mesh<DIM(data)>& face_mesh2) noexcept;                \
  template std::array<Spectral::MortarSize, DIM(data)> mortar_size(       \
      const domain::ElementId<DIM(data) + 1>& self,                       \
      const domain::ElementId<DIM(data) + 1>& neighbor, size_t dimension, \
      const domain::OrientationMap<DIM(data) + 1>& orientation) noexcept;

GENERATE_INSTANTIATIONS(INSTANTIATE, (0, 1, 2))

#undef INSTANTIATE
#undef DIM
}  // namespace dg
