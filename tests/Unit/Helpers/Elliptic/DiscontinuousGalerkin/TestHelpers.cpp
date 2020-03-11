// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Helpers/Elliptic/DiscontinuousGalerkin/TestHelpers.hpp"

#include <cstddef>
#include <string>
#include <unordered_map>
#include <vector>

#include "DataStructures/Tensor/Tensor.hpp"
#include "Domain/CreateInitialElement.hpp"
#include "Domain/CreateInitialMesh.hpp"
#include "Domain/Creators/DomainCreator.hpp"
#include "Domain/Element.hpp"
#include "Domain/ElementId.hpp"
#include "Domain/ElementMap.hpp"
#include "Domain/InitialElementIds.hpp"
#include "Domain/LogicalCoordinates.hpp"
#include "Domain/Mesh.hpp"
#include "Domain/SegmentId.hpp"
#include "NumericalAlgorithms/DiscontinuousGalerkin/MortarHelpers.hpp"
#include "Utilities/GenerateInstantiations.hpp"

/// \cond
namespace TestHelpers {
namespace elliptic {
namespace dg {

template <size_t Dim>
bool ElementOrdering<Dim>::operator()(const ElementId<Dim>& lhs,
                                      const ElementId<Dim>& rhs) const
    noexcept {
  if (lhs.block_id() != rhs.block_id()) {
    return lhs.block_id() < rhs.block_id();
  }
  for (size_t d = 0; d < Dim; d++) {
    const auto& lhs_segment_id = lhs.segment_ids().at(d);
    const auto& rhs_segment_id = rhs.segment_ids().at(d);
    if (lhs_segment_id.refinement_level() !=
        lhs_segment_id.refinement_level()) {
      return lhs_segment_id.refinement_level() <
             lhs_segment_id.refinement_level();
    }
    if (lhs_segment_id.index() != rhs_segment_id.index()) {
      return lhs_segment_id.index() < rhs_segment_id.index();
    }
  }
  return false;
}

template <size_t Dim>
DgElementArray<Dim> create_elements(
    const DomainCreator<Dim>& domain_creator) noexcept {
  const auto domain = domain_creator.create_domain();
  const auto domain_extents = domain_creator.initial_extents();
  const auto refinement_levels = domain_creator.initial_refinement_levels();
  DgElementArray<Dim> elements{};
  for (const auto& block : domain.blocks()) {
    const auto element_ids =
        initial_element_ids(block.id(), refinement_levels[block.id()]);
    for (const auto& element_id : element_ids) {
      const auto mesh = domain::Initialization::create_initial_mesh(
          domain_extents, element_id);
      ElementMap<Dim, Frame::Inertial> element_map{
          element_id, block.stationary_map().get_clone()};
      const auto logical_coords = logical_coordinates(mesh);
      auto inv_jacobian = element_map.inv_jacobian(logical_coords);
      elements.emplace(
          element_id,
          DgElement<Dim>{mesh,
                         domain::Initialization::create_initial_element(
                             element_id, block, refinement_levels),
                         std::move(element_map), std::move(inv_jacobian)});
    }
  }
  return elements;
}

template <size_t VolumeDim>
::dg::MortarMap<VolumeDim,
                std::pair<Mesh<VolumeDim - 1>, ::dg::MortarSize<VolumeDim - 1>>>
create_mortars(const ElementId<VolumeDim>& element_id,
               const DgElementArray<VolumeDim>& dg_elements) noexcept {
  const auto& dg_element = dg_elements.at(element_id);
  ::dg::MortarMap<VolumeDim, std::pair<Mesh<VolumeDim - 1>,
                                       ::dg::MortarSize<VolumeDim - 1>>>
      mortars{};
  for (const auto& direction_and_neighbors : dg_element.element.neighbors()) {
    const auto& direction = direction_and_neighbors.first;
    const auto& neighbors = direction_and_neighbors.second;
    const size_t dimension = direction.dimension();
    const auto face_mesh = dg_element.mesh.slice_away(dimension);
    for (const auto& neighbor : neighbors) {
      ::dg::MortarId<VolumeDim> mortar_id{direction, neighbor};
      mortars.emplace(
          std::move(mortar_id),
          std::make_pair(
              ::dg::mortar_mesh(face_mesh,
                                dg_elements.at(neighbor).mesh.slice_away(
                                    direction.dimension())),
              ::dg::mortar_size(element_id, neighbor, dimension,
                                neighbors.orientation())));
    }
  }
  for (const auto& direction : dg_element.element.external_boundaries()) {
    const size_t dimension = direction.dimension();
    auto face_mesh = dg_element.mesh.slice_away(dimension);
    ::dg::MortarId<VolumeDim> mortar_id{
        direction, ElementId<VolumeDim>::external_boundary_id()};
    mortars.emplace(
        std::move(mortar_id),
        std::make_pair(std::move(face_mesh),
                       make_array<VolumeDim - 1>(Spectral::MortarSize::Full)));
  }
  return mortars;
}

#define DIM(data) BOOST_PP_TUPLE_ELEM(0, data)

#define INSTANTIATE(r, data)                                           \
  template struct ElementOrdering<DIM(data)>;                          \
  template DgElementArray<DIM(data)> create_elements(                  \
      const DomainCreator<DIM(data)>& domain_creator) noexcept;        \
  template ::dg::MortarMap<                                            \
      DIM(data),                                                       \
      std::pair<Mesh<DIM(data) - 1>, ::dg::MortarSize<DIM(data) - 1>>> \
  create_mortars(const ElementId<DIM(data)>& element_id,               \
                 const DgElementArray<DIM(data)>& dg_elements) noexcept;

GENERATE_INSTANTIATIONS(INSTANTIATE, (1, 2, 3))

#undef DIM
#undef INSTANTIATE

}  // namespace dg
}  // namespace elliptic
}  // namespace TestHelpers
/// \endcond
