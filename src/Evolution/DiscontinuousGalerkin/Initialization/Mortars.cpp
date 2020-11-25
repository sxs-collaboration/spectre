// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Evolution/DiscontinuousGalerkin/Initialization/Mortars.hpp"

#include <array>
#include <cstddef>
#include <tuple>
#include <unordered_map>
#include <utility>
#include <vector>

#include "Domain/Structure/CreateInitialMesh.hpp"
#include "Domain/Structure/Direction.hpp"
#include "Domain/Structure/Element.hpp"
#include "Evolution/DiscontinuousGalerkin/MortarData.hpp"
#include "NumericalAlgorithms/DiscontinuousGalerkin/MortarHelpers.hpp"
#include "NumericalAlgorithms/Spectral/Mesh.hpp"
#include "NumericalAlgorithms/Spectral/Projection.hpp"
#include "Time/TimeStepId.hpp"
#include "Utilities/GenerateInstantiations.hpp"

namespace evolution::dg::Initialization {
template <size_t Dim, bool AddFluxBoundaryConditionMortars>
auto Mortars<Dim, AddFluxBoundaryConditionMortars>::apply_impl(
    const std::vector<std::array<size_t, Dim>>& initial_extents,
    const Spectral::Quadrature quadrature, const Element<Dim>& element,
    const TimeStepId& next_temporal_id,
    const std::unordered_map<Direction<Dim>, Mesh<Dim - 1>>& interface_meshes,
    const std::unordered_map<Direction<Dim>, Mesh<Dim - 1>>&
        boundary_meshes) noexcept
    -> std::tuple<MortarMap<evolution::dg::MortarData<Dim>>,
                  MortarMap<Mesh<Dim - 1>>,
                  MortarMap<std::array<Spectral::MortarSize, Dim - 1>>,
                  MortarMap<TimeStepId>> {
  MortarMap<evolution::dg::MortarData<Dim>> mortar_data{};
  MortarMap<Mesh<Dim - 1>> mortar_meshes{};
  MortarMap<std::array<Spectral::MortarSize, Dim - 1>> mortar_sizes{};
  MortarMap<TimeStepId> mortar_next_temporal_ids{};
  for (const auto& [direction, neighbors] : element.neighbors()) {
    for (const auto& neighbor : neighbors) {
      const auto mortar_id = std::make_pair(direction, neighbor);
      mortar_data[mortar_id];  // Default initialize data
      mortar_meshes.emplace(
          mortar_id,
          ::dg::mortar_mesh(interface_meshes.at(direction),
                            ::domain::Initialization::create_initial_mesh(
                                initial_extents, neighbor, quadrature,
                                neighbors.orientation())
                                .slice_away(direction.dimension())));
      mortar_sizes.emplace(
          mortar_id,
          ::dg::mortar_size(element.id(), neighbor, direction.dimension(),
                            neighbors.orientation()));
      // Since no communication needs to happen for boundary conditions
      // the temporal id is not advanced on the boundary, so we only need to
      // initialize it on internal boundaries
      mortar_next_temporal_ids.insert({mortar_id, next_temporal_id});
    }
  }

  // In a future update, we will update the logic below to also check the actual
  // boundary condition being imposed.
  if constexpr (AddFluxBoundaryConditionMortars) {
    for (const auto& direction : element.external_boundaries()) {
      const auto mortar_id =
          std::make_pair(direction, ElementId<Dim>::external_boundary_id());
      mortar_data[mortar_id];  // Default initialize data
      mortar_meshes.emplace(mortar_id, boundary_meshes.at(direction));
      mortar_sizes.emplace(mortar_id,
                           make_array<Dim - 1>(Spectral::MortarSize::Full));
    }
  } else {
    (void)boundary_meshes;
  }

  return {std::move(mortar_data), std::move(mortar_meshes),
          std::move(mortar_sizes), std::move(mortar_next_temporal_ids)};
}

#define DIM(data) BOOST_PP_TUPLE_ELEM(0, data)
#define ADD_BOUNDARY_FLUX(data) BOOST_PP_TUPLE_ELEM(1, data)

#define INSTANTIATION(r, data) \
  template class Mortars<DIM(data), ADD_BOUNDARY_FLUX(data)>;

GENERATE_INSTANTIATIONS(INSTANTIATION, (1, 2, 3), (true, false))

#undef INSTANTIATION
#undef ADD_BOUNDARY_FLUX
#undef DIM
}  // namespace evolution::dg::Initialization
