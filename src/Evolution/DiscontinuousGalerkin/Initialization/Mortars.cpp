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

namespace evolution::dg::Initialization::detail {
namespace {
template <size_t Dim>
using Key = std::pair<Direction<Dim>, ElementId<Dim>>;
template <typename MappedType, size_t Dim>
using MortarMap =
    std::unordered_map<Key<Dim>, MappedType, boost::hash<Key<Dim>>>;
}  // namespace

template <size_t Dim>
std::tuple<
    std::unordered_map<std::pair<Direction<Dim>, ElementId<Dim>>,
                       evolution::dg::MortarData<Dim>,
                       boost::hash<std::pair<Direction<Dim>, ElementId<Dim>>>>,
    std::unordered_map<std::pair<Direction<Dim>, ElementId<Dim>>, Mesh<Dim - 1>,
                       boost::hash<std::pair<Direction<Dim>, ElementId<Dim>>>>,
    std::unordered_map<std::pair<Direction<Dim>, ElementId<Dim>>,
                       std::array<Spectral::MortarSize, Dim - 1>,
                       boost::hash<std::pair<Direction<Dim>, ElementId<Dim>>>>,
    std::unordered_map<std::pair<Direction<Dim>, ElementId<Dim>>, TimeStepId,
                       boost::hash<std::pair<Direction<Dim>, ElementId<Dim>>>>,
    DirectionMap<Dim, std::optional<Variables<tmpl::list<
                          evolution::dg::Tags::MagnitudeOfNormal,
                          evolution::dg::Tags::NormalCovector<Dim>>>>>>
mortars_apply_impl(const std::vector<std::array<size_t, Dim>>& initial_extents,
                   const Spectral::Quadrature quadrature,
                   const Element<Dim>& element,
                   const TimeStepId& next_temporal_id,
                   const Mesh<Dim>& volume_mesh) {
  MortarMap<evolution::dg::MortarData<Dim>, Dim> mortar_data{};
  MortarMap<Mesh<Dim - 1>, Dim> mortar_meshes{};
  MortarMap<std::array<Spectral::MortarSize, Dim - 1>, Dim> mortar_sizes{};
  MortarMap<TimeStepId, Dim> mortar_next_temporal_ids{};
  DirectionMap<Dim, std::optional<Variables<
                        tmpl::list<evolution::dg::Tags::MagnitudeOfNormal,
                                   evolution::dg::Tags::NormalCovector<Dim>>>>>
      normal_covector_quantities{};
  for (const auto& [direction, neighbors] : element.neighbors()) {
    normal_covector_quantities[direction] = std::nullopt;
    for (const auto& neighbor : neighbors) {
      const auto mortar_id = std::make_pair(direction, neighbor);
      mortar_data[mortar_id];  // Default initialize data
      mortar_meshes.emplace(
          mortar_id,
          ::dg::mortar_mesh(volume_mesh.slice_away(direction.dimension()),
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

  for (const auto& direction : element.external_boundaries()) {
    normal_covector_quantities[direction] = std::nullopt;
  }

  return {std::move(mortar_data), std::move(mortar_meshes),
          std::move(mortar_sizes), std::move(mortar_next_temporal_ids),
          std::move(normal_covector_quantities)};
}

#define DIM(data) BOOST_PP_TUPLE_ELEM(0, data)

#define INSTANTIATION(r, data)                                                 \
  template std::tuple<                                                         \
      std::unordered_map<                                                      \
          std::pair<Direction<DIM(data)>, ElementId<DIM(data)>>,               \
          evolution::dg::MortarData<DIM(data)>,                                \
          boost::hash<std::pair<Direction<DIM(data)>, ElementId<DIM(data)>>>>, \
      std::unordered_map<                                                      \
          std::pair<Direction<DIM(data)>, ElementId<DIM(data)>>,               \
          Mesh<DIM(data) - 1>,                                                 \
          boost::hash<std::pair<Direction<DIM(data)>, ElementId<DIM(data)>>>>, \
      std::unordered_map<                                                      \
          std::pair<Direction<DIM(data)>, ElementId<DIM(data)>>,               \
          std::array<Spectral::MortarSize, DIM(data) - 1>,                     \
          boost::hash<std::pair<Direction<DIM(data)>, ElementId<DIM(data)>>>>, \
      std::unordered_map<                                                      \
          std::pair<Direction<DIM(data)>, ElementId<DIM(data)>>, TimeStepId,   \
          boost::hash<std::pair<Direction<DIM(data)>, ElementId<DIM(data)>>>>, \
      DirectionMap<DIM(data),                                                  \
                   std::optional<Variables<tmpl::list<                         \
                       evolution::dg::Tags::MagnitudeOfNormal,                 \
                       evolution::dg::Tags::NormalCovector<DIM(data)>>>>>>     \
  mortars_apply_impl(                                                          \
      const std::vector<std::array<size_t, DIM(data)>>& initial_extents,       \
      const Spectral::Quadrature quadrature,                                   \
      const Element<DIM(data)>& element, const TimeStepId& next_temporal_id,   \
      const Mesh<DIM(data)>& volume_mesh);

GENERATE_INSTANTIATIONS(INSTANTIATION, (1, 2, 3))

#undef INSTANTIATION
#undef DIM
}  // namespace evolution::dg::Initialization::detail
