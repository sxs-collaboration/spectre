// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Evolution/DiscontinuousGalerkin/Initialization/Mortars.hpp"

#include <array>
#include <cstddef>
#include <tuple>
#include <utility>
#include <vector>

#include "Domain/Structure/Direction.hpp"
#include "Domain/Structure/Element.hpp"
#include "Evolution/DiscontinuousGalerkin/InterfaceDataPolicy.hpp"
#include "Evolution/DiscontinuousGalerkin/MortarData.hpp"
#include "Evolution/DiscontinuousGalerkin/MortarInfo.hpp"
#include "NumericalAlgorithms/DiscontinuousGalerkin/MortarHelpers.hpp"
#include "NumericalAlgorithms/Spectral/Mesh.hpp"
#include "NumericalAlgorithms/Spectral/Quadrature.hpp"
#include "NumericalAlgorithms/Spectral/SegmentSize.hpp"
#include "Time/TimeStepId.hpp"
#include "Utilities/GenerateInstantiations.hpp"

namespace evolution::dg::Initialization::detail {

template <size_t Dim>
::dg::MortarMap<Dim, evolution::dg::MortarDataHolder<Dim>> empty_mortar_data(
    const Element<Dim>& element) {
  ::dg::MortarMap<Dim, evolution::dg::MortarDataHolder<Dim>> mortar_data{};
  for (const auto& [direction, neighbors] : element.neighbors()) {
    for (const auto& neighbor : neighbors) {
      const DirectionalId<Dim> mortar_id{direction, neighbor};
      mortar_data.emplace(mortar_id, MortarDataHolder<Dim>{});
    }
  }

  return mortar_data;
}

template <size_t Dim>
::dg::MortarMap<Dim, MortarInfo<Dim>> mortar_infos(
    const Element<Dim>& element) {
  ::dg::MortarMap<Dim, MortarInfo<Dim>> mortar_infos{};
  for (const auto& [direction, neighbors] : element.neighbors()) {
    for (const auto& neighbor : neighbors) {
      const DirectionalId<Dim> mortar_id{direction, neighbor};
      const auto& neighbor_orientation = neighbors.orientation(neighbor);
      mortar_infos.emplace(
          mortar_id,
          MortarInfo<Dim>{
              {.mortar_size = ::dg::mortar_size(element.id(), neighbor,
                                                direction.dimension(),
                                                neighbor_orientation),
               .policy = neighbor_orientation.is_aligned()
                             ? InterfaceDataPolicy::CopyProject
                             : InterfaceDataPolicy::OrientCopyProject}});
    }
  }

  return mortar_infos;
}

template <size_t Dim>
std::tuple<::dg::MortarMap<Dim, Mesh<Dim - 1>>,
           ::dg::MortarMap<Dim, TimeStepId>,
           DirectionMap<Dim, std::optional<Variables<tmpl::list<
                                 evolution::dg::Tags::MagnitudeOfNormal,
                                 evolution::dg::Tags::NormalCovector<Dim>>>>>>
mortars_apply_impl(const Element<Dim>& element,
                   const TimeStepId& next_temporal_id,
                   const Mesh<Dim>& volume_mesh,
                   const ::dg::MortarMap<Dim, Mesh<Dim>>& neighbor_mesh) {
  ::dg::MortarMap<Dim, Mesh<Dim - 1>> mortar_meshes{};
  ::dg::MortarMap<Dim, TimeStepId> mortar_next_temporal_ids{};
  DirectionMap<Dim, std::optional<Variables<
                        tmpl::list<evolution::dg::Tags::MagnitudeOfNormal,
                                   evolution::dg::Tags::NormalCovector<Dim>>>>>
      normal_covector_quantities{};
  for (const auto& [direction, neighbors] : element.neighbors()) {
    normal_covector_quantities[direction] = std::nullopt;
    for (const auto& neighbor : neighbors) {
      const DirectionalId<Dim> mortar_id{direction, neighbor};
      mortar_meshes.emplace(
          mortar_id,
          ::dg::mortar_mesh(
              volume_mesh.slice_away(direction.dimension()),
              neighbor_mesh.at(mortar_id).slice_away(direction.dimension())));
      // Since no communication needs to happen for boundary conditions
      // the temporal id is not advanced on the boundary, so we only need to
      // initialize it on internal boundaries
      mortar_next_temporal_ids.insert({mortar_id, next_temporal_id});
    }
  }

  for (const auto& direction : element.external_boundaries()) {
    normal_covector_quantities[direction] = std::nullopt;
  }

  return {std::move(mortar_meshes), std::move(mortar_next_temporal_ids),
          std::move(normal_covector_quantities)};
}

#define DIM(data) BOOST_PP_TUPLE_ELEM(0, data)

#define INSTANTIATION(r, data)                                               \
  template ::dg::MortarMap<DIM(data),                                        \
                           evolution::dg::MortarDataHolder<DIM(data)>>       \
  empty_mortar_data(const Element<DIM(data)>& element);                      \
  template ::dg::MortarMap<DIM(data), MortarInfo<DIM(data)>> mortar_infos(   \
      const Element<DIM(data)>& element);                                    \
  template std::tuple<                                                       \
      ::dg::MortarMap<DIM(data), Mesh<DIM(data) - 1>>,                       \
      ::dg::MortarMap<DIM(data), TimeStepId>,                                \
      DirectionMap<DIM(data),                                                \
                   std::optional<Variables<tmpl::list<                       \
                       evolution::dg::Tags::MagnitudeOfNormal,               \
                       evolution::dg::Tags::NormalCovector<DIM(data)>>>>>>   \
  mortars_apply_impl(                                                        \
      const Element<DIM(data)>& element, const TimeStepId& next_temporal_id, \
      const Mesh<DIM(data)>& volume_mesh,                                    \
      const ::dg::MortarMap<DIM(data), Mesh<DIM(data)>>& neighbor_mesh);

GENERATE_INSTANTIATIONS(INSTANTIATION, (1, 2, 3))

#undef INSTANTIATION
#undef DIM
}  // namespace evolution::dg::Initialization::detail
