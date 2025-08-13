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
#include "Domain/Structure/FaceType.hpp"
#include "Evolution/DiscontinuousGalerkin/InterfaceDataPolicy.hpp"
#include "Evolution/DiscontinuousGalerkin/MortarData.hpp"
#include "Evolution/DiscontinuousGalerkin/MortarDataHolder.hpp"
#include "Evolution/DiscontinuousGalerkin/MortarInfo.hpp"
#include "Evolution/DiscontinuousGalerkin/NormalVectorTags.hpp"
#include "Evolution/DiscontinuousGalerkin/TimeSteppingPolicy.hpp"
#include "NumericalAlgorithms/DiscontinuousGalerkin/MortarHelpers.hpp"
#include "NumericalAlgorithms/Spectral/Mesh.hpp"
#include "NumericalAlgorithms/Spectral/Quadrature.hpp"
#include "NumericalAlgorithms/Spectral/SegmentSize.hpp"
#include "Time/TimeStepId.hpp"
#include "Utilities/ErrorHandling/Error.hpp"
#include "Utilities/GenerateInstantiations.hpp"
#include "Utilities/MakeString.hpp"

namespace evolution::dg::Initialization::detail {

template <size_t Dim>
::dg::MortarMap<Dim, evolution::dg::MortarDataHolder<Dim>> empty_mortar_data(
    const Element<Dim>& element) {
  ::dg::MortarMap<Dim, evolution::dg::MortarDataHolder<Dim>> mortar_data{};
  for (const auto& [direction, neighbors] : element.neighbors()) {
    const domain::FaceType face_type = element.face_types().at(direction);
    switch (face_type) {
      // NOLINTNEXTLINE(bugprone-branch-clone)
      case (domain::FaceType::Uninitialized):
        [[fallthrough]];
      case (domain::FaceType::External):
        [[fallthrough]];
      case (domain::FaceType::Topological): {
        const std::string message =
            MakeString{} << "Cannot have FaceType " << face_type
                         << " in direction " << direction
                         << " in which there are neighboring elements.";
        ERROR(message);
      }
      // NOLINTNEXTLINE(bugprone-branch-clone)
      case (domain::FaceType::ConformingAligned):
        [[fallthrough]];
      case (domain::FaceType::ConformingUnaligned):
        [[fallthrough]];
      case (domain::FaceType::SingleNonconforming): {
        for (const auto& neighbor : neighbors) {
          const DirectionalId<Dim> mortar_id{direction, neighbor};
          mortar_data.emplace(mortar_id, MortarDataHolder<Dim>{});
        }
        break;
      }
      case (domain::FaceType::MultipleNonconforming): {
        // Although there are multiple neighbors, we only create a single
        // mortar labeled by the host ElementId.  This is done because
        // the data from all neighbors will be combined onto a single mortar
        // as it makes no sense to have multiple mortars between
        // non-conforming Elements
        const DirectionalId<Dim> mortar_id{direction, element.id()};
        mortar_data.emplace(mortar_id, MortarDataHolder<Dim>{});
        break;
      }
      default:
        ERROR("Invalid face type" << face_type);
    }
  }
  return mortar_data;
}

template <size_t Dim>
::dg::MortarMap<Dim, MortarInfo<Dim>> mortar_infos(
    const Domain<Dim>& domain, const Element<Dim>& element,
    const Mesh<Dim>& volume_mesh,
    const ::dg::MortarMap<Dim, Mesh<Dim>>& neighbor_mesh,
    const bool local_time_stepping) {
  ::dg::MortarMap<Dim, MortarInfo<Dim>> result{};
  for (const auto& [direction, neighbors] : element.neighbors()) {
    const domain::FaceType face_type = element.face_types().at(direction);
    switch (face_type) {
      // NOLINTNEXTLINE(bugprone-branch-clone)
      case (domain::FaceType::Uninitialized):
        [[fallthrough]];
      case (domain::FaceType::External):
        [[fallthrough]];
      case (domain::FaceType::Topological): {
        const std::string message =
            MakeString{} << "Cannot have FaceType " << face_type
                         << " in direction " << direction
                         << " in which there are neighboring elements.";
        ERROR(message);
      }
      case (domain::FaceType::ConformingAligned):
        [[fallthrough]];
      case (domain::FaceType::ConformingUnaligned):
        for (const auto& neighbor : neighbors) {
          const DirectionalId<Dim> mortar_id{direction, neighbor};
          const auto& neighbor_orientation = neighbors.orientation(neighbor);
          result.emplace(
              mortar_id,
              MortarInfo<Dim>{
                  {.mortar_size = ::dg::mortar_size(element.id(), neighbor,
                                                    direction.dimension(),
                                                    neighbor_orientation),
                   .interface_data_policy =
                       neighbor_orientation.is_aligned()
                           ? InterfaceDataPolicy::CopyProject
                           : InterfaceDataPolicy::OrientCopyProject,
                   .time_stepping_policy =
                       local_time_stepping ? TimeSteppingPolicy::Conservative
                                           : TimeSteppingPolicy::EqualRate}});
        }
        break;
      case (domain::FaceType::SingleNonconforming):
        if constexpr (Dim > 1) {
          for (const auto& neighbor : neighbors) {
            const DirectionalId<Dim> mortar_id{direction, neighbor};
            const auto& neighbor_orientation = neighbors.orientation(neighbor);
            result.emplace(
                mortar_id,
                MortarInfo<Dim>{
                    {.interpolator =
                         ::dg::MortarInterpolator<Dim>{
                             element.id(), mortar_id, domain,
                             volume_mesh.slice_away(direction.dimension()),
                             neighbor_mesh.at(mortar_id).slice_away(
                                 neighbor_orientation(direction.dimension()))},
                     .interface_data_policy =
                         InterfaceDataPolicy::NonconformingSelfInterpolates,
                     .time_stepping_policy =
                         local_time_stepping ? TimeSteppingPolicy::Conservative
                                             : TimeSteppingPolicy::EqualRate}});
          }
        }
        break;
      case (domain::FaceType::MultipleNonconforming): {
        // Although there are multiple neighbors, we only create a single
        // mortar labeled by the host ElementId.  This is done because
        // the data from all neighbors will be combined onto a single mortar
        // as it makes no sense to have multiple mortars between
        // non-conforming Elements
        const DirectionalId<Dim> mortar_id{direction, element.id()};
        result.emplace(
            mortar_id,
            MortarInfo<Dim>{
                {.interface_data_policy =
                     InterfaceDataPolicy::NonconformingNeighborInterpolates,
                 .time_stepping_policy = local_time_stepping
                                             ? TimeSteppingPolicy::Conservative
                                             : TimeSteppingPolicy::EqualRate}});
        break;
      }
      default:
        ERROR("Invalid face type" << face_type);
    }
  }
  return result;
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
    const Mesh<Dim - 1> face_mesh =
        volume_mesh.slice_away(direction.dimension());
    const domain::FaceType face_type = element.face_types().at(direction);
    switch (face_type) {
      // NOLINTNEXTLINE(bugprone-branch-clone)
      case (domain::FaceType::Uninitialized):
        [[fallthrough]];
      case (domain::FaceType::External):
        [[fallthrough]];
      case (domain::FaceType::Topological): {
        const std::string message =
            MakeString{} << "Cannot have FaceType " << face_type
                         << " in direction " << direction
                         << " in which there are neighboring elements.";
        ERROR(message);
      }
      case (domain::FaceType::ConformingAligned):
        [[fallthrough]];
      case (domain::FaceType::ConformingUnaligned):
        for (const auto& neighbor : neighbors) {
          const DirectionalId<Dim> mortar_id{direction, neighbor};
          mortar_meshes.emplace(
              mortar_id, ::dg::mortar_mesh(
                             face_mesh, neighbor_mesh.at(mortar_id).slice_away(
                                            direction.dimension())));
          // Since no communication needs to happen for boundary conditions
          // the temporal id is not advanced on the boundary, so we only need
          // to initialize it on internal boundaries
          mortar_next_temporal_ids.insert({mortar_id, next_temporal_id});
        }
        break;
      case (domain::FaceType::SingleNonconforming):
        for (const auto& neighbor : neighbors) {
          const DirectionalId<Dim> mortar_id{direction, neighbor};
          mortar_meshes.emplace(mortar_id, face_mesh);
          // Since no communication needs to happen for boundary conditions
          // the temporal id is not advanced on the boundary, so we only need
          // to initialize it on internal boundaries
          mortar_next_temporal_ids.insert({mortar_id, next_temporal_id});
        }
        break;
      case (domain::FaceType::MultipleNonconforming): {
        // Although there are multiple neighbors, we only create a single
        // mortar labeled by the host ElementId.  This is done because
        // the data from all neighbors will be combined onto a single mortar
        // as it makes no sense to have multiple mortars between
        // non-conforming Elements
        const DirectionalId<Dim> mortar_id{direction, element.id()};
        mortar_meshes.emplace(mortar_id, face_mesh);
        // Since no communication needs to happen for boundary conditions
        // the temporal id is not advanced on the boundary, so we only need
        // to initialize it on internal boundaries
        mortar_next_temporal_ids.insert({mortar_id, next_temporal_id});
        break;
      }
      default:
        ERROR("Invalid face type" << face_type);
    }
  }

  for (const auto& direction : element.external_boundaries()) {
    normal_covector_quantities[direction] = std::nullopt;
  }

  return {std::move(mortar_meshes), std::move(mortar_next_temporal_ids),
          std::move(normal_covector_quantities)};
}

template <size_t Dim>
void h_refine_structure(
    const gsl::not_null<
        ::dg::MortarMap<Dim, evolution::dg::MortarDataHolder<Dim>>*>
        mortar_data,
    const gsl::not_null<::dg::MortarMap<Dim, Mesh<Dim - 1>>*> mortar_mesh,
    const gsl::not_null<::dg::MortarMap<Dim, MortarInfo<Dim>>*> mortar_infos,
    const gsl::not_null<::dg::MortarMap<Dim, TimeStepId>*>
        mortar_next_temporal_id,
    const gsl::not_null<
        DirectionMap<Dim, std::optional<::Variables<tmpl::list<
                              ::evolution::dg::Tags::MagnitudeOfNormal,
                              ::evolution::dg::Tags::NormalCovector<Dim>>>>>*>
        normal_covector_and_magnitude,
    const Domain<Dim>& domain, const Mesh<Dim>& new_mesh,
    const Element<Dim>& new_element,
    const ::dg::MortarMap<Dim, Mesh<Dim>>& neighbor_mesh,
    const TimeStepId& current_temporal_id, const bool local_time_stepping) {
  *mortar_data = detail::empty_mortar_data(new_element);
  *mortar_infos = detail::mortar_infos(domain, new_element, new_mesh,
                                       neighbor_mesh, local_time_stepping);
  for (const auto& direction : Direction<Dim>::all_directions()) {
    (*normal_covector_and_magnitude)[direction] = std::nullopt;
  }

  for (const auto& [direction, neighbors] : new_element.neighbors()) {
    const auto sliced_away_dimension = direction.dimension();
    const auto new_face_mesh = new_mesh.slice_away(sliced_away_dimension);
    for (const auto& neighbor : neighbors) {
      const DirectionalId<Dim> mortar_id{direction, neighbor};
      const auto& new_neighbor_mesh = neighbor_mesh.at(mortar_id);
      const auto new_mortar_mesh = ::dg::mortar_mesh(
          new_face_mesh, new_neighbor_mesh.slice_away(sliced_away_dimension));
      mortar_mesh->emplace(mortar_id, new_mortar_mesh);
      // We only do h refinement at a slab boundary, so we know all
      // the neighbors are aligned with us temporally.
      mortar_next_temporal_id->emplace(mortar_id, current_temporal_id);
    }
  }
}

#define DIM(data) BOOST_PP_TUPLE_ELEM(0, data)

#define INSTANTIATION(r, data)                                               \
  template ::dg::MortarMap<DIM(data),                                        \
                           evolution::dg::MortarDataHolder<DIM(data)>>       \
  empty_mortar_data(const Element<DIM(data)>& element);                      \
  template ::dg::MortarMap<DIM(data), MortarInfo<DIM(data)>> mortar_infos(   \
      const Domain<DIM(data)>& domain, const Element<DIM(data)>& element,    \
      const Mesh<DIM(data)>& volume_mesh,                                    \
      const ::dg::MortarMap<DIM(data), Mesh<DIM(data)>>& neighbor_mesh,      \
      bool local_time_stepping);                                             \
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
      const ::dg::MortarMap<DIM(data), Mesh<DIM(data)>>& neighbor_mesh);     \
  template void h_refine_structure(                                          \
      gsl::not_null<::dg::MortarMap<                                         \
          DIM(data), evolution::dg::MortarDataHolder<DIM(data)>>*>           \
          mortar_data,                                                       \
      gsl::not_null<::dg::MortarMap<DIM(data), Mesh<DIM(data) - 1>>*>        \
          mortar_mesh,                                                       \
      gsl::not_null<::dg::MortarMap<DIM(data), MortarInfo<DIM(data)>>*>      \
          mortar_infos,                                                      \
      gsl::not_null<::dg::MortarMap<DIM(data), TimeStepId>*>                 \
          mortar_next_temporal_id,                                           \
      gsl::not_null<DirectionMap<                                            \
          DIM(data),                                                         \
          std::optional<::Variables<tmpl::list<                              \
              ::evolution::dg::Tags::MagnitudeOfNormal,                      \
              ::evolution::dg::Tags::NormalCovector<DIM(data)>>>>>*>         \
          normal_covector_and_magnitude,                                     \
      const Domain<DIM(data)>& domain, const Mesh<DIM(data)>& new_mesh,      \
      const Element<DIM(data)>& new_element,                                 \
      const ::dg::MortarMap<DIM(data), Mesh<DIM(data)>>& neighbor_mesh,      \
      const TimeStepId& current_temporal_id, bool local_time_stepping);

GENERATE_INSTANTIATIONS(INSTANTIATION, (1, 2, 3))

#undef INSTANTIATION
#undef DIM
}  // namespace evolution::dg::Initialization::detail
