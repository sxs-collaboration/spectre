// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <array>
#include <cstddef>
#include <tuple>
#include <utility>
#include <vector>

#include "DataStructures/DataBox/DataBox.hpp"
#include "DataStructures/DataBox/DataBoxTag.hpp"
#include "DataStructures/DataBox/Prefixes.hpp"
#include "Domain/Direction.hpp"
#include "Domain/Element.hpp"
#include "Domain/Mesh.hpp"
#include "Domain/Neighbors.hpp"
#include "Domain/OrientationMap.hpp"
#include "Domain/Tags.hpp"
#include "Evolution/Conservative/Tags.hpp"
#include "NumericalAlgorithms/DiscontinuousGalerkin/FluxCommunicationTypes.hpp"
#include "NumericalAlgorithms/DiscontinuousGalerkin/MortarHelpers.hpp"
#include "NumericalAlgorithms/DiscontinuousGalerkin/Tags.hpp"
#include "ParallelAlgorithms/Initialization/MergeIntoDataBox.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/TMPL.hpp"

/// \cond
namespace Frame {
struct Inertial;
}  // namespace Frame
namespace Parallel {
template <typename Metavariables>
class ConstGlobalCache;
}  // namespace Parallel
/// \endcond

namespace dg {
namespace Actions {
/// \ingroup InitializationGroup
/// \brief Initialize items required for computing boundary fluxes
///
/// Uses:
/// - DataBox:
///   * `Tags::Interface<Tags::InternalDirections<Dim>, Tags::Mesh<Dim - 1>>`
///
/// DataBox changes:
/// - Adds:
///   * Tags::Interface<Tags::InternalDirections<Dim>,
///                     typename flux_comm_types::normal_dot_fluxes_tag>
///   * Tags::Interface<Tags::BoundaryDirectionsInterior<Dim>,
///                     typename flux_comm_types::normal_dot_fluxes_tag>
///   * Tags::Interface<Tags::BoundaryDirectionsExterior<Dim>,
///                     typename flux_comm_types::normal_dot_fluxes_tag>
/// - Removes: nothing
/// - Modifies: nothing
template <typename Metavariables>
struct InitializeFluxes {
 private:
  static constexpr size_t dim = Metavariables::system::volume_dim;
  using flux_comm_types = dg::FluxCommunicationTypes<Metavariables>;
  using normal_dot_fluxes_tag = typename flux_comm_types::normal_dot_fluxes_tag;

  // For nonconservative systems, add simple interface tags for the
  // normal-dot-fluxes. These will be updated manually during the simulation.
  template <typename LocalSystem, bool IsInFluxConservativeForm =
                                      LocalSystem::is_in_flux_conservative_form>
  struct Impl {
    template <typename TagsList>
    static auto initialize(db::DataBox<TagsList>&& box) noexcept {
      using simple_tags = db::AddSimpleTags<
          ::Tags::Interface<::Tags::InternalDirections<dim>,
                            normal_dot_fluxes_tag>,
          ::Tags::Interface<::Tags::BoundaryDirectionsInterior<dim>,
                            normal_dot_fluxes_tag>,
          ::Tags::Interface<::Tags::BoundaryDirectionsExterior<dim>,
                            normal_dot_fluxes_tag>>;
      using compute_tags = db::AddComputeTags<>;

      const auto& internal_directions =
          db::get<::Tags::InternalDirections<dim>>(box);
      const auto& boundary_directions =
          db::get<::Tags::BoundaryDirectionsInterior<dim>>(box);
      const auto& interface_meshes =
          db::get<::Tags::Interface<::Tags::InternalDirections<dim>,
                                    ::Tags::Mesh<dim - 1>>>(box);
      const auto& boundary_meshes =
          db::get<::Tags::Interface<::Tags::BoundaryDirectionsInterior<dim>,
                                    ::Tags::Mesh<dim - 1>>>(box);

      db::item_type<::Tags::Interface<::Tags::InternalDirections<dim>,
                                      normal_dot_fluxes_tag>>
          normal_dot_fluxes_interface{};
      for (const auto& direction : internal_directions) {
        const auto& interface_num_points =
            interface_meshes.at(direction).number_of_grid_points();
        normal_dot_fluxes_interface[direction].initialize(interface_num_points,
                                                          0.);
      }

      db::item_type<::Tags::Interface<::Tags::BoundaryDirectionsInterior<dim>,
                                      normal_dot_fluxes_tag>>
          normal_dot_fluxes_boundary_exterior{},
          normal_dot_fluxes_boundary_interior{};
      for (const auto& direction : boundary_directions) {
        const auto& boundary_num_points =
            boundary_meshes.at(direction).number_of_grid_points();
        normal_dot_fluxes_boundary_exterior[direction].initialize(
            boundary_num_points, 0.);
        normal_dot_fluxes_boundary_interior[direction].initialize(
            boundary_num_points, 0.);
      }

      return ::Initialization::merge_into_databox<InitializeFluxes, simple_tags,
                                                  compute_tags>(
          std::move(box), std::move(normal_dot_fluxes_interface),
          std::move(normal_dot_fluxes_boundary_interior),
          std::move(normal_dot_fluxes_boundary_exterior));
    }
  };

  // For conservative systems, add compute items for the normal-dot-fluxes.
  template <typename LocalSystem>
  struct Impl<LocalSystem, true> {
    using char_speed_tag = typename LocalSystem::char_speeds_tag;
    using variables_tag = typename LocalSystem::variables_tag;

    using simple_tags = db::AddSimpleTags<>;
    using compute_tags = db::AddComputeTags<
        ::Tags::Slice<::Tags::InternalDirections<dim>,
                      db::add_tag_prefix<::Tags::Flux, variables_tag,
                                         tmpl::size_t<dim>, Frame::Inertial>>,
        ::Tags::InterfaceComputeItem<
            ::Tags::InternalDirections<dim>,
            ::Tags::ComputeNormalDotFlux<variables_tag, dim, Frame::Inertial>>,
        ::Tags::InterfaceComputeItem<::Tags::InternalDirections<dim>,
                                     char_speed_tag>,
        ::Tags::Slice<::Tags::BoundaryDirectionsInterior<dim>,
                      db::add_tag_prefix<::Tags::Flux, variables_tag,
                                         tmpl::size_t<dim>, Frame::Inertial>>,
        ::Tags::InterfaceComputeItem<
            ::Tags::BoundaryDirectionsInterior<dim>,
            ::Tags::ComputeNormalDotFlux<variables_tag, dim, Frame::Inertial>>,
        ::Tags::InterfaceComputeItem<::Tags::BoundaryDirectionsInterior<dim>,
                                     char_speed_tag>,
        ::Tags::Slice<::Tags::BoundaryDirectionsExterior<dim>,
                      db::add_tag_prefix<::Tags::Flux, variables_tag,
                                         tmpl::size_t<dim>, Frame::Inertial>>,
        ::Tags::InterfaceComputeItem<
            ::Tags::BoundaryDirectionsExterior<dim>,
            ::Tags::ComputeNormalDotFlux<variables_tag, dim, Frame::Inertial>>,
        ::Tags::InterfaceComputeItem<::Tags::BoundaryDirectionsExterior<dim>,
                                     char_speed_tag>>;

    template <typename TagsList>
    static auto initialize(db::DataBox<TagsList>&& box) noexcept {
      return ::Initialization::merge_into_databox<InitializeFluxes, simple_tags,
                                                  compute_tags>(std::move(box));
    }
  };

 public:
  template <typename DbTagsList, typename... InboxTags, typename ArrayIndex,
            typename ActionList, typename ParallelComponent>
  static auto apply(db::DataBox<DbTagsList>& box,
                    const tuples::TaggedTuple<InboxTags...>& /*inboxes*/,
                    const Parallel::ConstGlobalCache<Metavariables>& /*cache*/,
                    const ArrayIndex& /*array_index*/, ActionList /*meta*/,
                    const ParallelComponent* const /*meta*/) noexcept {
    return std::make_tuple(
        Impl<typename Metavariables::system>::initialize(std::move(box)));
  }
};
}  // namespace Actions
}  // namespace dg
