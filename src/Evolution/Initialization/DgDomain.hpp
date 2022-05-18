// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <array>
#include <cstddef>
#include <memory>
#include <tuple>
#include <utility>
#include <vector>

#include "ControlSystem/Tags/FunctionsOfTimeInitialize.hpp"
#include "DataStructures/DataBox/DataBox.hpp"
#include "Domain/CoordinateMaps/CoordinateMap.hpp"
#include "Domain/CoordinateMaps/Identity.hpp"
#include "Domain/CoordinateMaps/Tags.hpp"
#include "Domain/CreateInitialElement.hpp"
#include "Domain/Domain.hpp"
#include "Domain/ElementMap.hpp"
#include "Domain/LogicalCoordinates.hpp"
#include "Domain/MinimumGridSpacing.hpp"
#include "Domain/Structure/CreateInitialMesh.hpp"
#include "Domain/Structure/Element.hpp"
#include "Domain/Structure/ElementId.hpp"
#include "Domain/Tags.hpp"
#include "Domain/TagsTimeDependent.hpp"
#include "Evolution/DiscontinuousGalerkin/Initialization/QuadratureTag.hpp"
#include "Evolution/TagsDomain.hpp"
#include "NumericalAlgorithms/Spectral/Mesh.hpp"
#include "Parallel/GlobalCache.hpp"
#include "ParallelAlgorithms/Initialization/MutateAssign.hpp"
#include "Utilities/Requires.hpp"
#include "Utilities/TMPL.hpp"
#include "Utilities/TaggedTuple.hpp"

/// \cond
template <size_t VolumeDim>
class ElementId;
namespace Frame {
struct Inertial;
}  // namespace Frame
/// \endcond

namespace evolution {
namespace dg {
namespace Initialization {
/*!
 * \ingroup InitializationGroup
 * \brief Initialize items related to the basic structure of the element
 *
 * GlobalCache:
 * - Uses:
 *   - `domain::Tags::Domain<Dim, Frame::Inertial>`
 * DataBox:
 * - Uses:
 *   - `domain::Tags::InitialExtents<Dim>`
 *   - `domain::Tags::FunctionsOfTimeInitialize` if UseControlSystems = false,
 *      or `control_system::Tags::FunctionsOfTimeInitialize` if
 *      UseControlSystems = true
 * - Adds:
 *   - `domain::Tags::Mesh<Dim>`
 *   - `domain::Tags::Element<Dim>`
 *   - `domain::Tags::ElementMap<Dim, Frame::Inertial>`
 *   - `domain::CoordinateMaps::Tags::CoordinateMap<Dim, Frame::Grid,
 *      Frame::Inertial>`
 *   - `domain::Tags::CoordinatesMeshVelocityAndJacobiansCompute<
 *      CoordinateMap<Dim, Frame::Grid, Frame::Inertial>>`
 *   - `domain::Tags::Coordinates<Dim, Frame::ElementLogical>`
 *   - `domain::Tags::Coordinates<Dim, Frame::Grid>`
 *   - `domain::Tags::Coordinates<Dim, Frame::Inertial>`
 *   - `domain::Tags::InverseJacobian<Dim, Frame::ElementLogical, Frame::Grid>`
 *   - `domain::Tags::InverseJacobian<Dim, Frame::ElementLogical,
 *      Frame::Inertial>`
 *   - `domain::Tags::DetInvJacobian<Frame::ElementLogical, Frame::Inertial>`
 *   - `domain::Tags::MeshVelocity<Dim, Frame::Inertial>`
 *   - `domain::Tags::DivMeshVelocity`
 *   - `domain::Tags::MinimumGridSpacingCompute<Dim, Frame::Inertial>>`
 * - Removes: nothing
 * - Modifies: nothing
 *
 * \note This action relies on the `SetupDataBox` aggregated initialization
 * mechanism, so `Actions::SetupDataBox` must be present in the `Initialization`
 * phase action list prior to this action.
 * \note If OverrideCubicFunctionsOfTime == true, then cubic functions
 * of time are overriden via `read_spec_piecewise_polynomial()`
 */
template <size_t Dim, bool OverrideCubicFunctionsOfTime = false,
          bool UseControlSystems = false>
struct Domain {
  using initialization_tags =
      tmpl::list<::domain::Tags::InitialExtents<Dim>,
                 ::domain::Tags::InitialRefinementLevels<Dim>,
                 evolution::dg::Tags::Quadrature>;
  using const_global_cache_tags = tmpl::list<::domain::Tags::Domain<Dim>>;

  using mutable_global_cache_tags = tmpl::list<tmpl::conditional_t<
      UseControlSystems, ::control_system::Tags::FunctionsOfTimeInitialize,
      ::domain::Tags::FunctionsOfTimeInitialize>>;

  using simple_tags =
      tmpl::list<::domain::Tags::Mesh<Dim>, ::domain::Tags::Element<Dim>,
                 ::domain::Tags::ElementMap<Dim, Frame::Grid>,
                 ::domain::CoordinateMaps::Tags::CoordinateMap<
                     Dim, Frame::Grid, Frame::Inertial>>;

  using compute_tags = tmpl::list<
      ::domain::Tags::LogicalCoordinates<Dim>,
      // Compute tags for Frame::Grid quantities
      ::domain::Tags::MappedCoordinates<
          ::domain::Tags::ElementMap<Dim, Frame::Grid>,
          ::domain::Tags::Coordinates<Dim, Frame::ElementLogical>>,
      ::domain::Tags::InverseJacobianCompute<
          ::domain::Tags::ElementMap<Dim, Frame::Grid>,
          ::domain::Tags::Coordinates<Dim, Frame::ElementLogical>>,
      // Compute tag to retrieve functions of time from global cache.
      Parallel::Tags::FromGlobalCache<tmpl::conditional_t<
          UseControlSystems, ::control_system::Tags::FunctionsOfTimeInitialize,
          ::domain::Tags::FunctionsOfTimeInitialize>>,
      // Compute tags for Frame::Inertial quantities
      ::domain::Tags::CoordinatesMeshVelocityAndJacobiansCompute<
          ::domain::CoordinateMaps::Tags::CoordinateMap<Dim, Frame::Grid,
                                                        Frame::Inertial>>,

      ::domain::Tags::InertialFromGridCoordinatesCompute<Dim>,
      ::domain::Tags::ElementToInertialInverseJacobian<Dim>,
      ::domain::Tags::DetInvJacobianCompute<Dim, Frame::ElementLogical,
                                            Frame::Inertial>,
      ::domain::Tags::InertialMeshVelocityCompute<Dim>,
      evolution::domain::Tags::DivMeshVelocityCompute<Dim>,
      // Compute tags for other mesh quantities
      ::domain::Tags::MinimumGridSpacingCompute<Dim, Frame::Inertial>>;

  template <
      typename DataBox, typename... InboxTags, typename Metavariables,
      typename ActionList, typename ParallelComponent,
      Requires<tmpl::all<initialization_tags,
                         tmpl::bind<db::tag_is_retrievable, tmpl::_1,
                                    tmpl::pin<DataBox>>>::value> = nullptr>
  static auto apply(DataBox& box,
                    const tuples::TaggedTuple<InboxTags...>& /*inboxes*/,
                    const Parallel::GlobalCache<Metavariables>& /*cache*/,
                    const ElementId<Dim>& array_index,
                    const ActionList /*meta*/,
                    const ParallelComponent* const /*meta*/) {
    const auto& initial_extents =
        db::get<::domain::Tags::InitialExtents<Dim>>(box);
    const auto& initial_refinement =
        db::get<::domain::Tags::InitialRefinementLevels<Dim>>(box);
    const auto& domain = db::get<::domain::Tags::Domain<Dim>>(box);

    const ElementId<Dim> element_id{array_index};
    const auto& my_block = domain.blocks()[element_id.block_id()];
    Mesh<Dim> mesh = ::domain::Initialization::create_initial_mesh(
        initial_extents, element_id,
        db::get<evolution::dg::Tags::Quadrature>(box));
    Element<Dim> element = ::domain::Initialization::create_initial_element(
        element_id, my_block, initial_refinement);
    ElementMap<Dim, Frame::Grid> element_map{
        element_id, my_block.is_time_dependent()
                        ? my_block.moving_mesh_logical_to_grid_map().get_clone()
                        : my_block.stationary_map().get_to_grid_frame()};

    std::unique_ptr<
        ::domain::CoordinateMapBase<Frame::Grid, Frame::Inertial, Dim>>
        grid_to_inertial_map;
    if (my_block.is_time_dependent()) {
      grid_to_inertial_map =
          my_block.moving_mesh_grid_to_inertial_map().get_clone();
    } else {
      grid_to_inertial_map =
          ::domain::make_coordinate_map_base<Frame::Grid, Frame::Inertial>(
              ::domain::CoordinateMaps::Identity<Dim>{});
    }
    ::Initialization::mutate_assign<simple_tags>(
        make_not_null(&box), std::move(mesh), std::move(element),
        std::move(element_map), std::move(grid_to_inertial_map));

    return std::make_tuple(std::move(box));
  }

  template <
      typename DataBox, typename... InboxTags, typename Metavariables,
      typename ArrayIndex, typename ActionList, typename ParallelComponent,
      Requires<not tmpl::all<initialization_tags,
                             tmpl::bind<db::tag_is_retrievable, tmpl::_1,
                                        tmpl::pin<DataBox>>>::value> = nullptr>
  static std::tuple<DataBox&&> apply(
      DataBox& /*box*/, const tuples::TaggedTuple<InboxTags...>& /*inboxes*/,
      const Parallel::GlobalCache<Metavariables>& /*cache*/,
      const ArrayIndex& /*array_index*/, ActionList /*meta*/,
      const ParallelComponent* const /*meta*/) {
    ERROR(
        "Dependencies not fulfilled. Did you forget to terminate the phase "
        "after removing options?");
  }
};
}  // namespace Initialization
}  // namespace dg
}  // namespace evolution
