// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <array>
#include <boost/functional/hash.hpp>
#include <cstddef>
#include <memory>
#include <optional>
#include <tuple>
#include <utility>
#include <vector>

#include "ControlSystem/Tags/FunctionsOfTimeInitialize.hpp"
#include "DataStructures/DataBox/DataBox.hpp"
#include "DataStructures/FixedHashMap.hpp"
#include "Domain/CoordinateMaps/CoordinateMap.hpp"
#include "Domain/CoordinateMaps/Identity.hpp"
#include "Domain/CoordinateMaps/Tags.hpp"
#include "Domain/CreateInitialElement.hpp"
#include "Domain/Creators/Tags/Domain.hpp"
#include "Domain/Creators/Tags/FunctionsOfTime.hpp"
#include "Domain/Creators/Tags/InitialExtents.hpp"
#include "Domain/Creators/Tags/InitialRefinementLevels.hpp"
#include "Domain/Domain.hpp"
#include "Domain/ElementMap.hpp"
#include "Domain/MinimumGridSpacing.hpp"
#include "Domain/Structure/CreateInitialMesh.hpp"
#include "Domain/Structure/Direction.hpp"
#include "Domain/Structure/Element.hpp"
#include "Domain/Structure/ElementId.hpp"
#include "Domain/Structure/MaxNumberOfNeighbors.hpp"
#include "Domain/Tags.hpp"
#include "Domain/TagsTimeDependent.hpp"
#include "Evolution/DiscontinuousGalerkin/Initialization/QuadratureTag.hpp"
#include "Evolution/DiscontinuousGalerkin/Tags/NeighborMesh.hpp"
#include "Evolution/TagsDomain.hpp"
#include "NumericalAlgorithms/Spectral/LogicalCoordinates.hpp"
#include "NumericalAlgorithms/Spectral/Mesh.hpp"
#include "Parallel/AlgorithmExecution.hpp"
#include "Parallel/GlobalCache.hpp"
#include "Parallel/Tags/ArrayIndex.hpp"
#include "ParallelAlgorithms/Initialization/MutateAssign.hpp"
#include "Utilities/Requires.hpp"
#include "Utilities/TMPL.hpp"
#include "Utilities/TaggedTuple.hpp"

/// \cond
namespace Frame {
struct Inertial;
}  // namespace Frame
/// \endcond

namespace evolution::dg::Initialization {

/// \ingroup InitializationGroup
/// \brief Initialize items related to the basic structure of the element
///
/// \details See the type aliases defined below for what items are added to the
/// GlobalCache, MutableGlobalCache, and DataBox and how they are initialized

template <size_t Dim, bool UseControlSystems = false>
struct Domain {
  /// Tags for constant items added to the GlobalCache.  These items are
  /// initialized from input file options.
  using const_global_cache_tags = tmpl::list<::domain::Tags::Domain<Dim>>;

  /// Tags for mutable items added to the MutableGlobalCache.  These items are
  /// initialized from input file options.
  using mutable_global_cache_tags = tmpl::list<tmpl::conditional_t<
      UseControlSystems, ::control_system::Tags::FunctionsOfTimeInitialize,
      ::domain::Tags::FunctionsOfTimeInitialize>>;

  /// Tags for simple DataBox items that are initialized from input file options
  using simple_tags_from_options =
      tmpl::list<::domain::Tags::InitialExtents<Dim>,
                 ::domain::Tags::InitialRefinementLevels<Dim>,
                 evolution::dg::Tags::Quadrature>;

  /// Tags for simple DataBox items that are default initialized.
  using default_initialized_simple_tags =
      tmpl::list<evolution::dg::Tags::NeighborMesh<Dim>>;

  /// Tags for items fetched by the DataBox and passed to the apply function
  using argument_tags =
      tmpl::append<const_global_cache_tags, simple_tags_from_options,
                   tmpl::list<::Parallel::Tags::ArrayIndex>>;

  /// Tags for items in the DataBox that are mutated by the apply function
  using return_tags =
      tmpl::list<::domain::Tags::Mesh<Dim>, ::domain::Tags::Element<Dim>,
                 ::domain::Tags::ElementMap<Dim, Frame::Grid>,
                 ::domain::CoordinateMaps::Tags::CoordinateMap<
                     Dim, Frame::Grid, Frame::Inertial>>;

  /// Tags for mutable DataBox items that are either default initialized or
  /// initialized by the apply function
  using simple_tags =
      tmpl::append<default_initialized_simple_tags, return_tags>;

  /// Tags for immutable DataBox items (compute items or reference items) added
  /// to the DataBox.
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

  /// Given the items fetched from a DataBox by the argument_tags, mutate
  /// the items in the DataBox corresponding to return_tags
  static void apply(
      const gsl::not_null<Mesh<Dim>*> mesh,
      const gsl::not_null<Element<Dim>*> element,
      const gsl::not_null<ElementMap<Dim, Frame::Grid>*> element_map,
      const gsl::not_null<std::unique_ptr<
          ::domain::CoordinateMapBase<Frame::Grid, Frame::Inertial, Dim>>*>
          grid_to_inertial_map,
      const ::Domain<Dim>& domain,
      const std::vector<std::array<size_t, Dim>>& initial_extents,
      const std::vector<std::array<size_t, Dim>>& initial_refinement,
      const Spectral::Quadrature& quadrature,
      const ElementId<Dim>& element_id) {
    const auto& my_block = domain.blocks()[element_id.block_id()];
    *mesh = ::domain::Initialization::create_initial_mesh(
        initial_extents, element_id, quadrature);
    *element = ::domain::Initialization::create_initial_element(
        element_id, my_block, initial_refinement);
    *element_map = ElementMap<Dim, Frame::Grid>{
        element_id, my_block.is_time_dependent()
                        ? my_block.moving_mesh_logical_to_grid_map().get_clone()
                        : my_block.stationary_map().get_to_grid_frame()};

    if (my_block.is_time_dependent()) {
      *grid_to_inertial_map =
          my_block.moving_mesh_grid_to_inertial_map().get_clone();
    } else {
      *grid_to_inertial_map =
          ::domain::make_coordinate_map_base<Frame::Grid, Frame::Inertial>(
              ::domain::CoordinateMaps::Identity<Dim>{});
    }
  }
};
}  // namespace evolution::dg::Initialization
