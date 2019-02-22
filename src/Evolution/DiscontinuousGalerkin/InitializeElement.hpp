// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <tuple>
#include <utility>  // IWYU pragma: keep
#include <vector>

#include "DataStructures/DataBox/DataBox.hpp"
#include "DataStructures/DataBox/DataBoxTag.hpp"
#include "DataStructures/DataBox/Prefixes.hpp"
#include "DataStructures/Index.hpp"
#include "DataStructures/Tensor/EagerMath/Magnitude.hpp"  // IWYU pragma: keep
#include "DataStructures/Variables.hpp"                   // IWYU pragma: keep
#include "Domain/CreateInitialElement.hpp"
#include "Domain/Domain.hpp"
#include "Domain/Element.hpp"
#include "Domain/ElementId.hpp"
#include "Domain/ElementMap.hpp"
#include "Domain/FaceNormal.hpp"
#include "Domain/LogicalCoordinates.hpp"  // IWYU pragma: keep
#include "Domain/Mesh.hpp"
#include "Domain/MinimumGridSpacing.hpp"
#include "Domain/OrientationMap.hpp"  // IWYU pragma: keep
#include "Domain/SizeOfElement.hpp"
#include "Domain/Tags.hpp"
#include "ErrorHandling/Assert.hpp"
#include "Evolution/Conservative/Tags.hpp"  // IWYU pragma: keep
#include "NumericalAlgorithms/DiscontinuousGalerkin/FluxCommunicationTypes.hpp"
#include "NumericalAlgorithms/DiscontinuousGalerkin/MortarHelpers.hpp"
#include "NumericalAlgorithms/DiscontinuousGalerkin/Tags.hpp"
#include "NumericalAlgorithms/LinearOperators/Divergence.hpp"  // IWYU pragma: keep
#include "NumericalAlgorithms/LinearOperators/PartialDerivatives.hpp"
#include "NumericalAlgorithms/Spectral/Spectral.hpp"
#include "Parallel/ConstGlobalCache.hpp"
#include "PointwiseFunctions/AnalyticSolutions/Tags.hpp"
#include "Time/Slab.hpp"
#include "Time/Tags.hpp"
#include "Time/Time.hpp"
#include "Time/TimeId.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/Requires.hpp"
#include "Utilities/TMPL.hpp"

// IWYU pragma: no_include "DataStructures/VariablesHelpers.hpp"

/// \cond
// IWYU pragma: no_forward_declare db::DataBox
template <size_t VolumeDim>
class ElementIndex;
namespace Frame {
struct Inertial;
}  // namespace Frame
namespace tuples {
template <typename... Tags>
class TaggedTuple;  // IWYU pragma: keep
}  // namespace tuples
/// \endcond

namespace dg {
namespace Actions {
/// \ingroup ActionsGroup
/// \ingroup DiscontinuousGalerkinGroup
/// \brief Initialize a dG element with analytic initial data
///
/// Uses:
/// - ConstGlobalCache:
///   * A tag deriving off of Cache::AnalyticSolutionBase
///   * OptionTags::TimeStepper
///
/// DataBox changes:
/// - Adds:
///   * Tags::TimeId
///   * Tags::Time
///   * Tags::TimeStep
///   * Tags::LogicalCoordinates<Dim>
///   * Tags::Mesh<Dim>
///   * Tags::Element<Dim>
///   * Tags::ElementMap<Dim>
///   * System::variables_tag
///   * Tags::HistoryEvolvedVariables<System::variables_tag,
///                  db::add_tag_prefix<Tags::dt, System::variables_tag>>
///   * Tags::Coordinates<Tags::ElementMap<Dim>,
///                       Tags::LogicalCoordinates<Dim>>
///   * Tags::InverseJacobian<Tags::ElementMap<Dim>,
///                           Tags::LogicalCoordinates<Dim>>
///   * Tags::deriv<System::gradients_tags>
///   * db::add_tag_prefix<Tags::dt, System::variables_tag>
///   * Tags::UnnormalizedFaceNormal<Dim>
/// - Removes: nothing
/// - Modifies: nothing
template <size_t Dim>
struct InitializeElement {
  static Mesh<Dim> element_mesh(
      const std::vector<std::array<size_t, Dim>>& initial_extents,
      const ElementId<Dim>& element_id,
      const OrientationMap<Dim>& orientation = {}) noexcept {
    const auto& unoriented_extents = initial_extents[element_id.block_id()];
    Index<Dim> extents;
    for (size_t i = 0; i < Dim; ++i) {
      extents[i] = gsl::at(unoriented_extents, orientation(i));
    }
    return {extents.indices(), Spectral::Basis::Legendre,
            Spectral::Quadrature::GaussLobatto};
  }

  // Items related to the basic structure of the domain
  struct DomainTags {
    using simple_tags =
        db::AddSimpleTags<Tags::Mesh<Dim>, Tags::Element<Dim>,
                          Tags::ElementMap<Dim>>;

    using compute_tags = db::AddComputeTags<
        Tags::LogicalCoordinates<Dim>,
        Tags::MappedCoordinates<Tags::ElementMap<Dim>,
                                Tags::LogicalCoordinates<Dim>>,
        Tags::InverseJacobian<Tags::ElementMap<Dim>,
                              Tags::LogicalCoordinates<Dim>>,
        Tags::MinimumGridSpacing<Dim, Frame::Inertial>>;

    template <typename TagsList>
    static auto initialize(
        db::DataBox<TagsList>&& box, const ElementIndex<Dim>& array_index,
        const std::vector<std::array<size_t, Dim>>& initial_extents,
        const Domain<Dim, Frame::Inertial>& domain) noexcept {
      const ElementId<Dim> element_id{array_index};
      const auto& my_block = domain.blocks()[element_id.block_id()];
      Mesh<Dim> mesh = element_mesh(initial_extents, element_id);
      Element<Dim> element = create_initial_element(element_id, my_block);
      ElementMap<Dim, Frame::Inertial> map{
          element_id, my_block.coordinate_map().get_clone()};

      return db::create_from<db::RemoveTags<>, simple_tags, compute_tags>(
          std::move(box), std::move(mesh), std::move(element), std::move(map));
    }
  };

  // Tags related only to the system
  template <typename System, bool IsInFluxConservativeForm =
                                 System::is_in_flux_conservative_form>
  struct SystemTags {
    using simple_tags = db::AddSimpleTags<typename System::variables_tag>;

    using compute_tags = db::AddComputeTags<>;

    template <typename TagsList, typename Metavariables>
    static auto initialize(
        db::DataBox<TagsList>&& box,
        const Parallel::ConstGlobalCache<Metavariables>& cache,
        const double initial_time) noexcept {
      using Vars = typename System::variables_tag::type;

      const size_t num_grid_points =
          db::get<Tags::Mesh<Dim>>(box).number_of_grid_points();
      const auto& inertial_coords =
          db::get<Tags::Coordinates<Dim, Frame::Inertial>>(box);

      // Set initial data from analytic solution
      using solution_tag = OptionTags::AnalyticSolutionBase;
      Vars vars{num_grid_points};
      vars.assign_subset(Parallel::get<solution_tag>(cache).variables(
          inertial_coords, initial_time, typename Vars::tags_list{}));

      return db::create_from<db::RemoveTags<>, simple_tags, compute_tags>(
          std::move(box), std::move(vars));
    }
  };

  template <typename System>
  struct SystemTags<System, true> {
    using variables_tag = typename System::variables_tag;
    using fluxes_tag = db::add_tag_prefix<Tags::Flux, variables_tag,
                                          tmpl::size_t<Dim>, Frame::Inertial>;
    using sources_tag = db::add_tag_prefix<Tags::Source, variables_tag>;

    using simple_tags =
        db::AddSimpleTags<variables_tag, fluxes_tag, sources_tag>;

    using compute_tags = db::AddComputeTags<>;

    template <typename TagsList, typename Metavariables>
    static auto initialize(
        db::DataBox<TagsList>&& box,
        const Parallel::ConstGlobalCache<Metavariables>& cache,
        const double initial_time) noexcept {
      using Vars = typename System::variables_tag::type;

      const size_t num_grid_points =
          db::get<Tags::Mesh<Dim>>(box).number_of_grid_points();
      const auto& inertial_coords =
          db::get<Tags::Coordinates<Dim, Frame::Inertial>>(box);

      // Set initial data from analytic solution
      using solution_tag = OptionTags::AnalyticSolutionBase;
      Vars vars{num_grid_points};
      vars.assign_subset(Parallel::get<solution_tag>(cache).variables(
          inertial_coords, initial_time, typename Vars::tags_list{}));

      // Will be set before use
      typename fluxes_tag::type fluxes(num_grid_points);
      typename sources_tag::type sources(num_grid_points);

      return db::create_from<db::RemoveTags<>, simple_tags, compute_tags>(
          std::move(box), std::move(vars), std::move(fluxes),
          std::move(sources));
    }
  };

  // Items related to the structure of the interfaces
  template <typename System>
  struct DomainInterfaceTags {
    using simple_tags =
        db::AddSimpleTags<Tags::Interface<Tags::BoundaryDirectionsExterior<Dim>,
                                          typename System::variables_tag>>;

    template <typename Directions>
    using face_tags = tmpl::list<
        Directions,
        Tags::InterfaceComputeItem<Directions, Tags::Direction<Dim>>,
        Tags::InterfaceComputeItem<Directions, Tags::InterfaceMesh<Dim>>,
        Tags::Slice<Directions, typename System::variables_tag>,
        Tags::InterfaceComputeItem<Directions,
                                   Tags::UnnormalizedFaceNormal<Dim>>,
        Tags::InterfaceComputeItem<Directions,
                                   typename System::template magnitude_tag<
                                       Tags::UnnormalizedFaceNormal<Dim>>>,
        Tags::InterfaceComputeItem<
            Directions, Tags::Normalized<Tags::UnnormalizedFaceNormal<Dim>>>>;

    using ext_tags = tmpl::list<
        Tags::BoundaryDirectionsExterior<Dim>,
        Tags::InterfaceComputeItem<Tags::BoundaryDirectionsExterior<Dim>,
                                   Tags::Direction<Dim>>,
        Tags::InterfaceComputeItem<Tags::BoundaryDirectionsExterior<Dim>,
                                   Tags::InterfaceMesh<Dim>>,
        Tags::InterfaceComputeItem<Tags::BoundaryDirectionsExterior<Dim>,
                                   Tags::BoundaryCoordinates<Dim>>,
        Tags::InterfaceComputeItem<Tags::BoundaryDirectionsExterior<Dim>,
                                   Tags::UnnormalizedFaceNormal<Dim>>,
        Tags::InterfaceComputeItem<Tags::BoundaryDirectionsExterior<Dim>,
                                   typename System::template magnitude_tag<
                                       Tags::UnnormalizedFaceNormal<Dim>>>,
        Tags::InterfaceComputeItem<
            Tags::BoundaryDirectionsExterior<Dim>,
            Tags::Normalized<Tags::UnnormalizedFaceNormal<Dim>>>>;

    using compute_tags =
        tmpl::append<face_tags<Tags::InternalDirections<Dim>>,
                     face_tags<Tags::BoundaryDirectionsInterior<Dim>>,
                     ext_tags>;

    template <typename TagsList>
    static auto initialize(db::DataBox<TagsList>&& box) noexcept {
      const auto& mesh = db::get<Tags::Mesh<Dim>>(box);
      std::unordered_map<Direction<Dim>,
                         db::item_type<typename System::variables_tag>>
          external_boundary_vars{};

      for (const auto& direction :
           db::get<Tags::Element<Dim>>(box).external_boundaries()) {
        external_boundary_vars[direction] =
            db::item_type<typename System::variables_tag>{
                mesh.slice_away(direction.dimension()).number_of_grid_points()};
      }

      return db::create_from<db::RemoveTags<>, simple_tags, compute_tags>(
          std::move(box), std::move(external_boundary_vars));
    }
  };

  // Tags related to time-evolution of the system.
  template <typename System>
  struct EvolutionTags {
    using variables_tag = typename System::variables_tag;
    using dt_variables_tag = db::add_tag_prefix<Tags::dt, variables_tag>;

    using simple_tags = db::AddSimpleTags<
        Tags::TimeId, Tags::Next<Tags::TimeId>, Tags::TimeStep,
        dt_variables_tag,
        Tags::HistoryEvolvedVariables<variables_tag, dt_variables_tag>>;

    template <typename LocalSystem,
              bool IsInFluxConservativeForm =
                  LocalSystem::is_in_flux_conservative_form>
    struct ComputeTags {
      using type = db::AddComputeTags<
          Tags::Time, Tags::DerivCompute<
                          variables_tag,
                          Tags::InverseJacobian<Tags::ElementMap<Dim>,
                                                Tags::LogicalCoordinates<Dim>>,
                          typename System::gradients_tags>>;
    };

    template <typename LocalSystem>
    struct ComputeTags<LocalSystem, true> {
      using type = db::AddComputeTags<
          Tags::Time,
          Tags::DivCompute<
              db::add_tag_prefix<Tags::Flux, variables_tag, tmpl::size_t<Dim>,
                                 Frame::Inertial>,
              Tags::InverseJacobian<Tags::ElementMap<Dim>,
                                    Tags::LogicalCoordinates<Dim>>>>;
    };

    using compute_tags = typename ComputeTags<System>::type;

    // Global time stepping
    template <typename Metavariables,
              Requires<not Metavariables::local_time_stepping> = nullptr>
    static TimeDelta get_initial_time_step(
        const Time& initial_time, const double initial_dt_value,
        const Parallel::ConstGlobalCache<Metavariables>& /*cache*/) noexcept {
      return (initial_dt_value > 0.0 ? 1 : -1) * initial_time.slab().duration();
    }

    // Local time stepping
    template <typename Metavariables,
              Requires<Metavariables::local_time_stepping> = nullptr>
    static TimeDelta get_initial_time_step(
        const Time& initial_time, const double initial_dt_value,
        const Parallel::ConstGlobalCache<Metavariables>& cache) noexcept {
      const auto& step_controller =
          Parallel::get<OptionTags::StepController>(cache);
      return step_controller.choose_step(initial_time, initial_dt_value);
    }

    template <typename TagsList, typename Metavariables>
    static auto initialize(
        db::DataBox<TagsList>&& box,
        const Parallel::ConstGlobalCache<Metavariables>& cache,
        const double initial_time_value, const double initial_dt_value,
        const double initial_slab_size) noexcept {
      using DtVars = typename dt_variables_tag::type;

      const size_t num_grid_points =
          db::get<Tags::Mesh<Dim>>(box).number_of_grid_points();

      // Will be overwritten before use
      DtVars dt_vars{num_grid_points};

      const bool time_runs_forward = initial_dt_value > 0.0;
      const Slab initial_slab =
          time_runs_forward ? Slab::with_duration_from_start(initial_time_value,
                                                             initial_slab_size)
                            : Slab::with_duration_to_end(initial_time_value,
                                                         initial_slab_size);
      const Time initial_time =
          time_runs_forward ? initial_slab.start() : initial_slab.end();
      const TimeDelta initial_dt =
          get_initial_time_step(initial_time, initial_dt_value, cache);

      const auto& time_stepper = Parallel::get<OptionTags::TimeStepper>(cache);

      typename Tags::HistoryEvolvedVariables<variables_tag,
                                             dt_variables_tag>::type history;
      // This is stored as Tags::Next<Tags::TimeId> and will be used
      // to update Tags::TimeId at the start of the algorithm.
      //
      // The slab number is increased in the self-start phase each
      // time one order of accuracy is obtained, and the evolution
      // proper starts with slab 0.
      const TimeId time_id =
          TimeId(time_runs_forward,
                 -static_cast<int64_t>(time_stepper.number_of_past_steps()),
                 initial_time);

      return db::create_from<db::RemoveTags<>, simple_tags, compute_tags>(
          std::move(box), TimeId{}, time_id, initial_dt, std::move(dt_vars),
          std::move(history));
    }
  };

  // Tags related to limiting
  template <typename Metavariables>
  struct LimiterTags {
    // Add the tags needed by the minmod limiter, whether or not the limiter is
    // in use. This struct will have to be generalized to handle initialization
    // of arbitrary limiters. Doing so will require more precise type aliases
    // in the limiters, and then adding these tags to (minus anything already
    // present in) the databox.
    using simple_tags = db::AddSimpleTags<>;
    using compute_tags = tmpl::list<Tags::SizeOfElement<Dim>>;

    template <typename TagsList>
    static auto initialize(db::DataBox<TagsList>&& box) noexcept {
      return db::create_from<db::RemoveTags<>, simple_tags, compute_tags>(
          std::move(box));
    }
  };

  // Tags related to DG details (numerical fluxes, etc.).
  template <typename Metavariables>
  struct DgTags {
    using temporal_id_tag = typename Metavariables::temporal_id;
    using flux_comm_types = FluxCommunicationTypes<Metavariables>;
    using mortar_data_tag = tmpl::conditional_t<
        Metavariables::local_time_stepping,
        typename flux_comm_types::local_time_stepping_mortar_data_tag,
        typename flux_comm_types::simple_mortar_data_tag>;

    template <typename Tag>
    using interface_tag = Tags::Interface<Tags::InternalDirections<Dim>, Tag>;

    template <typename Tag>
    using interior_boundary_tag =
        Tags::Interface<Tags::BoundaryDirectionsInterior<Dim>, Tag>;

    template <typename Tag>
    using external_boundary_tag =
        Tags::Interface<Tags::BoundaryDirectionsExterior<Dim>, Tag>;

    template <typename TagsList>
    static auto add_mortar_data(
        db::DataBox<TagsList>&& box,
        const std::vector<std::array<size_t, Dim>>& initial_extents) noexcept {
      const auto& element = db::get<Tags::Element<Dim>>(box);
      const auto& mesh = db::get<Tags::Mesh<Dim>>(box);

      typename mortar_data_tag::type mortar_data{};
      typename Tags::Mortars<Tags::Next<temporal_id_tag>, Dim>::type
          mortar_next_temporal_ids{};
      typename Tags::Mortars<Tags::Mesh<Dim - 1>, Dim>::type mortar_meshes{};
      typename Tags::Mortars<Tags::MortarSize<Dim - 1>, Dim>::type
          mortar_sizes{};
      const auto& temporal_id = get<Tags::Next<temporal_id_tag>>(box);
      for (const auto& direction_neighbors : element.neighbors()) {
        const auto& direction = direction_neighbors.first;
        const auto& neighbors = direction_neighbors.second;
        for (const auto& neighbor : neighbors) {
          const auto mortar_id = std::make_pair(direction, neighbor);
          mortar_data[mortar_id];  // Default initialize data
          mortar_next_temporal_ids.insert({mortar_id, temporal_id});
          mortar_meshes.emplace(
              mortar_id,
              dg::mortar_mesh(mesh.slice_away(direction.dimension()),
                              element_mesh(initial_extents, neighbor,
                                           neighbors.orientation())
                                  .slice_away(direction.dimension())));
          mortar_sizes.emplace(
              mortar_id,
              dg::mortar_size(element.id(), neighbor, direction.dimension(),
                              neighbors.orientation()));
        }
      }

      for (const auto& direction : element.external_boundaries()) {
        const auto mortar_id = std::make_pair(
            direction, ElementId<Dim>::external_boundary_id());
        mortar_data[mortar_id];
        // Since no communication needs to happen for boundary conditions,
        // the temporal id is not advanced on the boundary, so we set it equal
        // to the current temporal id in the element
        mortar_next_temporal_ids.insert({mortar_id, temporal_id});
        mortar_meshes.emplace(mortar_id,
                              mesh.slice_away(direction.dimension()));
        mortar_sizes.emplace(mortar_id,
                             make_array<Dim - 1>(Spectral::MortarSize::Full));
      }

      return db::create_from<
          db::RemoveTags<>,
          db::AddSimpleTags<mortar_data_tag,
                            Tags::Mortars<Tags::Next<temporal_id_tag>, Dim>,
                            Tags::Mortars<Tags::Mesh<Dim - 1>, Dim>,
                            Tags::Mortars<Tags::MortarSize<Dim - 1>, Dim>>>(
          std::move(box), std::move(mortar_data),
          std::move(mortar_next_temporal_ids), std::move(mortar_meshes),
          std::move(mortar_sizes));
    }

    template <typename LocalSystem,
              bool IsInFluxConservativeForm =
                  LocalSystem::is_in_flux_conservative_form>
    struct Impl {
      using simple_tags = db::AddSimpleTags<
          mortar_data_tag,
          Tags::Mortars<Tags::Next<temporal_id_tag>, Dim>,
          Tags::Mortars<Tags::Mesh<Dim - 1>, Dim>,
          Tags::Mortars<Tags::MortarSize<Dim - 1>, Dim>,
          interface_tag<typename flux_comm_types::normal_dot_fluxes_tag>,
          interior_boundary_tag<
              typename flux_comm_types::normal_dot_fluxes_tag>,
          external_boundary_tag<
              typename flux_comm_types::normal_dot_fluxes_tag>>;

      using compute_tags = db::AddComputeTags<>;

      template <typename TagsList>
      static auto initialize(db::DataBox<TagsList>&& box,
                             const std::vector<std::array<size_t, Dim>>&
                                 initial_extents) noexcept {
        auto box2 = add_mortar_data(std::move(box), initial_extents);

        const auto& internal_directions =
            db::get<Tags::InternalDirections<Dim>>(box2);

        const auto& boundary_directions =
            db::get<Tags::BoundaryDirectionsInterior<Dim>>(box2);

        typename interface_tag<
            typename flux_comm_types::normal_dot_fluxes_tag>::type
            normal_dot_fluxes_interface{};
        for (const auto& direction : internal_directions) {
          const auto& interface_num_points =
              db::get<interface_tag<Tags::Mesh<Dim - 1>>>(box2)
                  .at(direction)
                  .number_of_grid_points();
          normal_dot_fluxes_interface[direction].initialize(
              interface_num_points, 0.);
        }

        typename interior_boundary_tag<
            typename flux_comm_types::normal_dot_fluxes_tag>::type
            normal_dot_fluxes_boundary_exterior{},
            normal_dot_fluxes_boundary_interior{};
        for (const auto& direction : boundary_directions) {
          const auto& boundary_num_points =
              db::get<interior_boundary_tag<Tags::Mesh<Dim - 1>>>(box2)
                  .at(direction)
                  .number_of_grid_points();
          normal_dot_fluxes_boundary_exterior[direction].initialize(
              boundary_num_points, 0.);
          normal_dot_fluxes_boundary_interior[direction].initialize(
              boundary_num_points, 0.);
        }

        return db::create_from<
            db::RemoveTags<>,
            db::AddSimpleTags<
                interface_tag<typename flux_comm_types::normal_dot_fluxes_tag>,
                interior_boundary_tag<
                    typename flux_comm_types::normal_dot_fluxes_tag>,
                external_boundary_tag<
                    typename flux_comm_types::normal_dot_fluxes_tag>>>(
            std::move(box2), std::move(normal_dot_fluxes_interface),
            std::move(normal_dot_fluxes_boundary_interior),
            std::move(normal_dot_fluxes_boundary_exterior));
      }
    };

    template <typename LocalSystem>
    struct Impl<LocalSystem, true> {
      using simple_tags =
          db::AddSimpleTags<mortar_data_tag,
                            Tags::Mortars<Tags::Next<temporal_id_tag>, Dim>,
                            Tags::Mortars<Tags::Mesh<Dim - 1>, Dim>,
                            Tags::Mortars<Tags::MortarSize<Dim - 1>, Dim>>;

      template <typename Tag>
      using interface_compute_tag =
          Tags::InterfaceComputeItem<Tags::InternalDirections<Dim>, Tag>;

      template <typename Tag>
      using boundary_interior_compute_tag =
          Tags::InterfaceComputeItem<Tags::BoundaryDirectionsInterior<Dim>,
                                     Tag>;

      template <typename Tag>
      using boundary_exterior_compute_tag =
          Tags::InterfaceComputeItem<Tags::BoundaryDirectionsExterior<Dim>,
                                     Tag>;

      using compute_tags = db::AddComputeTags<
          Tags::Slice<Tags::InternalDirections<Dim>,
                      db::add_tag_prefix<Tags::Flux,
                                         typename LocalSystem::variables_tag,
                                         tmpl::size_t<Dim>, Frame::Inertial>>,
          interface_compute_tag<Tags::ComputeNormalDotFlux<
              typename LocalSystem::variables_tag, Dim, Frame::Inertial>>,
          Tags::Slice<Tags::BoundaryDirectionsInterior<Dim>,
                      db::add_tag_prefix<Tags::Flux,
                                           typename LocalSystem::variables_tag,
                                           tmpl::size_t<Dim>, Frame::Inertial>>,
          boundary_interior_compute_tag<Tags::ComputeNormalDotFlux<
              typename LocalSystem::variables_tag, Dim, Frame::Inertial>>,
          Tags::Slice<Tags::BoundaryDirectionsExterior<Dim>,
                      db::add_tag_prefix<Tags::Flux,
                                         typename LocalSystem::variables_tag,
                                         tmpl::size_t<Dim>, Frame::Inertial>>,
          boundary_exterior_compute_tag<Tags::ComputeNormalDotFlux<
              typename LocalSystem::variables_tag, Dim, Frame::Inertial>>>;

      template <typename TagsList>
      static auto initialize(db::DataBox<TagsList>&& box,
                             const std::vector<std::array<size_t, Dim>>&
                                 initial_extents) noexcept {
        return db::create_from<db::RemoveTags<>, db::AddSimpleTags<>,
                               compute_tags>(
            add_mortar_data(std::move(box), initial_extents));
      }
    };

    using impl = Impl<typename Metavariables::system>;
    using simple_tags = typename impl::simple_tags;
    using compute_tags = typename impl::compute_tags;

    template <typename TagsList>
    static auto initialize(
        db::DataBox<TagsList>&& box,
        const std::vector<std::array<size_t, Dim>>& initial_extents) noexcept {
      return impl::initialize(std::move(box), initial_extents);
    }
  };

  template <class Metavariables>
  using return_tag_list = tmpl::append<
      typename DomainTags::simple_tags,
      typename SystemTags<typename Metavariables::system>::simple_tags,
      typename DomainInterfaceTags<typename Metavariables::system>::simple_tags,
      typename EvolutionTags<typename Metavariables::system>::simple_tags,
      typename DgTags<Metavariables>::simple_tags,
      typename LimiterTags<Metavariables>::simple_tags,
      typename DomainTags::compute_tags,
      typename SystemTags<typename Metavariables::system>::compute_tags,
      typename DomainInterfaceTags<
          typename Metavariables::system>::compute_tags,
      typename EvolutionTags<typename Metavariables::system>::compute_tags,
      typename DgTags<Metavariables>::compute_tags,
      typename LimiterTags<Metavariables>::compute_tags>;

  template <typename... InboxTags, typename Metavariables, typename ActionList,
            typename ParallelComponent>
  static auto apply(const db::DataBox<tmpl::list<>>& /*box*/,
                    const tuples::TaggedTuple<InboxTags...>& /*inboxes*/,
                    const Parallel::ConstGlobalCache<Metavariables>& cache,
                    const ElementIndex<Dim>& array_index,
                    const ActionList /*meta*/,
                    const ParallelComponent* const /*meta*/,
                    std::vector<std::array<size_t, Dim>> initial_extents,
                    Domain<Dim, Frame::Inertial> domain,
                    const double initial_time, const double initial_dt,
                    const double initial_slab_size) noexcept {
    using system = typename Metavariables::system;
    auto domain_box = DomainTags::initialize(
        db::DataBox<tmpl::list<>>{}, array_index, initial_extents, domain);
    auto system_box = SystemTags<system>::initialize(std::move(domain_box),
                                                     cache, initial_time);
    auto domain_interface_box =
        DomainInterfaceTags<system>::initialize(std::move(system_box));
    auto evolution_box = EvolutionTags<system>::initialize(
        std::move(domain_interface_box), cache, initial_time, initial_dt,
        initial_slab_size);
    auto dg_box = DgTags<Metavariables>::initialize(std::move(evolution_box),
                                                    initial_extents);
    auto limiter_box =
        LimiterTags<Metavariables>::initialize(std::move(dg_box));

    return std::make_tuple(std::move(limiter_box));
  }
};
}  // namespace Actions
}  // namespace dg
