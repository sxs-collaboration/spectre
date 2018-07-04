// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <array>
#include <cstddef>
#include <tuple>
#include <utility>  // IWYU pragma: keep
#include <vector>

#include "DataStructures/DataBox/DataBox.hpp"
#include "DataStructures/DataBox/DataBoxTag.hpp"
#include "DataStructures/DataBox/Prefixes.hpp"
#include "DataStructures/Tensor/EagerMath/Magnitude.hpp"  // IWYU pragma: keep
#include "DataStructures/Variables.hpp"                   // IWYU pragma: keep
#include "Domain/Mesh.hpp"
// IWYU pragma: no_include "DataStructures/VariablesHelpers.hpp"
#include "Domain/CreateInitialElement.hpp"
#include "Domain/Domain.hpp"
#include "Domain/Element.hpp"
#include "Domain/ElementId.hpp"
#include "Domain/ElementMap.hpp"
#include "Domain/FaceNormal.hpp"
#include "Domain/Tags.hpp"
#include "Evolution/Conservative/Tags.hpp"  // IWYU pragma: keep
#include "NumericalAlgorithms/DiscontinuousGalerkin/FluxCommunicationTypes.hpp"
#include "NumericalAlgorithms/DiscontinuousGalerkin/Tags.hpp"
#include "NumericalAlgorithms/LinearOperators/Divergence.hpp"  // IWYU pragma: keep
#include "NumericalAlgorithms/LinearOperators/PartialDerivatives.hpp"
#include "NumericalAlgorithms/Spectral/Spectral.hpp"
#include "Parallel/ConstGlobalCache.hpp"
#include "PointwiseFunctions/AnalyticSolutions/Tags.hpp"
// IWYU pragma: no_include "Time/Slab.hpp"
#include "Time/Tags.hpp"
#include "Time/Time.hpp"
#include "Time/TimeId.hpp"
#include "Utilities/TMPL.hpp"

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
///   * CacheTags::TimeStepper
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
///   * Tags::deriv<System::variables_tag::tags_list,
///                 System::gradients_tags,
///                 Tags::InverseJacobian<Tags::ElementMap<Dim>,
///                                       Tags::LogicalCoordinates<Dim>>>
///   * db::add_tag_prefix<Tags::dt, System::variables_tag>
///   * Tags::UnnormalizedFaceNormal<Dim>
/// - Removes: nothing
/// - Modifies: nothing
template <size_t Dim>
struct InitializeElement {
  // Items related to the basic structure of the domain
  struct DomainTags {
    using simple_tags =
        db::AddSimpleTags<Tags::Mesh<Dim>, Tags::Element<Dim>,
                          Tags::ElementMap<Dim>>;

    using compute_tags = db::AddComputeTags<
        Tags::LogicalCoordinates<Dim>,
        Tags::Coordinates<Tags::ElementMap<Dim>, Tags::LogicalCoordinates<Dim>>,
        Tags::InverseJacobian<Tags::ElementMap<Dim>,
                              Tags::LogicalCoordinates<Dim>>>;

    template <typename TagsList>
    static auto initialize(
        db::DataBox<TagsList>&& box, const ElementIndex<Dim>& array_index,
        const std::vector<std::array<size_t, Dim>>& initial_extents,
        const Domain<Dim, Frame::Inertial>& domain) noexcept {
      const ElementId<Dim> element_id{array_index};
      const auto& my_block = domain.blocks()[element_id.block_id()];
      Mesh<Dim> mesh{initial_extents[element_id.block_id()],
                     Spectral::Basis::Legendre,
                     Spectral::Quadrature::GaussLobatto};
      Element<Dim> element = create_initial_element(element_id, my_block);
      ElementMap<Dim, Frame::Inertial> map{
          element_id, my_block.coordinate_map().get_clone()};

      return db::create_from<db::RemoveTags<>, simple_tags, compute_tags>(
          std::move(box), std::move(mesh), std::move(element), std::move(map));
    }
  };

  // Tags related only to the system
  template <typename System, bool IsConservative = System::is_conservative>
  struct SystemTags {
    using simple_tags = db::AddSimpleTags<typename System::variables_tag>;

    using compute_tags = db::AddComputeTags<>;

    template <typename TagsList, typename Metavariables>
    static auto initialize(
        db::DataBox<TagsList>&& box,
        const Parallel::ConstGlobalCache<Metavariables>& cache,
        const Time& initial_time) noexcept {
      using Vars = typename System::variables_tag::type;

      const size_t num_grid_points =
          db::get<Tags::Mesh<Dim>>(box).number_of_grid_points();
      const auto& inertial_coords =
          db::get<Tags::Coordinates<Tags::ElementMap<Dim>,
                                    Tags::LogicalCoordinates<Dim>>>(box);

      // Set initial data from analytic solution
      using solution_tag = CacheTags::AnalyticSolutionBase;
      Vars vars{num_grid_points};
      vars.assign_subset(Parallel::get<solution_tag>(cache).variables(
          inertial_coords, initial_time.value(), typename Vars::tags_list{}));

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
        const Time& initial_time) noexcept {
      using Vars = typename System::variables_tag::type;

      const size_t num_grid_points =
          db::get<Tags::Mesh<Dim>>(box).number_of_grid_points();
      const auto& inertial_coords =
          db::get<Tags::Coordinates<Tags::ElementMap<Dim>,
                                    Tags::LogicalCoordinates<Dim>>>(box);

      // Set initial data from analytic solution
      using solution_tag = CacheTags::AnalyticSolutionBase;
      Vars vars{num_grid_points};
      vars.assign_subset(Parallel::get<solution_tag>(cache).variables(
          inertial_coords, initial_time.value(), typename Vars::tags_list{}));

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
    using simple_tags = db::AddSimpleTags<>;

    template <typename Directions>
    using face_tags = tmpl::list<
        Directions, Tags::Interface<Directions, Tags::Direction<Dim>>,
        Tags::Interface<Directions, Tags::Mesh<Dim - 1>>,
        Tags::Interface<Directions, typename System::variables_tag>,
        Tags::Interface<Directions, Tags::UnnormalizedFaceNormal<Dim>>,
        Tags::Interface<Directions, typename System::template magnitude_tag<
                                        Tags::UnnormalizedFaceNormal<Dim>>>,
        Tags::Interface<Directions,
                        Tags::Normalized<Tags::UnnormalizedFaceNormal<Dim>>>>;

    using compute_tags = tmpl::append<face_tags<Tags::InternalDirections<Dim>>,
                                      face_tags<Tags::BoundaryDirections<Dim>>>;

    template <typename TagsList>
    static auto initialize(db::DataBox<TagsList>&& box) noexcept {
      return db::create_from<db::RemoveTags<>, simple_tags, compute_tags>(
          std::move(box));
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
              bool IsConservative = LocalSystem::is_conservative>
    struct ComputeTags {
      using type = db::AddComputeTags<
          Tags::Time,
          Tags::deriv<typename variables_tag::tags_list,
                      typename System::gradients_tags,
                      Tags::InverseJacobian<Tags::ElementMap<Dim>,
                                            Tags::LogicalCoordinates<Dim>>>>;
    };

    template <typename LocalSystem>
    struct ComputeTags<LocalSystem, true> {
      using type = db::AddComputeTags<
          Tags::Time,
          Tags::ComputeDiv<
              db::add_tag_prefix<Tags::Flux, variables_tag, tmpl::size_t<Dim>,
                                 Frame::Inertial>,
              Tags::InverseJacobian<Tags::ElementMap<Dim>,
                                    Tags::LogicalCoordinates<Dim>>>>;
    };

    using compute_tags = typename ComputeTags<System>::type;

    template <typename TagsList, typename Metavariables>
    static auto initialize(
        db::DataBox<TagsList>&& box,
        const Parallel::ConstGlobalCache<Metavariables>& cache,
        const Time& initial_time, const TimeDelta& initial_dt) noexcept {
      using Vars = typename variables_tag::type;
      using DtVars = typename dt_variables_tag::type;

      const auto& time_stepper = Parallel::get<CacheTags::TimeStepper>(cache);

      const TimeId time_id(initial_dt.is_positive(), 0, initial_time);
      TimeId next_time_id = time_stepper.next_time_id(time_id, initial_dt);
      if (next_time_id.is_at_slab_boundary()) {
        const auto next_time = next_time_id.step_time();
        next_time_id = TimeId(
            initial_dt.is_positive(), 1,
            next_time.with_slab(next_time.slab().advance_towards(initial_dt)));
      }

      const size_t num_grid_points =
          db::get<Tags::Mesh<Dim>>(box).number_of_grid_points();
      const auto& inertial_coords =
          db::get<Tags::Coordinates<Tags::ElementMap<Dim>,
                                    Tags::LogicalCoordinates<Dim>>>(box);

      // Will be overwritten before use
      DtVars dt_vars{num_grid_points};

      typename Tags::HistoryEvolvedVariables<variables_tag,
                                             dt_variables_tag>::type history;
      using solution_tag = CacheTags::AnalyticSolutionBase;
      if (not time_stepper.is_self_starting()) {
        // We currently just put initial points at past slab boundaries.
        Time past_t = initial_time;
        TimeDelta past_dt = initial_dt;
        for (size_t i = time_stepper.number_of_past_steps(); i > 0; --i) {
          past_dt = past_dt.with_slab(past_dt.slab().advance_towards(-past_dt));
          past_t -= past_dt;
          Vars hist_vars{num_grid_points};
          DtVars dt_hist_vars{num_grid_points};
          hist_vars.assign_subset(Parallel::get<solution_tag>(cache).variables(
              inertial_coords, past_t.value(), typename Vars::tags_list{}));
          dt_hist_vars.assign_subset(
              Parallel::get<solution_tag>(cache).variables(
                  inertial_coords, past_t.value(),
                  typename DtVars::tags_list{}));
          history.insert_initial(past_t, std::move(hist_vars),
                                 std::move(dt_hist_vars));
        }
      }

      return db::create_from<db::RemoveTags<>, simple_tags, compute_tags>(
          std::move(box), time_id, next_time_id, initial_dt, std::move(dt_vars),
          std::move(history));
    }
  };

  // Tags related to DG details (numerical fluxes, etc.).
  template <typename Metavariables>
  struct DgTags {
    using temporal_id_tag = typename Metavariables::temporal_id;
    using flux_comm_types = FluxCommunicationTypes<Metavariables>;

    template <typename Tag>
    using interface_tag = Tags::Interface<Tags::InternalDirections<Dim>, Tag>;

    template <typename TagsList>
    static auto add_mortar_data(db::DataBox<TagsList>&& box) noexcept {
      const auto& element = db::get<Tags::Element<Dim>>(box);

      typename flux_comm_types::simple_mortar_data_tag::type mortar_data{};
      typename Tags::Mortars<Tags::Next<temporal_id_tag>, Dim>::type
          mortar_next_temporal_ids{};
      const auto& temporal_id = get<temporal_id_tag>(box);
      for (const auto& direction_neighbors : element.neighbors()) {
        const auto& direction = direction_neighbors.first;
        const auto& neighbors = direction_neighbors.second;
        for (const auto& neighbor : neighbors) {
          const auto mortar_id = std::make_pair(direction, neighbor);
          mortar_data.insert({mortar_id, {}});
          mortar_next_temporal_ids.insert({mortar_id, temporal_id});
        }
      }

      return db::create_from<
          db::RemoveTags<>,
          db::AddSimpleTags<typename flux_comm_types::simple_mortar_data_tag,
                            Tags::Mortars<Tags::Next<temporal_id_tag>, Dim>>>(
          std::move(box), std::move(mortar_data),
          std::move(mortar_next_temporal_ids));
    }

    template <typename LocalSystem,
              bool IsConservative = LocalSystem::is_conservative>
    struct Impl {
      using simple_tags = db::AddSimpleTags<
          typename flux_comm_types::simple_mortar_data_tag,
          Tags::Mortars<Tags::Next<temporal_id_tag>, Dim>,
          interface_tag<typename flux_comm_types::normal_dot_fluxes_tag>>;

      using compute_tags = db::AddComputeTags<>;

      template <typename TagsList>
      static auto initialize(db::DataBox<TagsList>&& box) noexcept {
        auto box2 = add_mortar_data(std::move(box));

        const auto& internal_directions =
            db::get<Tags::InternalDirections<Dim>>(box2);

        typename interface_tag<
            typename flux_comm_types::normal_dot_fluxes_tag>::type
            normal_dot_fluxes{};
        for (const auto& direction : internal_directions) {
          const auto& interface_num_points =
              db::get<interface_tag<Tags::Mesh<Dim - 1>>>(box2)
                  .at(direction)
                  .number_of_grid_points();
          normal_dot_fluxes[direction].initialize(interface_num_points, 0.);
        }

        return db::create_from<
            db::RemoveTags<>,
            db::AddSimpleTags<interface_tag<
                typename flux_comm_types::normal_dot_fluxes_tag>>>(
            std::move(box2), std::move(normal_dot_fluxes));
      }
    };

    template <typename LocalSystem>
    struct Impl<LocalSystem, true> {
      using simple_tags =
          db::AddSimpleTags<typename flux_comm_types::simple_mortar_data_tag,
                            Tags::Mortars<Tags::Next<temporal_id_tag>, Dim>>;

      using compute_tags = db::AddComputeTags<
          interface_tag<db::add_tag_prefix<Tags::Flux,
                                           typename LocalSystem::variables_tag,
                                           tmpl::size_t<Dim>, Frame::Inertial>>,
          interface_tag<Tags::ComputeNormalDotFlux<
              typename LocalSystem::variables_tag, Dim, Frame::Inertial>>>;

      template <typename TagsList>
      static auto initialize(db::DataBox<TagsList>&& box) noexcept {
        return db::create_from<db::RemoveTags<>, db::AddSimpleTags<>,
                               compute_tags>(add_mortar_data(std::move(box)));
      }
    };

    using impl = Impl<typename Metavariables::system>;
    using simple_tags = typename impl::simple_tags;
    using compute_tags = typename impl::compute_tags;

    template <typename TagsList>
    static auto initialize(db::DataBox<TagsList>&& box) noexcept {
      return impl::initialize(std::move(box));
    }
  };

  template <class Metavariables>
  using return_tag_list = tmpl::append<
      typename DomainTags::simple_tags,
      typename SystemTags<typename Metavariables::system>::simple_tags,
      typename DomainInterfaceTags<typename Metavariables::system>::simple_tags,
      typename EvolutionTags<typename Metavariables::system>::simple_tags,
      typename DgTags<Metavariables>::simple_tags,
      typename DomainTags::compute_tags,
      typename SystemTags<typename Metavariables::system>::compute_tags,
      typename DomainInterfaceTags<
          typename Metavariables::system>::compute_tags,
      typename EvolutionTags<typename Metavariables::system>::compute_tags,
      typename DgTags<Metavariables>::compute_tags>;

  template <typename... InboxTags, typename Metavariables, typename ActionList,
            typename ParallelComponent>
  static auto apply(const db::DataBox<tmpl::list<>>& /*box*/,
                    const tuples::TaggedTuple<InboxTags...>& /*inboxes*/,
                    const Parallel::ConstGlobalCache<Metavariables>& cache,
                    const ElementIndex<Dim>& array_index,
                    const ActionList /*meta*/,
                    const ParallelComponent* const /*meta*/,
                    std::vector<std::array<size_t, Dim>> initial_extents,
                    Domain<Dim, Frame::Inertial> domain, Time initial_time,
                    TimeDelta initial_dt) noexcept {
    using system = typename Metavariables::system;
    auto domain_box = DomainTags::initialize(
        db::DataBox<tmpl::list<>>{}, array_index, initial_extents, domain);
    auto system_box = SystemTags<system>::initialize(std::move(domain_box),
                                                     cache, initial_time);
    auto domain_interface_box =
        DomainInterfaceTags<system>::initialize(std::move(system_box));
    auto evolution_box = EvolutionTags<system>::initialize(
        std::move(domain_interface_box), cache, initial_time, initial_dt);
    auto dg_box = DgTags<Metavariables>::initialize(std::move(evolution_box));
    return std::make_tuple(std::move(dg_box));
  }
};
}  // namespace Actions
}  // namespace dg
