// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <tuple>
#include <unordered_set>
#include <utility>

#include "ApparentHorizons/Strahlkorper.hpp"
#include "DataStructures/DataBox/DataBox.hpp"
#include "DataStructures/DataBox/DataBoxTag.hpp"
#include "DataStructures/IdPair.hpp"
#include "DataStructures/Tensor/EagerMath/DeterminantAndInverse.hpp"
#include "DataStructures/Tensor/TypeAliases.hpp"
#include "DataStructures/Variables.hpp"
#include "Domain/BlockId.hpp"
#include "Domain/BlockLogicalCoordinates.hpp"
#include "Domain/ElementId.hpp"
#include "Domain/ElementIndex.hpp"
#include "Domain/ElementLogicalCoordinates.hpp"
#include "Domain/Mesh.hpp"
#include "Evolution/Systems/GeneralizedHarmonic/Tags.hpp"
#include "Informer/Tags.hpp"
#include "Informer/Verbosity.hpp"
#include "NumericalAlgorithms/Interpolation/IrregularInterpolant.hpp"
#include "Parallel/Algorithm.hpp"
#include "Parallel/ConstGlobalCache.hpp"
#include "Parallel/Info.hpp"
#include "Parallel/Invoke.hpp"
#include "Parallel/Printf.hpp"
#include "PointwiseFunctions/GeneralRelativity/Christoffel.hpp"
#include "PointwiseFunctions/GeneralRelativity/ComputeGhQuantities.hpp"
#include "PointwiseFunctions/GeneralRelativity/ComputeSpacetimeQuantities.hpp"
#include "PointwiseFunctions/GeneralRelativity/GrTags.hpp"
#include "PointwiseFunctions/GeneralRelativity/IndexManipulation.hpp"
#include "Time/Time.hpp"
#include "Utilities/Literals.hpp"
#include "Utilities/Requires.hpp"
#include "Utilities/TMPL.hpp"
#include "Utilities/TaggedTuple.hpp"

/// \ingroup SurfacesGroup
namespace ah {

/// \cond
template <class Metavariables, typename AhTag>
struct Finder;

namespace Actions {
namespace Finder {
template <typename AhTag>
struct ReceiveInterpolatedVars;
}  // namespace Finder
}  // namespace Actions
/// \endcond

/// Tags describing items in a HorizonManager's DataBox.
namespace Tags {
// Number of elements on a particular HorizonManager.
struct NumElements : db::SimpleTag {
  static std::string name() noexcept { return "NumElements"; }
  using type = size_t;
};
}  // namespace Tags

/// Variables that are sent from Elements to DataInterpolators.
using vars_tags_from_element = ::Tags::Variables<
    tmpl::list<::gr::Tags::SpacetimeMetric<3>, ::GeneralizedHarmonic::Pi<3>,
               ::GeneralizedHarmonic::Phi<3>>>;

/// Variables that are sent from DataInterpolators to ah::Finders.
template <typename Frame>
using vars_tags = ::Tags::Variables<
    tmpl::list<::gr::Tags::InverseSpatialMetric<3, Frame>,
               ::gr::Tags::ExtrinsicCurvature<3, Frame>,
               ::gr::Tags::SpatialChristoffelSecondKind<3, Frame>>>;

namespace Tags {
/// DataBox tag for type that holds all volume variables at all times
/// on all elements associated with this DataInterpolator.
struct VolumeVarsInfo : db::SimpleTag {
  struct Info {
    Mesh<3> mesh;
    vars_tags<Frame::Inertial>::type vars;
  };
  using type = std::unordered_map<Time, std::unordered_map<ElementId<3>, Info>>;
  static std::string name() noexcept { return "VolumeVarsInfo"; }
};
}  // namespace Tags

/// Holds interpolated variables.
template <typename Frame>
struct InterpolatedVarsInfo {
  /// block_coord_holders holds the list of points (in block logical
  /// coordinates) that need to be interpolated onto by all
  /// HorizonManagers.  The number of interpolated points that are
  /// stored in this InterpolatedVarsInfo corresponds to a subset of
  /// the points in block_coord_holders.  Moreover, the number of
  /// interpolated points stored in this InterpolatedVarsInfo will
  /// change as more elements interpolate, and will be less than or
  /// equal to the size of 'block_coord_holders' even after all
  /// Elements have interpolated (it is usually less than, because
  /// this InterpolatedVarsInfo lives only on a single core, and this
  /// core will not have access to all the Elements).
  std::vector<
      IdPair<domain::BlockId, tnsr::I<double, 3, typename ::Frame::Logical>>>
      block_coord_holders;
  /// `vars` holds the variables on some subset of the points in
  /// `block_coord_holders`.  The grid points inside vars are indexed
  /// according to `global_offsets`.  The size of 'vars' changes as more
  /// elements interpolate.
  std::vector<typename vars_tags<Frame>::type> vars{};
  /// global_offsets[j][i] is the index into block_coord_holders that
  /// corresponds to the index `i` of the DataVector held in `vars[j]`.
  /// The size of 'global_offsets' changes as more elements
  /// interpolate.
  std::vector<std::vector<size_t>> global_offsets{};
  /// Holds the elements that have already interpolated.
  std::unordered_set<ElementId<3>> element_has_interpolated{};
  /// Once all the elements have interpolated, the data will be sent
  /// to a ah::Finder and data_has_been_sent will be set to true.
  bool data_has_been_sent{false};
};

/// Holds interpolated variables at all times for a given AhTag.
/// After a horizon has been found, the held data is cleared.
template <typename AhTag>
struct HorizonHolder {
  // At some point in the future if we want to do time interpolation
  // inside the horizon manager, Time here could go to a double.
  std::unordered_map<Time, InterpolatedVarsInfo<typename AhTag::frame>>
      interpolated_vars_info;
  std::unordered_set<Time> time_steps_where_horizon_has_been_found;
};

/// Tag for indexing a particular HorizonHolder in a TaggedTuple.
template <typename AhTag>
struct HorizonHolderTag {
  using type = HorizonHolder<AhTag>;
};

namespace horizon_holders_detail {
template <typename T>
struct wrapper {
  using type = HorizonHolderTag<T>;
};
}  // namespace horizon_holders_detail

namespace Tags {
/// DataBox tag for a TaggedTuple containing all HorizonHolders.
template <typename Metavariables>
struct HorizonHolders : db::SimpleTag {
  using type = tuples::TaggedTupleTypelist<
      tmpl::transform<typename Metavariables::horizon_tags,
                      horizon_holders_detail::wrapper<tmpl::_1>>>;
  static std::string name() noexcept { return "HorizonHolders"; }
};
}  // namespace Tags

namespace DataInterpolator_detail {
// Interpolates data onto a set of points desired by a ah::Finder.
template <typename Metavariables, typename DbTags, typename AhTag>
void interpolate_data(db::DataBox<DbTags>& box, const Time& timestep) {
  db::mutate_apply<tmpl::list<Tags::HorizonHolders<Metavariables>>,
                   tmpl::list<::Tags::Verbosity, Tags::VolumeVarsInfo>>(
      [&timestep](const gsl::not_null<
                      db::item_type<Tags::HorizonHolders<Metavariables>>*>
                      holders,
                  const db::item_type<::Tags::Verbosity>& verbose,
                  const db::item_type<Tags::VolumeVarsInfo>& volume_vars_info) {
        auto& interp_info =
            get<HorizonHolderTag<AhTag>>(*holders).interpolated_vars_info.at(
                timestep);

        if (verbose == ::Verbosity::Debug) {
          Parallel::printf(
              "### Node:%d  Proc:%d ###\n"
              "{%s,%s} : interpolate_data\n\n",
              Parallel::my_node(), Parallel::my_proc(), AhTag::label(),
              timestep);
        }

        for (const auto& volume_info_outer : volume_vars_info) {
          // Are we at the right time?
          if (volume_info_outer.first != timestep) {
            continue;
          }

          // Get list of ElementIds that have the same timestep as the
          // HorizonId, and which have not yet been interpolated.
          std::vector<ElementId<3>> element_ids;

          for (const auto& volume_info_inner : volume_info_outer.second) {
            // Have we interpolated this element before?
            if (interp_info.element_has_interpolated.find(
                    volume_info_inner.first) !=
                interp_info.element_has_interpolated.end()) {
              if (verbose == ::Verbosity::Debug) {
                Parallel::printf(
                    "### Node:%d  Proc:%d ###\n"
                    "{%s,%s}: interpolate_data: Already interpolated %s\n\n",
                    Parallel::my_node(), Parallel::my_proc(), AhTag::label(),
                    timestep, volume_info_inner.first);
              }
            } else {
              if (verbose == ::Verbosity::Debug) {
                Parallel::printf(
                    "### Node:%d  Proc:%d ###\n"
                    "{%s,%s}: interpolate_data: Will try to interpolate %s\n\n",
                    Parallel::my_node(), Parallel::my_proc(), AhTag::label(),
                    timestep, volume_info_inner.first);
              }
              interp_info.element_has_interpolated.emplace(
                  volume_info_inner.first);
              element_ids.push_back(volume_info_inner.first);
            }
          }

          // Get element logical coordinates.
          const auto element_coord_holders = element_logical_coordinates(
              element_ids, interp_info.block_coord_holders);

          // Interpolate.
          for (const auto& element_coord_pair : element_coord_holders) {
            const auto& element_id = element_coord_pair.first;
            const auto& element_coord_holder = element_coord_pair.second;
            const auto& volume_info = volume_info_outer.second.at(element_id);

            intrp::Irregular<3> interpolator(
                volume_info.mesh, element_coord_holder.element_logical_coords);
            interp_info.vars.emplace_back(
                interpolator.interpolate(volume_info.vars));
            interp_info.global_offsets.emplace_back(
                element_coord_holder.offsets);
          }
        }
      },
      make_not_null(&box));
}

// Sends interpolated data to a ah::Finder.
template <typename Metavariables, typename DbTags, typename AhTag>
void send_data_to_ah_finder(db::DataBox<DbTags>& box,
                            Parallel::ConstGlobalCache<Metavariables>& cache,
                            const Time& timestep) noexcept {
  const auto& verbose = db::get<::Tags::Verbosity>(box);
  auto& holders = db::get<Tags::HorizonHolders<Metavariables>>(box);
  auto& interp_info =
      get<HorizonHolderTag<AhTag>>(holders).interpolated_vars_info.at(timestep);

  if (verbose == ::Verbosity::Debug) {
    size_t n_pts = 0;
    for (const auto& glob_off : interp_info.global_offsets) {
      n_pts += glob_off.size();
    }
    Parallel::printf(
        "### Node:%d  Proc:%d ###\n"
        "{%s,%s}: send_data_to_ah_finder: have %d points to send\n\n",
        Parallel::my_node(), Parallel::my_proc(), AhTag::label(), timestep,
        n_pts);
  }

  if (not interp_info.global_offsets.empty()) {
    auto& receiver_proxy =
        Parallel::get_parallel_component<ah::Finder<Metavariables, AhTag>>(
            cache);
    Parallel::simple_action<Actions::Finder::ReceiveInterpolatedVars<AhTag>>(
        receiver_proxy, interp_info.vars, interp_info.global_offsets);
  }

  // Flag that the data has been sent, and clear the arrays that we
  // don't need anymore so we free up memory.
  db::mutate<Tags::HorizonHolders<Metavariables>>(
      make_not_null(&box),
      [&timestep](const gsl::not_null<
                  db::item_type<Tags::HorizonHolders<Metavariables>>*>
                      holders_l) {
        auto& interp_info_l = get<HorizonHolderTag<AhTag>>(*holders_l)
                                  .interpolated_vars_info.at(timestep);
        interp_info_l.data_has_been_sent = true;
        interp_info_l.vars.clear();
        interp_info_l.global_offsets.clear();
      });
}

// Try to interpolate the data it has, and if successful,
// send interpolated data to the horizon finder.
template <typename Metavariables, typename DbTags, typename AhTag>
void try_to_interpolate_data(db::DataBox<DbTags>& box,
                             Parallel::ConstGlobalCache<Metavariables>& cache,
                             const Time& timestep) {
  const auto& holders = db::get<Tags::HorizonHolders<Metavariables>>(box);
  const auto& interp_vars_info =
      get<HorizonHolderTag<AhTag>>(holders).interpolated_vars_info;

  // If we don't yet have any points for this HorizonInfo,
  // we should exit (we can't interpolate anyway).
  // Also, if we have already reduced data for this HorizonInfo, we should
  // exit (we don't want to send the HorizonChare the same
  // information twice).
  if (interp_vars_info.find(timestep) == interp_vars_info.end() or
      interp_vars_info.at(timestep).data_has_been_sent) {
    return;
  }

  interpolate_data<Metavariables, DbTags, AhTag>(box, timestep);

  // Reduce the horizon data only if all of the elements have interpolated
  // onto it.
  const auto& verbose = db::get<::Tags::Verbosity>(box);
  const auto& num_elements = db::get<Tags::NumElements>(box);
  if (verbose == ::Verbosity::Debug) {
    Parallel::printf(
        "### Node:%d  Proc:%d ###\n"
        "{%s,%s}: try_to_interpolate_data: did %d of %d elems\n\n",
        Parallel::my_node(), Parallel::my_proc(), AhTag::label(), timestep,
        interp_vars_info.at(timestep).element_has_interpolated.size(),
        num_elements);
  }
  if (interp_vars_info.at(timestep).element_has_interpolated.size() ==
      num_elements) {
    send_data_to_ah_finder<Metavariables, DbTags, AhTag>(box, cache, timestep);
  }
}
}  // namespace DataInterpolator_detail

namespace Actions {
namespace DataInterpolator {

/// Simply prints the number of elements it talks to.
/// Used for diagnostics.
struct PrintNumElements {
  template <
      typename DbTags, typename... InboxTags, typename Metavariables,
      typename ArrayIndex, typename ActionList, typename ParallelComponent,
      Requires<tmpl::list_contains_v<DbTags, typename Tags::NumElements>> =
          nullptr>
  static void apply(const db::DataBox<DbTags>& box,  // HorizonManager's box
                    const tuples::TaggedTuple<InboxTags...>& /*inboxes*/,
                    const Parallel::ConstGlobalCache<Metavariables>& /*cache*/,
                    const ArrayIndex& /*array_index*/,
                    const ActionList /*meta*/,
                    const ParallelComponent* const /*meta*/) noexcept {
    const auto& num_elements = db::get<Tags::NumElements>(box);
    const auto& verbose = db::get<::Tags::Verbosity>(box);
    if (verbose > ::Verbosity::Quiet) {
      Parallel::printf("Number of elements on proc %d is %d\n",
                       Parallel::my_proc(), num_elements);
    }
  }
};

/// Every Element calls this once, so the DataInterpolator knows
/// how many Elements it is talking to.
///
/// Obviously this will need to change with h-refinement AMR, since
/// the number of elements will change.
struct ReceiveNumElements {
  template <
      typename DbTags, typename... InboxTags, typename Metavariables,
      typename ArrayIndex, typename ActionList, typename ParallelComponent,
      Requires<tmpl::list_contains_v<DbTags, typename Tags::NumElements>> =
          nullptr>
  static void apply(db::DataBox<DbTags>& box,  // HorizonManager's box
                    const tuples::TaggedTuple<InboxTags...>& /*inboxes*/,
                    const Parallel::ConstGlobalCache<Metavariables>& /*cache*/,
                    const ArrayIndex& /*array_index*/,
                    const ActionList /*meta*/,
                    const ParallelComponent* const /*meta*/) noexcept {
    db::mutate<Tags::NumElements>(
        make_not_null(&box), [](const gsl::not_null<
                                 db::item_type<Tags::NumElements>*>
                                    num_elements) noexcept {
          ++(*num_elements);
        });
  }
};

struct Initialize {
  template <typename Metavariables>
  using return_tag_list =
      tmpl::list<Tags::NumElements, ::Tags::Verbosity, Tags::VolumeVarsInfo,
                 Tags::HorizonHolders<Metavariables>>;
  template <typename... InboxTags, typename Metavariables, typename ArrayIndex,
            typename ActionList, typename ParallelComponent>
  static auto apply(const db::DataBox<tmpl::list<>>& /*box*/,
                    const tuples::TaggedTuple<InboxTags...>& /*inboxes*/,
                    const Parallel::ConstGlobalCache<Metavariables>& /*cache*/,
                    const ArrayIndex& /*array_index*/,
                    const ActionList /*meta*/,
                    const ParallelComponent* const /*meta*/,
                    ::Verbosity verbosity) noexcept {
    return std::make_tuple(
        db::create<db::get_items<return_tag_list<Metavariables>>>(
            0_st, verbosity, Tags::VolumeVarsInfo::type{},
            typename Tags::HorizonHolders<Metavariables>::type{}));
  }
};

/// Receives volume data from an Element at some time step.
///
/// Stores the data and tries to interpolate it to horizon points if it can.
struct GetVolumeDataFromElement {
  template <
      typename DbTags, typename... InboxTags, typename Metavariables,
      typename ArrayIndex, typename ActionList, typename ParallelComponent,
      Requires<tmpl::list_contains_v<DbTags, typename Tags::NumElements>> =
          nullptr>
  static void apply(db::DataBox<DbTags>& box,
                    const tuples::TaggedTuple<InboxTags...>& /*inboxes*/,
                    Parallel::ConstGlobalCache<Metavariables>& cache,
                    const ArrayIndex& /*array_index*/,
                    const ActionList /*meta*/,
                    const ParallelComponent* const /*meta*/,
                    const Time& timestep, const ElementId<3>& element_id,
                    const ::Mesh<3>& mesh,
                    const vars_tags_from_element::type& vars) noexcept {
    const auto& psi = get<::gr::Tags::SpacetimeMetric<3>>(vars);
    const auto& pi = get<::GeneralizedHarmonic::Pi<3>>(vars);
    const auto& phi = get<::GeneralizedHarmonic::Phi<3>>(vars);

    db::item_type<vars_tags<Frame::Inertial>> output_vars(
        mesh.number_of_grid_points());
    auto& inv_g =
        get<::gr::Tags::InverseSpatialMetric<3, Frame::Inertial>>(output_vars);
    auto& ex_curv =
        get<::gr::Tags::ExtrinsicCurvature<3, Frame::Inertial>>(output_vars);
    auto& christ =
        get<::gr::Tags::SpatialChristoffelSecondKind<3, Frame::Inertial>>(
            output_vars);

    inv_g = determinant_and_inverse(gr::spatial_metric(psi)).second;
    const auto shift = gr::shift(psi, inv_g);
    const auto lapse = gr::lapse(shift, psi);

    ex_curv = GeneralizedHarmonic::extrinsic_curvature(
        gr::spacetime_normal_vector(lapse, shift), pi, phi);
    christ = raise_or_lower_first_index(
        gr::christoffel_first_kind(
            GeneralizedHarmonic::deriv_spatial_metric(phi)),
        inv_g);

    db::mutate<Tags::VolumeVarsInfo>(
        make_not_null(&box),
        [&timestep, &element_id, &mesh, &output_vars ](
            const gsl::not_null<db::item_type<Tags::VolumeVarsInfo>*>
                container) noexcept {

          if (container->find(timestep) == container->end()) {
            container->emplace(
                timestep,
                std::unordered_map<ElementId<3>, Tags::VolumeVarsInfo::Info>{});
          }
          container->at(timestep).emplace(std::make_pair(
              element_id,
              Tags::VolumeVarsInfo::Info{mesh, std::move(output_vars)}));
        });

    // Try to interpolate data for all AhTags
    tmpl::for_each<typename Metavariables::horizon_tags>(
        [&box, &cache, &timestep](auto x) {
          using tag = typename decltype(x)::type;
          DataInterpolator_detail::try_to_interpolate_data<Metavariables,
                                                           DbTags, tag>(
              box, cache, timestep);
        });
  }
};

/// Called by a ah::Finder to send its list of interpolation points.
///
/// Cleans up previous interpolation information, and tries to interpolate
/// onto those points.  If it has interpolated to all the points it has
/// control over, it sends the interpolated variables to the ah::Finder.
template <typename AhTag>
struct ReceiveInterpolationPoints {
  template <
      typename DbTags, typename... InboxTags, typename Metavariables,
      typename ArrayIndex, typename ActionList, typename ParallelComponent,
      Requires<tmpl::list_contains_v<DbTags, typename Tags::NumElements>> =
          nullptr>
  static void apply(
      db::DataBox<DbTags>& box,  // HorizonManager's box
      const tuples::TaggedTuple<InboxTags...>& /*inboxes*/,
      Parallel::ConstGlobalCache<Metavariables>& cache,
      const ArrayIndex& /*array_index*/, const ActionList /*meta*/,
      const ParallelComponent* const /*meta*/, const Time& timestep,
      const std::vector<IdPair<domain::BlockId,
                               tnsr::I<double, 3, typename ::Frame::Logical>>>&
          block_coord_holders) noexcept {
    db::mutate<Tags::HorizonHolders<Metavariables>>(
        make_not_null(&box),
        [timestep, &block_coord_holders](
            const gsl::not_null<
                db::item_type<Tags::HorizonHolders<Metavariables>>*>
                HorizonHolders) {
          auto& vars_info = get<HorizonHolderTag<AhTag>>(*HorizonHolders)
                                .interpolated_vars_info;

          // We are starting a new iteration at this timestep, so erase
          // all information corresponding to the previous iteration at
          // this timestep.
          vars_info.erase(timestep);

          // Now add the target interpolation points at this timestep.
          vars_info.emplace(std::make_pair(
              timestep, InterpolatedVarsInfo<typename AhTag::frame>{
                            block_coord_holders}));
        });

    DataInterpolator_detail::try_to_interpolate_data<Metavariables, DbTags,
                                                     AhTag>(box, cache,
                                                            timestep);
  }
};

/// Action called by an ah::Finder when the horizon search has converged.
///
/// This cleans up stored interpolated data and volume data that is no
/// longer needed.
template <typename AhTag>
struct HorizonHasConverged {
  template <
      typename DbTags, typename... InboxTags, typename Metavariables,
      typename ArrayIndex, typename ActionList, typename ParallelComponent,
      Requires<tmpl::list_contains_v<DbTags, typename Tags::NumElements>> =
          nullptr>
  static void apply(db::DataBox<DbTags>& box,  // HorizonManager's box
                    const tuples::TaggedTuple<InboxTags...>& /*inboxes*/,
                    const Parallel::ConstGlobalCache<Metavariables>& /*cache*/,
                    const ArrayIndex& /*array_index*/,
                    const ActionList /*meta*/,
                    const ParallelComponent* const /*meta*/,
                    const Time& timestep) noexcept {
    const auto& verbose = db::get<::Tags::Verbosity>(box);
    if (verbose > ::Verbosity::Quiet) {
      Parallel::printf(
          "### Node:%d  Proc:%d ###\n"
          "{%s,%s} : Converged!\n\n",
          Parallel::my_node(), Parallel::my_proc(), AhTag::label(), timestep);
    }

    db::mutate<Tags::HorizonHolders<Metavariables>>(
        make_not_null(&box),
        [&timestep](const gsl::not_null<
                    db::item_type<Tags::HorizonHolders<Metavariables>>*>
                        holders) {
          get<HorizonHolderTag<AhTag>>(*holders)
              .time_steps_where_horizon_has_been_found.insert(timestep);
        });

    // If we don't need any of the volume data anymore for this
    // timestep, we will remove them.
    bool this_timestep_is_done = true;
    const auto& holders = db::get<Tags::HorizonHolders<Metavariables>>(box);
    tmpl::for_each<typename Metavariables::horizon_tags>([&](auto tag) {
      using Tag = typename decltype(tag)::type;
      const auto& found = get<HorizonHolderTag<Tag>>(holders)
                              .time_steps_where_horizon_has_been_found;
      if (found.find(timestep) == found.end()) {
        this_timestep_is_done = false;
      }
    });

    // We don't need any more volume data for this timestep,
    // so remove it.
    if (this_timestep_is_done) {
      db::mutate<Tags::VolumeVarsInfo>(
          make_not_null(&box),
          [&timestep](const gsl::not_null<db::item_type<Tags::VolumeVarsInfo>*>
                          volume_vars_info) {
            for (auto it = volume_vars_info->begin();
                 it != volume_vars_info->end();) {
              if (timestep == it->first) {
                it = volume_vars_info->erase(it);
              } else {
                ++it;
              }
            }
          });
    }

    // We don't need interpolated variables for this timestep anymore,
    // so remove them.
    db::mutate<Tags::HorizonHolders<Metavariables>>(
        make_not_null(&box),
        [&timestep](const gsl::not_null<
                    db::item_type<Tags::HorizonHolders<Metavariables>>*>
                        holders_l) {
          get<HorizonHolderTag<AhTag>>(*holders_l)
              .interpolated_vars_info.erase(timestep);
        });
  }
};
}  // namespace DataInterpolator
}  // namespace Actions
}  // namespace ah
