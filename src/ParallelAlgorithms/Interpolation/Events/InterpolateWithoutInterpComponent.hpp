// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <array>
#include <cstddef>
#include <limits>
#include <optional>
#include <pup.h>
#include <set>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

#include "DataStructures/DataBox/DataBox.hpp"
#include "DataStructures/DataVector.hpp"
#include "DataStructures/IdPair.hpp"
#include "DataStructures/Index.hpp"
#include "Domain/Creators/Tags/Domain.hpp"
#include "Domain/Domain.hpp"
#include "Domain/ElementLogicalCoordinates.hpp"
#include "Domain/FunctionsOfTime/FunctionOfTime.hpp"
#include "Domain/Structure/BlockId.hpp"
#include "Domain/Structure/ElementId.hpp"
#include "Domain/Tags.hpp"
#include "IO/Logging/Verbosity.hpp"
#include "NumericalAlgorithms/Interpolation/IrregularInterpolant.hpp"
#include "Options/String.hpp"
#include "Parallel/GlobalCache.hpp"
#include "Parallel/Invoke.hpp"
#include "Parallel/ParallelComponentHelpers.hpp"
#include "Parallel/Printf.hpp"
#include "ParallelAlgorithms/EventsAndTriggers/Event.hpp"
#include "ParallelAlgorithms/Interpolation/Actions/InterpolationTargetVarsFromElement.hpp"
#include "ParallelAlgorithms/Interpolation/Events/GetComputeItemsOnSource.hpp"
#include "ParallelAlgorithms/Interpolation/InterpolationTargetDetail.hpp"
#include "ParallelAlgorithms/Interpolation/PointInfoTag.hpp"
#include "ParallelAlgorithms/Interpolation/Tags.hpp"
#include "ParallelAlgorithms/Interpolation/Targets/Sphere.hpp"
#include "PointwiseFunctions/GeneralRelativity/Tags.hpp"
#include "Utilities/Algorithm.hpp"
#include "Utilities/CartesianProduct.hpp"
#include "Utilities/ConstantExpressions.hpp"
#include "Utilities/PrettyType.hpp"
#include "Utilities/Serialization/CharmPupable.hpp"
#include "Utilities/TMPL.hpp"
#include "Utilities/TypeTraits/CreateGetStaticMemberVariableOrDefault.hpp"
#include "Utilities/TypeTraits/IsA.hpp"

/// \cond
namespace Events::Tags {
template <size_t Dim>
struct ObserverMesh;
template <size_t Dim, typename Fr>
struct ObserverCoordinates;
}  // namespace Events::Tags
namespace Tags {
template <typename TagsList>
struct Variables;
}  // namespace Tags
template <size_t VolumeDim>
class ElementId;
namespace intrp {
template <typename Metavariables, typename Tag>
struct InterpolationTarget;
}  // namespace intrp
/// \endcond

namespace intrp::Events {
/// \cond
template <size_t VolumeDim, typename InterpolationTargetTag,
          typename SourceVarTags>
class InterpolateWithoutInterpComponent;
/// \endcond

/*!
 * \brief Does an interpolation onto an InterpolationTargetTag by calling
 * Actions on the InterpolationTarget component.
 *
 * \note The `intrp::TargetPoints::Sphere` target is handled specially because
 * it has the potential to be very slow due to it usually having the most points
 * out of all the stationary targets. An optimization for the future would be to
 * have each target be responsible for intelligently computing the
 * `block_logical_coordinates` for it's own points.
 */
template <size_t VolumeDim, typename InterpolationTargetTag,
          typename... SourceVarTags>
class InterpolateWithoutInterpComponent<VolumeDim, InterpolationTargetTag,
                                        tmpl::list<SourceVarTags...>>
    : public Event {
 private:
  using frame = typename InterpolationTargetTag::compute_target_points::frame;

 public:
  /// \cond
  explicit InterpolateWithoutInterpComponent(CkMigrateMessage* /*unused*/) {}
  using PUP::able::register_constructor;
  WRAPPED_PUPable_decl_template(InterpolateWithoutInterpComponent);  // NOLINT
  /// \endcond

  using options = tmpl::list<>;
  static constexpr Options::String help =
      "Does interpolation using the given InterpolationTargetTag, "
      "without an Interpolator ParallelComponent.";

  static std::string name() {
    return pretty_type::name<InterpolationTargetTag>();
  }

  InterpolateWithoutInterpComponent() = default;

  using compute_tags_for_observation_box =
      detail::get_compute_items_on_source_or_default_t<InterpolationTargetTag,
                                                       tmpl::list<>>;

  using return_tags = tmpl::list<>;
  using argument_tags =
      tmpl::list<typename InterpolationTargetTag::temporal_id,
                 Tags::InterpPointInfoBase,
                 ::Events::Tags::ObserverMesh<VolumeDim>,
                 // We always grab the DG coords because we use them to create a
                 // bounding box for the sphere target optimization. DG coords
                 // have points on the boundary, while FD coords don't. If we
                 // had used FD coords, it's possible a target point would fall
                 // between the outermost gridpoints of two elements. This point
                 // would be outside our bounding box, and thus wouldn't get
                 // interpolated to. We avoid this by always using DG coords,
                 // even if the mesh is FD.
                 domain::Tags::Coordinates<VolumeDim, frame>, SourceVarTags...>;

  template <typename ParallelComponent, typename Metavariables>
  void operator()(
      const typename InterpolationTargetTag::temporal_id::type& temporal_id,
      const typename Tags::InterpPointInfo<Metavariables>::type& point_infos,
      const Mesh<VolumeDim>& mesh,
      const tnsr::I<DataVector, VolumeDim, frame> coordinates,
      const typename SourceVarTags::type&... source_vars_input,
      Parallel::GlobalCache<Metavariables>& cache,
      const ElementId<VolumeDim>& array_index,
      const ParallelComponent* const /*meta*/,
      const ObservationValue& /*observation_value*/) const {
    const tnsr::I<DataVector, VolumeDim, frame>& all_target_points =
        get<Vars::PointInfoTag<InterpolationTargetTag, VolumeDim>>(point_infos);
    std::vector<std::optional<IdPair<
        domain::BlockId, tnsr::I<double, VolumeDim, ::Frame::BlockLogical>>>>
        block_logical_coords{};

    std::stringstream ss{};
    ss << std::setprecision(std::numeric_limits<double>::digits10 + 4)
       << std::scientific;
    const ::Verbosity verbosity = Parallel::get<intrp::Tags::Verbosity>(cache);
    const bool debug_print = verbosity >= ::Verbosity::Debug;
    if (debug_print) {
      ss << InterpolationTarget_detail::target_output_prefix<
                InterpolateWithoutInterpComponent, InterpolationTargetTag>(
                temporal_id)
         << ", " << array_index << ", ";
    }

    // The sphere target is special because we have a better idea of where the
    // points will be.
    if constexpr (tt::is_a_v<
                      TargetPoints::Sphere,
                      typename InterpolationTargetTag::compute_target_points>) {
      static_assert(VolumeDim == 3,
                    "Sphere target can only be used for VolumeDim = 3.");
      // Extremum for x, y, z, r
      std::array<std::pair<double, double>, 4> min_max_coordinates{};

      const auto& sphere =
          Parallel::get<Tags::Sphere<InterpolationTargetTag>>(cache);
      const std::array<double, 3>& center = sphere.center;

      // Calculate r^2 from center of sphere because sqrt is expensive
      DataVector radii_squared{get<0>(coordinates).size(), 0.0};
      for (size_t i = 0; i < VolumeDim; i++) {
        radii_squared += square(coordinates.get(i) - gsl::at(center, i));
      }

      // Compute min and max
      {
        const auto [min, max] = alg::minmax_element(radii_squared);
        min_max_coordinates[3].first = *min;
        min_max_coordinates[3].second = *max;
      }

      const std::set<double>& radii_of_sphere_target = sphere.radii;
      const size_t l_max = sphere.l_max;

      const size_t number_of_angular_points = (l_max + 1) * (2 * l_max + 1);
      // first size_t = position of first radius in bounds
      // second size_t = total bounds to use/check
      std::optional<std::pair<size_t, size_t>> offset_and_num_points{};

      // Have a very small buffer just in case of roundoff
      double epsilon =
          (min_max_coordinates[3].second - min_max_coordinates[3].first) *
          std::numeric_limits<double>::epsilon() * 100.0;
      size_t offset_index = 0;
      // Check if any radii of the target are within the radii of our element
      for (double radius : radii_of_sphere_target) {
        const double square_radius = square(radius);
        if (square_radius >=
                (gsl::at(min_max_coordinates, 3).first - epsilon) and
            square_radius <=
                (gsl::at(min_max_coordinates, 3).second + epsilon)) {
          if (offset_and_num_points.has_value()) {
            offset_and_num_points->second += number_of_angular_points;
          } else {
            offset_and_num_points =
                std::make_pair(offset_index * number_of_angular_points,
                               number_of_angular_points);
          }
        }
        offset_index++;
      }

      // If no radii pass through this element, there's nothing to do so return
      if (not offset_and_num_points.has_value()) {
        if (debug_print) {
          using ::operator<<;
          ss << "No radii in this element.";
          Parallel::printf("%s\n", ss.str());
        }
        return;
      }

      // Get the x,y,z bounds
      for (size_t i = 0; i < VolumeDim; i++) {
        const auto [min, max] = alg::minmax_element(coordinates.get(i));
        gsl::at(min_max_coordinates, i).first = *min;
        gsl::at(min_max_coordinates, i).second = *max;
      }

      const tnsr::I<DataVector, VolumeDim, frame> target_points_to_check{};
      // Use the offset and number of points to create a view. We assume that if
      // there are multiple radii in this element, that they are successive
      // radii. If this wasn't true, that'd be a really weird topology.
      for (size_t i = 0; i < VolumeDim; i++) {
        make_const_view(make_not_null(&target_points_to_check.get(i)),
                        all_target_points.get(i), offset_and_num_points->first,
                        offset_and_num_points->second);
      }

      // To break out of inner loop and skip the point
      bool skip_point = false;
      tnsr::I<double, VolumeDim, frame> sphere_coords_to_map{};
      // Other code below expects block_logical_coords to be sized to the total
      // number of points on the target (including all radii). By default these
      // will all be nullopt and we'll only fill the ones we need
      block_logical_coords.resize(get<0>(all_target_points).size());

      const Block<VolumeDim>& block =
          Parallel::get<domain::Tags::Domain<VolumeDim>>(cache)
              .blocks()[array_index.block_id()];

      // Now for every radius in this element, we check if their points are
      // within the x,y,z bounds of the element. If they are, map the point to
      // the block logical frame and add it to the vector f all block logical
      // coordinates.
      for (size_t index = 0; index < get<0>(target_points_to_check).size();
           index++) {
        skip_point = false;
        for (size_t i = 0; i < VolumeDim; i++) {
          const double coord = target_points_to_check.get(i)[index];
          epsilon = (gsl::at(min_max_coordinates, i).second -
                     gsl::at(min_max_coordinates, i).first) *
                    std::numeric_limits<double>::epsilon() * 100.0;
          // If a point is outside any of the bounding box, skip it
          if (coord < (gsl::at(min_max_coordinates, i).first - epsilon) or
              coord > (gsl::at(min_max_coordinates, i).second + epsilon)) {
            skip_point = true;
            break;
          }

          sphere_coords_to_map.get(i) = coord;
        }

        if (skip_point) {
          continue;
        }

        std::optional<tnsr::I<double, VolumeDim, ::Frame::BlockLogical>>
            block_coords_of_target_point{};

        if constexpr (Parallel::is_in_global_cache<
                          Metavariables, domain::Tags::FunctionsOfTime>) {
          const auto& functions_of_time =
              Parallel::get<domain::Tags::FunctionsOfTime>(cache);
          const double time =
              InterpolationTarget_detail::get_temporal_id_value(temporal_id);

          block_coords_of_target_point = block_logical_coordinates_single_point(
              sphere_coords_to_map, block, time, functions_of_time);
        } else {
          block_coords_of_target_point = block_logical_coordinates_single_point(
              sphere_coords_to_map, block);
        }

        if (block_coords_of_target_point.has_value()) {
          // Get index into vector of all grid points of the target. This is
          // just the offset + index
          block_logical_coords[offset_and_num_points->first + index] =
              make_id_pair(domain::BlockId(array_index.block_id()),
                           std::move(block_coords_of_target_point.value()));
        }
      }
    } else {
      (void)coordinates;
      block_logical_coords = InterpolationTarget_detail::block_logical_coords<
          InterpolationTargetTag>(cache, all_target_points, temporal_id);
    }

    const std::vector<ElementId<VolumeDim>> element_ids{{array_index}};
    const auto element_coord_holders =
        element_logical_coordinates(element_ids, block_logical_coords);

    if (element_coord_holders.count(array_index) == 0) {
      if (debug_print) {
        ss << "No target points in this element. Skipping.";
        Parallel::printf("%s\n", ss.str());
      }
      // There are no target points in this element, so we don't need
      // to do anything.
      return;
    }

    // There are points in this element, so interpolate to them and
    // send the interpolated data to the target.  This is done
    // in several steps:
    const auto& element_coord_holder = element_coord_holders.at(array_index);

    // 1. Get the list of variables
    Variables<typename InterpolationTargetTag::vars_to_interpolate_to_target>
        interp_vars(mesh.number_of_grid_points());

    if constexpr (InterpolationTarget_detail::has_compute_vars_to_interpolate_v<
                      InterpolationTargetTag>) {
      // 1a. Call compute_vars_to_interpolate.  Need the source in a
      // Variables, so copy the variables here.
      // This copy would be unnecessary if we passed a Variables into
      // InterpolateWithoutInterpComponent instead of passing
      // individual Tensors, which would require that this Variables is
      // something in the DataBox. (Note that
      // InterpolationTarget_detail::compute_dest_vars_from_source_vars
      // allows the source variables to be different from the
      // destination variables).
      Variables<tmpl::list<SourceVarTags...>> source_vars(
          mesh.number_of_grid_points());
      [[maybe_unused]] const auto copy_to_variables =
          [&source_vars](const auto source_var_tag_v, const auto& source_var) {
            using source_var_tag = tmpl::type_from<decltype(source_var_tag_v)>;
            get<source_var_tag>(source_vars) = source_var;
            return 0;
          };
      expand_pack(copy_to_variables(tmpl::type_<SourceVarTags>{},
                                    source_vars_input)...);

      InterpolationTarget_detail::compute_dest_vars_from_source_vars<
          InterpolationTargetTag>(make_not_null(&interp_vars), source_vars,
                                  get<domain::Tags::Domain<VolumeDim>>(cache),
                                  mesh, array_index, cache, temporal_id);
    } else {
      // 1b. There is no compute_vars_to_interpolate. So copy the
      // source vars directly into the variables.
      // This copy would be unnecessary if:
      //   - We passed a Variables into InterpolateWithoutInterpComponent
      //     instead of passing individual Tensors.
      //  and
      //   - This Variables was actually something in the DataBox.
      //  and
      //   - Either the passed-in Variables was exactly the same as
      //     InterpolationTargetTag::vars_to_interpolate_to_target,
      //     or IrregularInterpolant::interpolate had the ability to
      //     interpolate only a subset of the Variables passed into it,
      //     or IrregularInterpolant::interpolate can interpolate individual
      //     DataVectors.
      [[maybe_unused]] const auto copy_to_variables =
          [&interp_vars](const auto tensor_tag_v, const auto& tensor) {
            using tensor_tag = tmpl::type_from<decltype(tensor_tag_v)>;
            get<tensor_tag>(interp_vars) = tensor;
            return 0;
          };
      expand_pack(copy_to_variables(tmpl::type_<SourceVarTags>{},
                                    source_vars_input)...);
    }

    // 2. Set up interpolator
    intrp::Irregular<VolumeDim> interpolator(
        mesh, element_coord_holder.element_logical_coords);

    // 3. Interpolate and send interpolated data to target
    auto& receiver_proxy = Parallel::get_parallel_component<
        InterpolationTarget<Metavariables, InterpolationTargetTag>>(cache);
    Parallel::simple_action<
        Actions::InterpolationTargetVarsFromElement<InterpolationTargetTag>>(
        receiver_proxy,
        std::vector<Variables<
            typename InterpolationTargetTag::vars_to_interpolate_to_target>>(
            {interpolator.interpolate(interp_vars)}),
        block_logical_coords,
        std::vector<std::vector<size_t>>({element_coord_holder.offsets}),
        temporal_id);

    if (debug_print) {
      ss << "Sending points and vars to target.";
      Parallel::printf("%s\n", ss.str());
    }
  }

  using is_ready_argument_tags = tmpl::list<>;

  template <typename ArrayIndex, typename Component, typename Metavariables>
  bool is_ready(Parallel::GlobalCache<Metavariables>& /*cache*/,
                const ArrayIndex& /*array_index*/,
                const Component* const /*meta*/) const {
    return true;
  }

  bool needs_evolved_variables() const override { return true; }
};

/// \cond
template <size_t VolumeDim, typename InterpolationTargetTag,
          typename... SourceVarTags>
PUP::able::PUP_ID
    InterpolateWithoutInterpComponent<VolumeDim, InterpolationTargetTag,
                                      tmpl::list<SourceVarTags...>>::my_PUP_ID =
        0;  // NOLINT
/// \endcond

}  // namespace intrp::Events
