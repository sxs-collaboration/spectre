// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <array>
#include <cmath>
#include <limits>
#include <tuple>

#include "ApparentHorizons/StrahlkorperGr.hpp"
#include "ApparentHorizons/Tags.hpp"
#include "DataStructures/DataBox/DataBox.hpp"
#include "IO/Observer/ObserverComponent.hpp"
#include "IO/Observer/ReductionActions.hpp"
#include "NumericalAlgorithms/SphericalHarmonics/Spherepack.hpp"
#include "NumericalAlgorithms/SphericalHarmonics/Strahlkorper.hpp"
#include "NumericalAlgorithms/SphericalHarmonics/Tags.hpp"
#include "Parallel/GlobalCache.hpp"
#include "Parallel/Invoke.hpp"
#include "Parallel/ParallelComponentHelpers.hpp"
#include "ParallelAlgorithms/Interpolation/InterpolationTargetDetail.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/PrettyType.hpp"

/// \cond
namespace Frame {
struct Grid;
struct Distorted;
struct Inertial;
}  // namespace Frame
/// \endcond

namespace ah {
namespace callbacks {
/*!
 * \brief Writes the center of an apparent horizon to disk in both the
 * `Frame` template parameter frame and Frame::Inertial frame. Intended to be
 * used in the `post_horizon_find_callbacks` list of an InterpolationTargetTag.
 *
 * The centers will be written to a subfile with the name
 * `/ApparentHorizons/TargetName_Centers.dat` where `TargetName` is the
 * pretty_type::name of the InterpolationTargetTag template parameter.
 *
 * The columns of the dat file are:
 * - %Time
 * - GridCenter_x
 * - GridCenter_y
 * - GridCenter_z
 * - InertialCenter_x
 * - InertialCenter_y
 * - InertialCenter_z
 *
 * The `Frame` template parameter must be either `::Frame::Grid` or
 * `::Frame::Distorted`. Even though the template parameter can be
 * `::Frame::Distorted`, we still write `GridCenter_?` because the centers of
 * the objects are the same in the Grid and Distorted frames.
 *
 * \note Requires StrahlkorperTags::Strahlkorper<Frame>
 * and StrahlkorperTags::CartesianCoords<Frame::Inertial> and
 * StrahlkorperTags::EuclideanAreaElement<Frame> to be in the DataBox
 * of the InterpolationTarget.
 */
template <typename InterpolationTargetTag, typename Frame>
struct ObserveCenters {
  // Note that we don't add a const_global_cache_tags type alias here with
  // ah::Tags::ObserveCenters because we want to use the base tag and so must be
  // agnostic to how the tag was added to the cache. We do this so anything that
  // uses ObserveCenters can control when it gets printed.

  template <typename DbTags, typename Metavariables, typename TemporalId>
  static void apply(const db::DataBox<DbTags>& box,
                    Parallel::GlobalCache<Metavariables>& cache,
                    const TemporalId& temporal_id) {
    static_assert(std::is_same_v<Frame, ::Frame::Grid> or
                      std::is_same_v<Frame, ::Frame::Distorted>,
                  "Frame must be either Grid or Distorted.");
    using HorizonTag = StrahlkorperTags::Strahlkorper<Frame>;
    using CoordsTag = StrahlkorperTags::CartesianCoords<::Frame::Inertial>;
    using EuclideanAreaElementTag =
        StrahlkorperTags::EuclideanAreaElement<Frame>;
    static_assert(db::tag_is_retrievable_v<HorizonTag, db::DataBox<DbTags>>,
                  "DataBox must contain StrahlkorperTags::Strahlkorper<Frame>");
    static_assert(db::tag_is_retrievable_v<CoordsTag, db::DataBox<DbTags>>,
                  "DataBox must contain "
                  "StrahlkorperTags::CartesianCoords<Frame::Inertial>");
    static_assert(
        db::tag_is_retrievable_v<EuclideanAreaElementTag, db::DataBox<DbTags>>,
        "DataBox must contain StrahlkorperTags::EuclideanAreaElement<Frame>");

    // Only print the centers if we want to.
    if (not Parallel::get<ah::Tags::ObserveCentersBase>(cache)) {
      return;
    }

    const auto& grid_horizon = db::get<HorizonTag>(box);
    const std::array<double, 3> grid_center = grid_horizon.physical_center();

    // computes the inertial center to be the average value of the
    // inertial coordinates over the Strahlkorper, where the average is
    // computed by a surface integral.
    // Note that we use the Euclidean area element here, since we are trying
    // to find a geometric center of a surface without regard to GR.
    const auto& inertial_coords = db::get<CoordsTag>(box);
    const auto& area_element = db::get<EuclideanAreaElementTag>(box);
    std::array<double, 3> inertial_center =
        make_array<3>(std::numeric_limits<double>::signaling_NaN());
    auto integrand = make_with_value<Scalar<DataVector>>(get(area_element), 0.);
    const double denominator = grid_horizon.ylm_spherepack().definite_integral(
        get(area_element).data());
    for (size_t i = 0; i < 3; ++i) {
      get(integrand) = get(area_element) * inertial_coords.get(i);
      gsl::at(inertial_center, i) =
          grid_horizon.ylm_spherepack().definite_integral(
              get(integrand).data()) /
          denominator;
    }

    // time, grid_x, grid_y, grid_z, inertial_x, inertial_y, inertial_z
    const auto center_tuple = std::make_tuple(
        intrp::InterpolationTarget_detail::get_temporal_id_value(temporal_id),
        grid_center[0], grid_center[1], grid_center[2], inertial_center[0],
        inertial_center[1], inertial_center[2]);

    auto& observer_writer_proxy = Parallel::get_parallel_component<
        observers::ObserverWriter<Metavariables>>(cache);

    const std::string subfile_path{"/ApparentHorizons/" +
                                   pretty_type::name<InterpolationTargetTag>() +
                                   "_Centers"};

    Parallel::threaded_action<
        observers::ThreadedActions::WriteReductionDataRow>(
        // Node 0 is always the writer
        observer_writer_proxy[0], subfile_path, legend_, center_tuple);
  }

 private:
  const static inline std::vector<std::string> legend_{
      {"Time", "GridCenter_x", "GridCenter_y", "GridCenter_z",
       "InertialCenter_x", "InertialCenter_y", "InertialCenter_z"}};
};
}  // namespace callbacks
}  // namespace ah
