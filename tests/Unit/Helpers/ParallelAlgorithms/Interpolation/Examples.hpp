// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <string>
#include <type_traits>

#include "ApparentHorizons/ComputeHorizonVolumeQuantities.hpp"
#include "DataStructures/DataBox/DataBox.hpp"
#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/TypeAliases.hpp"
#include "DataStructures/Variables.hpp"
#include "NumericalAlgorithms/SphericalHarmonics/Tags.hpp"
#include "Parallel/GlobalCache.hpp"
#include "ParallelAlgorithms/Interpolation/InterpolationTargetDetail.hpp"
#include "ParallelAlgorithms/Interpolation/Protocols/ComputeTargetPoints.hpp"
#include "ParallelAlgorithms/Interpolation/Protocols/ComputeVarsToInterpolate.hpp"
#include "ParallelAlgorithms/Interpolation/Protocols/InterpolationTargetTag.hpp"
#include "ParallelAlgorithms/Interpolation/Protocols/PostInterpolationCallback.hpp"
#include "PointwiseFunctions/GeneralRelativity/Tags.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/PrettyType.hpp"
#include "Utilities/ProtocolHelpers.hpp"
#include "Utilities/TMPL.hpp"

/// \cond
namespace Frame {
struct Grid;
}  // namespace Frame
namespace Tags {
struct Time;
}  // namespace Tags
/// \endcond

namespace intrp::TestHelpers {
struct FakeCacheTag : db::SimpleTag {
  using type = int;
};
struct FakeSimpleTag : db::SimpleTag {
  using type = int;
};
struct FakeComputeTag : db::ComputeTag {
  using base = FakeSimpleTag;
  using return_type = int;
  using argument_tags = tmpl::list<>;
  static void function() {}
};

/// [ComputeTargetPoints]
struct ExampleComputeTargetPoints
    : tt::ConformsTo<intrp::protocols::ComputeTargetPoints> {
  // This is not required by the protocol, but these tags will also be added to
  // the global cache
  using const_global_cache_tags = tmpl::list<FakeCacheTag>;

  using is_sequential = std::true_type;

  using frame = ::Frame::Grid;

  // These are not required by the protocol, but these tags will be added to the
  // InterpolationTarget DataBox.
  using simple_tags = tmpl::list<FakeSimpleTag>;
  using compute_tags = tmpl::list<FakeComputeTag>;

  // This is not required by the protocol, but this function can be specified
  // and will be run during the Initialization phase of the InterpolationTarget
  // parallel component
  template <typename DbTags, typename Metavariables>
  static void initialize(
      const gsl::not_null<db::DataBox<DbTags>*> /*box*/,
      const Parallel::GlobalCache<Metavariables>& /*cache*/) {
    // Initialize FakeSimpleTag here
  }

  template <typename Metavariables, typename DbTags, typename TemporalId>
  static tnsr::I<DataVector, 3, frame> points(
      const db::DataBox<DbTags>& /*box*/,
      const tmpl::type_<Metavariables>& /*meta*/,
      const TemporalId& /*temporal_id*/) {
    // This function will compute points on a given surface that we are
    // interpolating onto
    return tnsr::I<DataVector, 3, frame>{};
  }
};
/// [ComputeTargetPoints]

/// [ComputeVarsToInterpolate]
struct ExampleComputeVarsToInterpolate
    : tt::ConformsTo<intrp::protocols::ComputeVarsToInterpolate> {
  template <typename SrcTagList, typename DestTagList, size_t Dim>
  static void apply(
      const gsl::not_null<Variables<DestTagList>*> /*target_vars*/,
      const Variables<SrcTagList>& /*src_vars*/, const Mesh<Dim>& /*mesh*/) {
    // Already in the same frame so no need to switch frames
    // Do some GR calculations to get correct variables
    // Then modify target_vars
    return;
  }

  template <typename SrcTagList, typename DestTagList, size_t Dim,
            typename TargetFrame>
  static void apply(
      const gsl::not_null<Variables<DestTagList>*> /*target_vars*/,
      const Variables<SrcTagList>& /*src_vars*/, const Mesh<Dim>& /*mesh*/,
      const Jacobian<DataVector, Dim, TargetFrame,
                     Frame::Inertial>& /*jacobian_target_to_inertial*/,
      const InverseJacobian<DataVector, Dim, TargetFrame, Frame::Inertial>&
      /*inverse_jacobian_target_to_inertial*/,
      const Jacobian<DataVector, 3, Frame::ElementLogical, TargetFrame>&
      /*jac_logical_to_target*/,
      const InverseJacobian<DataVector, 3, Frame::ElementLogical, TargetFrame>&
      /*invjac_logical_to_target*/,
      const tnsr::I<DataVector, 3, Frame::Inertial>& /*inertial_mesh_velocity*/,
      const tnsr::I<DataVector, 3, TargetFrame>&
      /*grid_to_target_frame_mesh_velocity*/) {
    // Need to switch frames first
    // Do some GR calculations to get correct variables
    // Then modify target_vars
    return;
  }

  // These need to exist but don't have to contain anything
  using allowed_src_tags = tmpl::list<>;
  using required_src_tags = tmpl::list<>;
  template <typename Frame>
  using allowed_dest_tags = tmpl::list<>;
  template <typename Frame>
  using required_dest_tags = tmpl::list<>;
};
/// [ComputeVarsToInterpolate]

/// [PostInterpolationCallback]
struct ExamplePostInterpolationCallback
    : tt::ConformsTo<intrp::protocols::PostInterpolationCallback> {
  // This is not required by the protocol, but can be specified.
  static constexpr double fill_invalid_points_with = 0.0;

  // Signature 1. This bool is false if another interpolation action is called
  template <typename DbTags, typename Metavariables, typename TemporalId>
  static bool apply(
      const gsl::not_null<db::DataBox<DbTags>*> /*box*/,
      const gsl::not_null<Parallel::GlobalCache<Metavariables>*> /*cache*/,
      const TemporalId& temporal_id) {
    return intrp::InterpolationTarget_detail::get_temporal_id_value(
               temporal_id) > 1.0;
  }

  // Signature 2. This is just as an example
  template <typename DbTags, typename Metavariables, typename TemporalId>
  static void apply(const db::DataBox<DbTags>& /*box*/,
                    const Parallel::GlobalCache<Metavariables>& /*cache*/,
                    const TemporalId& /*temporal_id*/) {}

  // Signature 3. This is just as an example
  template <typename DbTags, typename Metavariables, typename TemporalId>
  static void apply(const db::DataBox<DbTags>& /*box*/,
                    Parallel::GlobalCache<Metavariables>& /*cache*/,
                    const TemporalId& /*temporal_id*/) {}
};
/// [PostInterpolationCallback]

/// [InterpolationTargetTag]
struct ExampleInterpolationTargetTag
    : tt::ConformsTo<intrp::protocols::InterpolationTargetTag> {
  using temporal_id = ::Tags::Time;

  using vars_to_interpolate_to_target = tmpl::list<
      gr::Tags::SpatialMetric<DataVector, 3, ::Frame::Grid>,
      gr::Tags::InverseSpatialMetric<DataVector, 3, ::Frame::Grid>,
      gr::Tags::ExtrinsicCurvature<DataVector, 3, ::Frame::Grid>,
      gr::Tags::SpatialChristoffelSecondKind<DataVector, 3, ::Frame::Grid>>;

  // This is not necessary to conform to the protocol, but is often used
  using compute_vars_to_interpolate = ::ah::ComputeHorizonVolumeQuantities;

  // This list will be a lot longer for apparent horizon finding
  using compute_items_on_target =
      tmpl::list<StrahlkorperTags::ThetaPhiCompute<::Frame::Grid>,
                 StrahlkorperTags::RadiusCompute<::Frame::Grid>,
                 StrahlkorperTags::RhatCompute<::Frame::Grid>>;

  using compute_target_points = ExampleComputeTargetPoints;

  using post_interpolation_callback = ExamplePostInterpolationCallback;
};
/// [InterpolationTargetTag]
}  // namespace intrp::TestHelpers
