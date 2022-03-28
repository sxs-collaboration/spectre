// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <type_traits>

#include "DataStructures/DataBox/DataBox.hpp"
#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/TypeAliases.hpp"
#include "DataStructures/Variables.hpp"
#include "Parallel/GlobalCache.hpp"
#include "ParallelAlgorithms/Interpolation/Protocols/ComputeTargetPoints.hpp"
#include "ParallelAlgorithms/Interpolation/Protocols/ComputeVarsToInterpolate.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/ProtocolHelpers.hpp"
#include "Utilities/TMPL.hpp"

/// \cond
namespace Frame {
struct Grid;
}  // namespace Frame
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
                     Frame::Inertial>& /*jacobian*/,
      const InverseJacobian<DataVector, Dim, Frame::ElementLogical,
                            TargetFrame>&
      /*inverse_jacobian*/) {
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
}  // namespace intrp::TestHelpers
