// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>

#include "DataStructures/Tensor/TypeAliases.hpp"
#include "Evolution/Systems/GeneralizedHarmonic/ConstraintDamping/Tags.hpp"
#include "Evolution/Systems/GeneralizedHarmonic/Tags.hpp"
#include "ParallelAlgorithms/Interpolation/Protocols/ComputeVarsToInterpolate.hpp"
#include "PointwiseFunctions/GeneralRelativity/Tags.hpp"
#include "Utilities/TMPL.hpp"

/// \cond
class DataVector;
template <size_t Dim>
class Mesh;
template <typename VariablesTags>
class Variables;
namespace gsl {
template <typename T>
class not_null;
}  // namespace gsl
/// \endcond

namespace ah {

/// Given the generalized harmonic variables in the volume, computes
/// the quantities that will be interpolated onto an excision boundary.
///
/// This is meant to be the primary `compute_vars_to_interpolate`
/// for the computation of the characteristic speeds on the
/// excision boundary.
///
/// SrcTagList and DestTagList have limited flexibility, and their
/// restrictions are static_asserted inside the apply functions.  The
/// lack of complete flexibility is intentional, because most
/// computations (e.g. for observers) should be done only on the
/// horizon surface (i.e. after interpolation) as opposed to in the
/// volume; only those computations that require data in the volume
/// (e.g. volume numerical derivatives) should be done here.
///
/// For the dual-frame case, numerical derivatives of Jacobians are
/// taken in order to avoid Hessians.
///
/// SrcTagList is usually `interpolator_source_vars` in the
/// Metavariables, and the allowed and required tags in SrcTagList are
/// given by the type aliases `allowed_src_tags` and `required_src_tags`
/// below.
///
/// DestTagList is usually `vars_to_interpolate_to_target` in the
/// `InterpolationTarget` that uses `ComputeExcisionBoundaryVolumeQuantities`.
/// The allowed and required tags in DestTagList are given by
/// the type aliases `allowed_dest_tags` and `required_dest_tags` below.
struct ComputeExcisionBoundaryVolumeQuantities
    : tt::ConformsTo<intrp::protocols::ComputeVarsToInterpolate> {
  /// Single-frame case
  template <typename SrcTagList, typename DestTagList>
  static void apply(const gsl::not_null<Variables<DestTagList>*> target_vars,
                    const Variables<SrcTagList>& src_vars, const Mesh<3>& mesh);
  /// Dual-frame case
  template <typename SrcTagList, typename DestTagList, typename TargetFrame>
  static void apply(
      const gsl::not_null<Variables<DestTagList>*> target_vars,
      const Variables<SrcTagList>& src_vars, const Mesh<3>& mesh,
      const Jacobian<DataVector, 3, TargetFrame, Frame::Inertial>&
          jac_target_to_inertial,
      const InverseJacobian<DataVector, 3, TargetFrame, Frame::Inertial>&
          invjac_target_to_inertial,
      const Jacobian<DataVector, 3, Frame::ElementLogical, TargetFrame>&
          jac_logical_to_target,
      const InverseJacobian<DataVector, 3, Frame::ElementLogical, TargetFrame>&
          invjac_logical_to_target,
      const tnsr::I<DataVector, 3, Frame::Inertial>& inertial_mesh_velocity);

  using allowed_src_tags = tmpl::list<
      gr::Tags::SpacetimeMetric<3, Frame::Inertial>,
      GeneralizedHarmonic::Tags::Pi<3, Frame::Inertial>,
      GeneralizedHarmonic::Tags::Phi<3, Frame::Inertial>,
      ::Tags::deriv<GeneralizedHarmonic::Tags::Phi<3, Frame::Inertial>,
                    tmpl::size_t<3>, Frame::Inertial>,
      GeneralizedHarmonic::ConstraintDamping::Tags::ConstraintGamma1>;

  using required_src_tags =
      tmpl::list<gr::Tags::SpacetimeMetric<3, Frame::Inertial>>;

  template <typename TargetFrame>
  using allowed_dest_tags_target_frame = tmpl::list<
      gr::Tags::SpatialMetric<3, Frame::Inertial>,
      gr::Tags::SpatialMetric<3, Frame::Grid>,
      gr::Tags::SpacetimeMetric<3, Frame::Inertial>,
      gr::Tags::Lapse<DataVector>, gr::Tags::Shift<3, Frame::Inertial>,
      gr::Tags::Shift<3, Frame::Grid>,
      GeneralizedHarmonic::ConstraintDamping::Tags::ConstraintGamma1>;

  template <typename TargetFrame>
  using allowed_dest_tags = tmpl::remove_duplicates<
      tmpl::append<allowed_dest_tags_target_frame<TargetFrame>,
                   allowed_dest_tags_target_frame<Frame::Inertial>>>;

  template <typename TargetFrame>
  using required_dest_tags = tmpl::list<>;
};

}  // namespace ah
