// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>

/// \cond
class DataVector;
/// \endcond

#include "DataStructures/DataBox/Tag.hpp"
#include "DataStructures/Tensor/EagerMath/DeterminantAndInverse.hpp"
#include "DataStructures/Tensor/TypeAliases.hpp"
#include "Evolution/Systems/GeneralizedHarmonic/Tags.hpp"
#include "PointwiseFunctions/GeneralRelativity/Christoffel.hpp"
#include "PointwiseFunctions/GeneralRelativity/ExtrinsicCurvature.hpp"
#include "PointwiseFunctions/GeneralRelativity/GeneralizedHarmonic/DerivSpatialMetric.hpp"
#include "PointwiseFunctions/GeneralRelativity/GeneralizedHarmonic/ExtrinsicCurvature.hpp"
#include "PointwiseFunctions/GeneralRelativity/IndexManipulation.hpp"
#include "PointwiseFunctions/GeneralRelativity/Lapse.hpp"
#include "PointwiseFunctions/GeneralRelativity/Shift.hpp"
#include "PointwiseFunctions/GeneralRelativity/SpacetimeNormalVector.hpp"
#include "PointwiseFunctions/GeneralRelativity/SpatialMetric.hpp"
#include "PointwiseFunctions/GeneralRelativity/Tags.hpp"
#include "Utilities/ContainerHelpers.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/TMPL.hpp"

namespace ah {
namespace Tags {
/// @{
/// These ComputeItems are different from those used in
/// GeneralizedHarmonic evolution because these live only on the
/// intrp::Actions::ApparentHorizon DataBox, not in the volume
/// DataBox.  And these ComputeItems can do fewer allocations than the
/// volume ones, because (for example) Lapse, SpaceTimeNormalVector,
/// etc.  can be inlined instead of being allocated as a separate
/// ComputeItem.
template <size_t Dim, typename Frame>
struct InverseSpatialMetricCompute : gr::Tags::InverseSpatialMetric<Dim, Frame>,
                                     db::ComputeTag {
  using return_type = tnsr::II<DataVector, Dim, Frame>;
  static void function(
      const gsl::not_null<tnsr::II<DataVector, Dim, Frame>*> result,
      const tnsr::aa<DataVector, Dim, Frame>& psi) noexcept {
    destructive_resize_components(result, psi.begin()->size());
    *result = determinant_and_inverse(gr::spatial_metric(psi)).second;
  };
  using argument_tags = tmpl::list<gr::Tags::SpacetimeMetric<Dim, Frame>>;
  using base = gr::Tags::InverseSpatialMetric<Dim, Frame>;
};
template <size_t Dim, typename Frame>
struct ExtrinsicCurvatureCompute : gr::Tags::ExtrinsicCurvature<Dim, Frame>,
                                   db::ComputeTag {
  using return_type = tnsr::ii<DataVector, Dim, Frame>;
  static void function(
      const gsl::not_null<tnsr::ii<DataVector, Dim, Frame>*> result,
      const tnsr::aa<DataVector, Dim, Frame>& psi,
      const tnsr::aa<DataVector, Dim, Frame>& pi,
      const tnsr::iaa<DataVector, Dim, Frame>& phi,
      const tnsr::II<DataVector, Dim, Frame>& inv_g) noexcept {
    const auto shift = gr::shift(psi, inv_g);
    destructive_resize_components(result, psi.begin()->size());
    GeneralizedHarmonic::extrinsic_curvature(
        result, gr::spacetime_normal_vector(gr::lapse(shift, psi), shift), pi,
        phi);
  }
  using argument_tags = tmpl::list<gr::Tags::SpacetimeMetric<Dim, Frame>,
                                   GeneralizedHarmonic::Tags::Pi<Dim, Frame>,
                                   GeneralizedHarmonic::Tags::Phi<Dim, Frame>,
                                   gr::Tags::InverseSpatialMetric<Dim, Frame>>;
  using base = gr::Tags::ExtrinsicCurvature<Dim, Frame>;
};
template <size_t Dim, typename Frame>
struct SpatialChristoffelSecondKindCompute
    : ::gr::Tags::SpatialChristoffelSecondKind<Dim, Frame>,
      db::ComputeTag {
  using return_type = tnsr::Ijj<DataVector, Dim, Frame>;
  static void function(
      const gsl::not_null<tnsr::Ijj<DataVector, Dim, Frame>*> result,
      const tnsr::iaa<DataVector, Dim, Frame>& phi,
      const tnsr::II<DataVector, Dim, Frame>& inv_g) noexcept {
    destructive_resize_components(result, phi.begin()->size());
    raise_or_lower_first_index(
        result,
        gr::christoffel_first_kind(
            GeneralizedHarmonic::deriv_spatial_metric(phi)),
        inv_g);
  }
  using argument_tags = tmpl::list<GeneralizedHarmonic::Tags::Phi<Dim, Frame>,
                                   gr::Tags::InverseSpatialMetric<Dim, Frame>>;
  using base = ::gr::Tags::SpatialChristoffelSecondKind<Dim, Frame>;
};
/// @}
}  // namespace Tags
}  // namespace ah
