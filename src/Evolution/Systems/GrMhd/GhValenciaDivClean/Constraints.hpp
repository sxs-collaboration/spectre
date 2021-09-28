// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>

#include "DataStructures/DataBox/Prefixes.hpp"
#include "DataStructures/DataBox/Tag.hpp"
#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/TypeAliases.hpp"
#include "Evolution/Systems/GeneralizedHarmonic/ConstraintDamping/Tags.hpp"
#include "Evolution/Systems/GeneralizedHarmonic/Constraints.hpp"
#include "Evolution/Systems/GeneralizedHarmonic/Tags.hpp"
#include "Evolution/Systems/GrMhd/GhValenciaDivClean/Tags.hpp"
#include "NumericalAlgorithms/LinearOperators/PartialDerivatives.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/TMPL.hpp"

namespace grmhd::GhValenciaDivClean::Tags {
/*!
 * \brief Compute item to get the F-constraint for the generalized harmonic
 * evolution system with an MHD stress-energy source.
 *
 * \details See `GeneralizedHarmonic::f_constraint()`. Can be retrieved using
 * `GeneralizedHarmonic::Tags::FConstraint`.
 */
template <size_t SpatialDim, typename Frame>
struct FConstraintCompute
    : GeneralizedHarmonic::Tags::FConstraint<SpatialDim, Frame>,
      db::ComputeTag {
  using argument_tags = tmpl::list<
      GeneralizedHarmonic::Tags::GaugeH<SpatialDim, Frame>,
      GeneralizedHarmonic::Tags::SpacetimeDerivGaugeH<SpatialDim, Frame>,
      gr::Tags::SpacetimeNormalOneForm<SpatialDim, Frame, DataVector>,
      gr::Tags::SpacetimeNormalVector<SpatialDim, Frame, DataVector>,
      gr::Tags::InverseSpatialMetric<SpatialDim, Frame, DataVector>,
      gr::Tags::InverseSpacetimeMetric<SpatialDim, Frame, DataVector>,
      GeneralizedHarmonic::Tags::Pi<SpatialDim, Frame>,
      GeneralizedHarmonic::Tags::Phi<SpatialDim, Frame>,
      ::Tags::deriv<GeneralizedHarmonic::Tags::Pi<SpatialDim, Frame>,
                    tmpl::size_t<SpatialDim>, Frame>,
      ::Tags::deriv<GeneralizedHarmonic::Tags::Phi<SpatialDim, Frame>,
                    tmpl::size_t<SpatialDim>, Frame>,
      ::GeneralizedHarmonic::ConstraintDamping::Tags::ConstraintGamma2,
      GeneralizedHarmonic::Tags::ThreeIndexConstraint<SpatialDim, Frame>,
      Tags::TraceReversedStressEnergy>;

  using return_type = tnsr::a<DataVector, SpatialDim, Frame>;

  static constexpr auto function = static_cast<void (*)(
      gsl::not_null<tnsr::a<DataVector, SpatialDim, Frame>*>,
      const tnsr::a<DataVector, SpatialDim, Frame>&,
      const tnsr::ab<DataVector, SpatialDim, Frame>&,
      const tnsr::a<DataVector, SpatialDim, Frame>&,
      const tnsr::A<DataVector, SpatialDim, Frame>&,
      const tnsr::II<DataVector, SpatialDim, Frame>&,
      const tnsr::AA<DataVector, SpatialDim, Frame>&,
      const tnsr::aa<DataVector, SpatialDim, Frame>&,
      const tnsr::iaa<DataVector, SpatialDim, Frame>&,
      const tnsr::iaa<DataVector, SpatialDim, Frame>&,
      const tnsr::ijaa<DataVector, SpatialDim, Frame>&,
      const Scalar<DataVector>&,
      const tnsr::iaa<DataVector, SpatialDim, Frame>&,
      const tnsr::aa<DataVector, SpatialDim, Frame>&)>(
      &GeneralizedHarmonic::f_constraint<SpatialDim, Frame, DataVector>);

  using base = GeneralizedHarmonic::Tags::FConstraint<SpatialDim, Frame>;
};
}  // namespace grmhd::GhValenciaDivClean::Tags
