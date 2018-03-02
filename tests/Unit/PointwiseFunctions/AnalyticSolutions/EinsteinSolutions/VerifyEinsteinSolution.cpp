// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "tests/Unit/PointwiseFunctions/AnalyticSolutions/EinsteinSolutions/VerifyEinsteinSolution.hpp"

#include "DataStructures/Tensor/TypeAliases.hpp"
#include "Evolution/Systems/GeneralizedHarmonic/Tags.hpp"
#include "NumericalAlgorithms/LinearOperators/PartialDerivatives.hpp"
#include "PointwiseFunctions/GeneralRelativity/GrTags.hpp"
#include "Utilities/TMPL.hpp"

namespace {
// Shorter names for tags.
using SpacetimeMetric = gr::Tags::SpacetimeMetric<3, Frame::Inertial>;
using Pi = ::GeneralizedHarmonic::Pi<3, Frame::Inertial>;
using Phi = ::GeneralizedHarmonic::Phi<3, Frame::Inertial>;
using GaugeH = ::GeneralizedHarmonic::GaugeH<3, Frame::Inertial>;

using VariablesTags = tmpl::list<SpacetimeMetric, Pi, Phi, GaugeH>;
}  // namespace
