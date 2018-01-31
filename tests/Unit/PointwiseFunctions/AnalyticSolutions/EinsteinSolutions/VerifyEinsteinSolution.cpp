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

using VariablesTags = typelist<SpacetimeMetric, Pi, Phi, GaugeH>;
}  // namespace

// Need explicit instantiation of derivatives.

#include "NumericalAlgorithms/LinearOperators/PartialDerivatives.cpp"

template Variables<db::wrap_tags_in<Tags::deriv, VariablesTags, tmpl::size_t<3>,
                                    Frame::Inertial>>
partial_derivatives<VariablesTags, VariablesTags, 3, Frame::Inertial>(
    const Variables<VariablesTags>&, const Index<3>&,
    const Tensor<
        DataVector, tmpl::integral_list<std::int32_t, 2, 1>,
        typelist<SpatialIndex<3, UpLo::Up, Frame::Logical>,
                 SpatialIndex<3, UpLo::Lo, Frame::Inertial>>>&) noexcept;
