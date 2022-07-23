// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include "ApparentHorizons/TagsDeclarations.hpp"
#include "NumericalAlgorithms/LinearOperators/PartialDerivatives.hpp"
#include "NumericalAlgorithms/SphericalHarmonics/Tags.hpp"
#include "PointwiseFunctions/GeneralRelativity/Tags.hpp"
#include "Utilities/TMPL.hpp"

/// \cond
namespace Frame {
struct Grid;
struct Inertial;
}  // namespace Frame
struct DataVector;
/// \endcond

namespace ah {
template <size_t Dim>
using source_vars = tmpl::list<
    gr::Tags::SpacetimeMetric<Dim, ::Frame::Inertial>,
    GeneralizedHarmonic::Tags::Pi<Dim, ::Frame::Inertial>,
    GeneralizedHarmonic::Tags::Phi<Dim, ::Frame::Inertial>,
    ::Tags::deriv<GeneralizedHarmonic::Tags::Phi<Dim, Frame::Inertial>,
                  tmpl::size_t<Dim>, Frame::Inertial>>;

template <size_t Dim, typename Frame>
using vars_to_interpolate_to_target =
    tmpl::list<gr::Tags::SpatialMetric<Dim, Frame, DataVector>,
               gr::Tags::InverseSpatialMetric<Dim, Frame>,
               gr::Tags::ExtrinsicCurvature<Dim, Frame>,
               gr::Tags::SpatialChristoffelSecondKind<Dim, Frame>,
               gr::Tags::SpatialRicci<Dim, Frame>>;

template <typename Frame>
using tags_for_observing =
    tmpl::list<StrahlkorperGr::Tags::AreaCompute<Frame>,
               StrahlkorperGr::Tags::IrreducibleMassCompute<Frame>,
               StrahlkorperTags::MaxRicciScalarCompute,
               StrahlkorperTags::MinRicciScalarCompute,
               StrahlkorperGr::Tags::ChristodoulouMassCompute<Frame>,
               StrahlkorperGr::Tags::DimensionlessSpinMagnitudeCompute<Frame>>;

using surface_tags_for_observing = tmpl::list<StrahlkorperTags::RicciScalar>;

template <size_t Dim, typename Frame>
using compute_items_on_target = tmpl::append<
    tmpl::list<StrahlkorperTags::ThetaPhiCompute<Frame>,
               StrahlkorperTags::RadiusCompute<Frame>,
               StrahlkorperTags::RhatCompute<Frame>,
               StrahlkorperTags::InvJacobianCompute<Frame>,
               StrahlkorperTags::InvHessianCompute<Frame>,
               StrahlkorperTags::JacobianCompute<Frame>,
               StrahlkorperTags::DxRadiusCompute<Frame>,
               StrahlkorperTags::D2xRadiusCompute<Frame>,
               StrahlkorperTags::NormalOneFormCompute<Frame>,
               StrahlkorperTags::OneOverOneFormMagnitudeCompute<Dim, Frame,
                                                                DataVector>,
               StrahlkorperTags::TangentsCompute<Frame>,
               StrahlkorperTags::UnitNormalOneFormCompute<Frame>,
               StrahlkorperTags::UnitNormalVectorCompute<Frame>,
               StrahlkorperTags::GradUnitNormalOneFormCompute<Frame>,
               // Note that StrahlkorperTags::ExtrinsicCurvatureCompute is the
               // 2d extrinsic curvature of the strahlkorper embedded in the 3d
               // slice, whereas gr::tags::ExtrinsicCurvature is the 3d
               // extrinsic curvature of the slice embedded in 4d spacetime.
               // Both quantities are in the DataBox.
               StrahlkorperGr::Tags::AreaElementCompute<Frame>,
               StrahlkorperTags::ExtrinsicCurvatureCompute<Frame>,
               StrahlkorperTags::RicciScalarCompute<Frame>,
               StrahlkorperGr::Tags::SpinFunctionCompute<Frame>,
               StrahlkorperGr::Tags::DimensionfulSpinMagnitudeCompute<Frame>>,
    tags_for_observing<Frame>>;
}  // namespace ah
