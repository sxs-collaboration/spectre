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

using tags_for_observing = tmpl::list<
    StrahlkorperGr::Tags::AreaCompute<::Frame::Inertial>,
    StrahlkorperGr::Tags::IrreducibleMassCompute<::Frame::Inertial>,
    StrahlkorperTags::MaxRicciScalarCompute,
    StrahlkorperTags::MinRicciScalarCompute,
    StrahlkorperGr::Tags::ChristodoulouMassCompute<::Frame::Inertial>,
    StrahlkorperGr::Tags::DimensionlessSpinMagnitudeCompute<::Frame::Inertial>>;

using surface_tags_for_observing = tmpl::list<StrahlkorperTags::RicciScalar>;

template <size_t Dim>
using compute_items_on_target = tmpl::append<
    tmpl::list<
        StrahlkorperTags::ThetaPhiCompute<::Frame::Inertial>,
        StrahlkorperTags::RadiusCompute<::Frame::Inertial>,
        StrahlkorperTags::RhatCompute<::Frame::Inertial>,
        StrahlkorperTags::InvJacobianCompute<::Frame::Inertial>,
        StrahlkorperTags::InvHessianCompute<::Frame::Inertial>,
        StrahlkorperTags::JacobianCompute<::Frame::Inertial>,
        StrahlkorperTags::DxRadiusCompute<::Frame::Inertial>,
        StrahlkorperTags::D2xRadiusCompute<::Frame::Inertial>,
        StrahlkorperTags::NormalOneFormCompute<::Frame::Inertial>,
        StrahlkorperTags::OneOverOneFormMagnitudeCompute<Dim, ::Frame::Inertial,
                                                         DataVector>,
        StrahlkorperTags::TangentsCompute<::Frame::Inertial>,
        StrahlkorperTags::UnitNormalOneFormCompute<::Frame::Inertial>,
        StrahlkorperTags::UnitNormalVectorCompute<::Frame::Inertial>,
        StrahlkorperTags::GradUnitNormalOneFormCompute<::Frame::Inertial>,
        // Note that StrahlkorperTags::ExtrinsicCurvatureCompute is the
        // 2d extrinsic curvature of the strahlkorper embedded in the 3d
        // slice, whereas gr::tags::ExtrinsicCurvature is the 3d extrinsic
        // curvature of the slice embedded in 4d spacetime.  Both quantities
        // are in the DataBox.
        StrahlkorperGr::Tags::AreaElementCompute<::Frame::Inertial>,
        StrahlkorperTags::ExtrinsicCurvatureCompute<::Frame::Inertial>,
        StrahlkorperTags::RicciScalarCompute<::Frame::Inertial>,
        StrahlkorperGr::Tags::SpinFunctionCompute<::Frame::Inertial>,
        StrahlkorperGr::Tags::DimensionfulSpinMagnitudeCompute<
            ::Frame::Inertial>>,
    tags_for_observing>;
}  // namespace ah
