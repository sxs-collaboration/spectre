// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include "Evolution/Systems/GeneralizedHarmonic/ConstraintDamping/Tags.hpp"
#include "Evolution/Systems/GeneralizedHarmonic/TagsDeclarations.hpp"
#include "NumericalAlgorithms/LinearOperators/PartialDerivatives.hpp"
#include "NumericalAlgorithms/SphericalHarmonics/Tags.hpp"
#include "PointwiseFunctions/GeneralRelativity/Surfaces/TagsDeclarations.hpp"
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
using source_vars =
    tmpl::list<gr::Tags::SpacetimeMetric<DataVector, Dim>,
               gh::Tags::Pi<DataVector, Dim>, gh::Tags::Phi<DataVector, Dim>,
               ::Tags::deriv<gh::Tags::Phi<DataVector, Dim>, tmpl::size_t<Dim>,
                             Frame::Inertial>>;

template <size_t Dim, typename Frame>
using vars_to_interpolate_to_target =
    tmpl::list<gr::Tags::SpatialMetric<DataVector, Dim, Frame>,
               gr::Tags::InverseSpatialMetric<DataVector, Dim, Frame>,
               gr::Tags::ExtrinsicCurvature<DataVector, Dim, Frame>,
               gr::Tags::SpatialChristoffelSecondKind<DataVector, Dim, Frame>,
               gr::Tags::SpatialRicci<DataVector, Dim, Frame>>;

template <typename Frame>
using tags_for_observing = tmpl::list<
    gr::surfaces::Tags::AreaCompute<Frame>,
    gr::surfaces::Tags::IrreducibleMassCompute<Frame>,
    ylm::Tags::MaxRicciScalarCompute, ylm::Tags::MinRicciScalarCompute,
    gr::surfaces::Tags::ChristodoulouMassCompute<Frame>,
    gr::surfaces::Tags::DimensionlessSpinMagnitudeCompute<Frame>
    // Needs `ObserveTimeSeriesOnSurface` to be able to write a `std::array`
    // gr::surfaces::Tags::DimensionlessSpinVectorCompute<Frame, Frame>
    >;

using surface_tags_for_observing = tmpl::list<ylm::Tags::RicciScalar>;

template <size_t Dim, typename Frame>
using compute_items_on_target = tmpl::append<
    tmpl::list<
        ylm::Tags::ThetaPhiCompute<Frame>, ylm::Tags::RadiusCompute<Frame>,
        ylm::Tags::RhatCompute<Frame>, ylm::Tags::CartesianCoordsCompute<Frame>,
        ylm::Tags::InvJacobianCompute<Frame>,
        ylm::Tags::InvHessianCompute<Frame>, ylm::Tags::JacobianCompute<Frame>,
        ylm::Tags::DxRadiusCompute<Frame>, ylm::Tags::D2xRadiusCompute<Frame>,
        ylm::Tags::NormalOneFormCompute<Frame>,
        ylm::Tags::OneOverOneFormMagnitudeCompute<DataVector, Dim, Frame>,
        ylm::Tags::TangentsCompute<Frame>,
        ylm::Tags::UnitNormalOneFormCompute<Frame>,
        ylm::Tags::UnitNormalVectorCompute<Frame>,
        ylm::Tags::GradUnitNormalOneFormCompute<Frame>,
        // Note that ylm::Tags::ExtrinsicCurvatureCompute is the
        // 2d extrinsic curvature of the strahlkorper embedded in the 3d
        // slice, whereas gr::tags::ExtrinsicCurvature is the 3d
        // extrinsic curvature of the slice embedded in 4d spacetime.
        // Both quantities are in the DataBox.
        gr::surfaces::Tags::AreaElementCompute<Frame>,
        ylm::Tags::EuclideanAreaElementCompute<Frame>,
        ylm::Tags::ExtrinsicCurvatureCompute<Frame>,
        ylm::Tags::RicciScalarCompute<Frame>,
        gr::surfaces::Tags::SpinFunctionCompute<Frame>,
        gr::surfaces::Tags::DimensionfulSpinMagnitudeCompute<Frame>>,
    tags_for_observing<Frame>>;
}  // namespace ah
