// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <array>
#include <deque>
#include <string>
#include <utility>

#include "DataStructures/DataBox/Tag.hpp"
#include "DataStructures/DataBox/TagName.hpp"
#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/EagerMath/Magnitude.hpp"
#include "DataStructures/Tensor/EagerMath/RaiseOrLowerIndex.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "NumericalAlgorithms/SphericalHarmonics/Strahlkorper.hpp"
#include "NumericalAlgorithms/SphericalHarmonics/TagsDeclarations.hpp"
#include "NumericalAlgorithms/SphericalHarmonics/TagsTypeAliases.hpp"
#include "PointwiseFunctions/GeneralRelativity/Surfaces/AreaElement.hpp"
#include "PointwiseFunctions/GeneralRelativity/Surfaces/ExtrinsicCurvature.hpp"
#include "PointwiseFunctions/GeneralRelativity/Surfaces/GradUnitNormalOneForm.hpp"
#include "PointwiseFunctions/GeneralRelativity/Surfaces/Mass.hpp"
#include "PointwiseFunctions/GeneralRelativity/Surfaces/RicciScalar.hpp"
#include "PointwiseFunctions/GeneralRelativity/Surfaces/Spin.hpp"
#include "PointwiseFunctions/GeneralRelativity/Surfaces/SurfaceIntegralOfScalar.hpp"
#include "PointwiseFunctions/GeneralRelativity/Surfaces/SurfaceIntegralOfVector.hpp"
#include "PointwiseFunctions/GeneralRelativity/Surfaces/UnitNormalOneForm.hpp"
#include "PointwiseFunctions/GeneralRelativity/TagsDeclarations.hpp"
#include "Utilities/ForceInline.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/TMPL.hpp"

/// \ingroup SurfacesGroup
/// Holds tags and ComputeItems associated with a `ylm::Strahlkorper`.
namespace ylm::Tags {
/// The OneOverOneFormMagnitude is the reciprocal of the magnitude of the
/// one-form perpendicular to the horizon
struct OneOverOneFormMagnitude : db::SimpleTag {
  using type = DataVector;
};

/// Computes the reciprocal of the magnitude of the one form perpendicular to
/// the horizon
template <typename DataType, size_t Dim, typename Frame>
struct OneOverOneFormMagnitudeCompute : db::ComputeTag,
                                        OneOverOneFormMagnitude {
  using base = OneOverOneFormMagnitude;
  using return_type = DataVector;
  static void function(
      const gsl::not_null<DataVector*> one_over_magnitude,
      const tnsr::II<DataType, Dim, Frame>& inverse_spatial_metric,
      const tnsr::i<DataType, Dim, Frame>& normal_one_form) {
    *one_over_magnitude =
        1.0 / get(magnitude(normal_one_form, inverse_spatial_metric));
  }
  using argument_tags =
      tmpl::list<gr::Tags::InverseSpatialMetric<DataType, Dim, Frame>,
                 NormalOneForm<Frame>>;
};

/// The unit normal one-form \f$s_j\f$ to the horizon.
template <typename Frame>
struct UnitNormalOneForm : db::SimpleTag {
  using type = tnsr::i<DataVector, 3, Frame>;
};
/// Computes the unit one-form perpendicular to the horizon
template <typename Frame>
struct UnitNormalOneFormCompute : UnitNormalOneForm<Frame>, db::ComputeTag {
  using base = UnitNormalOneForm<Frame>;
  static constexpr auto function = static_cast<void (*)(
      const gsl::not_null<tnsr::i<DataVector, 3, Frame>*>,
      const tnsr::i<DataVector, 3, Frame>&, const DataVector&)>(
      &::gr::surfaces::unit_normal_one_form<Frame>);
  using argument_tags =
      tmpl::list<ylm::Tags::NormalOneForm<Frame>, OneOverOneFormMagnitude>;
  using return_type = tnsr::i<DataVector, 3, Frame>;
};

/// UnitNormalVector is defined as \f$S^i = \gamma^{ij} S_j\f$,
/// where \f$S_j\f$ is the unit normal one form
/// and \f$\gamma^{ij}\f$ is the inverse spatial metric.
template <typename Frame>
struct UnitNormalVector : db::SimpleTag {
  using type = tnsr::I<DataVector, 3, Frame>;
};
/// Computes the UnitNormalVector perpendicular to the horizon.
template <typename Frame>
struct UnitNormalVectorCompute : UnitNormalVector<Frame>, db::ComputeTag {
  using base = UnitNormalVector<Frame>;
  static void function(
      gsl::not_null<tnsr::I<DataVector, 3, Frame>*> result,
      const tnsr::II<DataVector, 3, Frame>& inverse_spatial_metric,
      const tnsr::i<DataVector, 3, Frame>& unit_normal_one_form) {
    raise_or_lower_index(result, unit_normal_one_form, inverse_spatial_metric);
  }
  using argument_tags =
      tmpl::list<gr::Tags::InverseSpatialMetric<DataVector, 3, Frame>,
                 UnitNormalOneForm<Frame>>;
  using return_type = tnsr::I<DataVector, 3, Frame>;
};

/// The 3-covariant gradient \f$D_i S_j\f$ of a Strahlkorper's normal
template <typename Frame>
struct GradUnitNormalOneForm : db::SimpleTag {
  using type = tnsr::ii<DataVector, 3, Frame>;
};
/// Computes 3-covariant gradient \f$D_i S_j\f$ of a Strahlkorper's normal
template <typename Frame>
struct GradUnitNormalOneFormCompute : GradUnitNormalOneForm<Frame>,
                                      db::ComputeTag {
  using base = GradUnitNormalOneForm<Frame>;
  static constexpr auto function = static_cast<void (*)(
      const gsl::not_null<tnsr::ii<DataVector, 3, Frame>*>,
      const tnsr::i<DataVector, 3, Frame>&, const Scalar<DataVector>&,
      const tnsr::i<DataVector, 3, Frame>&,
      const tnsr::ii<DataVector, 3, Frame>&, const DataVector&,
      const tnsr::Ijj<DataVector, 3, Frame>&)>(
      &gr::surfaces::grad_unit_normal_one_form<Frame>);
  using argument_tags =
      tmpl::list<Rhat<Frame>, Radius<Frame>, UnitNormalOneForm<Frame>,
                 D2xRadius<Frame>, OneOverOneFormMagnitude,
                 gr::Tags::SpatialChristoffelSecondKind<DataVector, 3, Frame>>;
  using return_type = tnsr::ii<DataVector, 3, Frame>;
};

/// Extrinsic curvature of a 2D Strahlkorper embedded in a 3D space.
template <typename Frame>
struct ExtrinsicCurvature : db::SimpleTag {
  using type = tnsr::ii<DataVector, 3, Frame>;
};
/// Calculates the Extrinsic curvature of a 2D Strahlkorper embedded in a 3D
/// space.
template <typename Frame>
struct ExtrinsicCurvatureCompute : ExtrinsicCurvature<Frame>, db::ComputeTag {
  using base = ExtrinsicCurvature<Frame>;
  static constexpr auto function =
      static_cast<void (*)(const gsl::not_null<tnsr::ii<DataVector, 3, Frame>*>,
                           const tnsr::ii<DataVector, 3, Frame>&,
                           const tnsr::i<DataVector, 3, Frame>&,
                           const tnsr::I<DataVector, 3, Frame>&)>(
          &gr::surfaces::extrinsic_curvature<Frame>);
  using argument_tags =
      tmpl::list<GradUnitNormalOneForm<Frame>, UnitNormalOneForm<Frame>,
                 UnitNormalVector<Frame>>;
  using return_type = tnsr::ii<DataVector, 3, Frame>;
};

/// Ricci scalar is the two-dimensional intrinsic Ricci scalar curvature
/// of a Strahlkorper
struct RicciScalar : db::SimpleTag {
  using type = Scalar<DataVector>;
};

/// Computes the two-dimensional intrinsic Ricci scalar of a Strahlkorper
template <typename Frame>
struct RicciScalarCompute : RicciScalar, db::ComputeTag {
  using base = RicciScalar;
  static constexpr auto function = static_cast<void (*)(
      gsl::not_null<Scalar<DataVector>*>, const tnsr::ii<DataVector, 3, Frame>&,
      const tnsr::I<DataVector, 3, Frame>&,
      const tnsr::ii<DataVector, 3, Frame>&,
      const tnsr::II<DataVector, 3, Frame>&)>(
      &gr::surfaces::ricci_scalar<Frame>);
  using argument_tags =
      tmpl::list<gr::Tags::SpatialRicci<DataVector, 3, Frame>,
                 UnitNormalVector<Frame>, ExtrinsicCurvature<Frame>,
                 gr::Tags::InverseSpatialMetric<DataVector, 3, Frame>>;
  using return_type = Scalar<DataVector>;
};

/// The pointwise maximum of the Strahlkorper's intrinsic Ricci scalar
/// curvature.
struct MaxRicciScalar : db::SimpleTag {
  using type = double;
};

/// Computes the pointwise maximum of the Strahlkorper's intrinsic Ricci
/// scalar curvature.
struct MaxRicciScalarCompute : MaxRicciScalar, db::ComputeTag {
  using base = MaxRicciScalar;
  using return_type = double;
  static void function(const gsl::not_null<double*> max_ricci_scalar,
                       const Scalar<DataVector>& ricci_scalar) {
    *max_ricci_scalar = max(get(ricci_scalar));
  }
  using argument_tags = tmpl::list<RicciScalar>;
};

/// The pointwise minimum of the Strahlkorper’s intrinsic Ricci scalar
/// curvature.
struct MinRicciScalar : db::SimpleTag {
  using type = double;
};

/// Computes the pointwise minimum of the Strahlkorper’s intrinsic Ricci
/// scalar curvature.
struct MinRicciScalarCompute : MinRicciScalar, db::ComputeTag {
  using base = MinRicciScalar;
  using return_type = double;
  static void function(const gsl::not_null<double*> min_ricci_scalar,
                       const Scalar<DataVector>& ricci_scalar) {
    *min_ricci_scalar = min(get(ricci_scalar));
  }
  using argument_tags = tmpl::list<RicciScalar>;
};

/// @{
/// Computes the Euclidean area element on a Strahlkorper.
/// Useful for flat space integrals.
template <typename Frame>
struct EuclideanAreaElement : db::SimpleTag {
  using type = Scalar<DataVector>;
};

template <typename Frame>
struct EuclideanAreaElementCompute : EuclideanAreaElement<Frame>,
                                     db::ComputeTag {
  using base = EuclideanAreaElement<Frame>;
  using return_type = Scalar<DataVector>;
  static constexpr auto function = static_cast<void (*)(
      gsl::not_null<Scalar<DataVector>*>,
      const ylm::Tags::aliases::Jacobian<Frame>&,
      const tnsr::i<DataVector, 3, Frame>&, const Scalar<DataVector>&,
      const tnsr::i<DataVector, 3, Frame>&)>(
      &::gr::surfaces::euclidean_area_element<Frame>);
  using argument_tags =
      tmpl::list<ylm::Tags::Jacobian<Frame>, ylm::Tags::NormalOneForm<Frame>,
                 ylm::Tags::Radius<Frame>, ylm::Tags::Rhat<Frame>>;
};
/// @}

/// @{
/// Computes the flat-space integral of a scalar over a Strahlkorper.
template <typename IntegrandTag, typename Frame>
struct EuclideanSurfaceIntegral : db::SimpleTag {
  static std::string name() {
    return "EuclideanSurfaceIntegral(" + db::tag_name<IntegrandTag>() + ")";
  }
  using type = double;
};

template <typename IntegrandTag, typename Frame>
struct EuclideanSurfaceIntegralCompute
    : EuclideanSurfaceIntegral<IntegrandTag, Frame>,
      db::ComputeTag {
  using base = EuclideanSurfaceIntegral<IntegrandTag, Frame>;
  using return_type = double;
  static void function(const gsl::not_null<double*> surface_integral,
                       const Scalar<DataVector>& euclidean_area_element,
                       const Scalar<DataVector>& integrand,
                       const ylm::Strahlkorper<Frame>& strahlkorper) {
    *surface_integral = ::gr::surfaces::surface_integral_of_scalar<Frame>(
        euclidean_area_element, integrand, strahlkorper);
  }
  using argument_tags = tmpl::list<EuclideanAreaElement<Frame>, IntegrandTag,
                                   ylm::Tags::Strahlkorper<Frame>>;
};
/// @}

/// @{
/// Computes the Euclidean-space integral of a vector over a
/// Strahlkorper, \f$\oint V^i s_i (s_j s_k \delta^{jk})^{-1/2} d^2 S\f$,
/// where \f$s_i\f$ is the Strahlkorper surface unit normal and
/// \f$\delta^{ij}\f$ is the Kronecker delta.  Note that \f$s_i\f$ is
/// not assumed to be normalized; the denominator of the integrand
/// effectively normalizes it using the Euclidean metric.
template <typename IntegrandTag, typename Frame>
struct EuclideanSurfaceIntegralVector : db::SimpleTag {
  static std::string name() {
    return "EuclideanSurfaceIntegralVector(" + db::tag_name<IntegrandTag>() +
           ")";
  }
  using type = double;
};

template <typename IntegrandTag, typename Frame>
struct EuclideanSurfaceIntegralVectorCompute
    : EuclideanSurfaceIntegralVector<IntegrandTag, Frame>,
      db::ComputeTag {
  using base = EuclideanSurfaceIntegralVector<IntegrandTag, Frame>;
  using return_type = double;
  static void function(const gsl::not_null<double*> surface_integral,
                       const Scalar<DataVector>& euclidean_area_element,
                       const tnsr::I<DataVector, 3, Frame>& integrand,
                       const tnsr::i<DataVector, 3, Frame>& normal_one_form,
                       const ylm::Strahlkorper<Frame>& strahlkorper) {
    *surface_integral =
        ::gr::surfaces::euclidean_surface_integral_of_vector<Frame>(
            euclidean_area_element, integrand, normal_one_form, strahlkorper);
  }
  using argument_tags = tmpl::list<EuclideanAreaElement<Frame>, IntegrandTag,
                                   ylm::Tags::NormalOneForm<Frame>,
                                   ylm::Tags::Strahlkorper<Frame>>;
};
/// @}
}  // namespace ylm::Tags

namespace gr::surfaces {
/// \ingroup SurfacesGroup
/// Holds tags and ComputeItems associated with a `ylm::Strahlkorper` that
/// also need a metric.
namespace Tags {

/// @{
/// Computes the area element on a Strahlkorper. Useful for integrals.
template <typename Frame>
struct AreaElement : db::SimpleTag {
  using type = Scalar<DataVector>;
};

template <typename Frame>
struct AreaElementCompute : AreaElement<Frame>, db::ComputeTag {
  using base = AreaElement<Frame>;
  using return_type = Scalar<DataVector>;
  static constexpr auto function = static_cast<void (*)(
      gsl::not_null<Scalar<DataVector>*>, const tnsr::ii<DataVector, 3, Frame>&,
      const ylm::Tags::aliases::Jacobian<Frame>&,
      const tnsr::i<DataVector, 3, Frame>&, const Scalar<DataVector>&,
      const tnsr::i<DataVector, 3, Frame>&)>(&area_element<Frame>);
  using argument_tags =
      tmpl::list<gr::Tags::SpatialMetric<DataVector, 3, Frame>,
                 ylm::Tags::Jacobian<Frame>, ylm::Tags::NormalOneForm<Frame>,
                 ylm::Tags::Radius<Frame>, ylm::Tags::Rhat<Frame>>;
};
/// @}

/// @{
/// Computes the integral of a scalar over a Strahlkorper.
template <typename IntegrandTag, typename Frame>
struct SurfaceIntegral : db::SimpleTag {
  static std::string name() {
    return "SurfaceIntegral(" + db::tag_name<IntegrandTag>() + ")";
  }
  using type = double;
};

template <typename IntegrandTag, typename Frame>
struct SurfaceIntegralCompute : SurfaceIntegral<IntegrandTag, Frame>,
                                db::ComputeTag {
  using base = SurfaceIntegral<IntegrandTag, Frame>;
  using return_type = double;
  static void function(const gsl::not_null<double*> surface_integral,
                       const Scalar<DataVector>& area_element,
                       const Scalar<DataVector>& integrand,
                       const ylm::Strahlkorper<Frame>& strahlkorper) {
    *surface_integral = ::gr::surfaces::surface_integral_of_scalar<Frame>(
        area_element, integrand, strahlkorper);
  }
  using argument_tags = tmpl::list<AreaElement<Frame>, IntegrandTag,
                                   ylm::Tags::Strahlkorper<Frame>>;
};
/// @}

/// Tag representing the surface area of a Strahlkorper
struct Area : db::SimpleTag {
  using type = double;
};

/// Computes the surface area of a Strahlkorer, \f$A = \oint_S dA\f$ given an
/// AreaElement \f$dA\f$ and a Strahlkorper \f$S\f$.
template <typename Frame>
struct AreaCompute : Area, db::ComputeTag {
  using base = Area;
  using return_type = double;
  static void function(const gsl::not_null<double*> result,
                       const ylm::Strahlkorper<Frame>& strahlkorper,
                       const Scalar<DataVector>& area_element) {
    *result = strahlkorper.ylm_spherepack().definite_integral(
        get(area_element).data());
  }
  using argument_tags =
      tmpl::list<ylm::Tags::Strahlkorper<Frame>, AreaElement<Frame>>;
};

/// The Irreducible (areal) mass of an apparent horizon
struct IrreducibleMass : db::SimpleTag {
  using type = double;
};

/// Computes the Irreducible mass of an apparent horizon from its area
template <typename Frame>
struct IrreducibleMassCompute : IrreducibleMass, db::ComputeTag {
  using base = IrreducibleMass;
  using return_type = double;
  static void function(const gsl::not_null<double*> result, const double area) {
    *result = ::gr::surfaces::irreducible_mass(area);
  }

  using argument_tags = tmpl::list<Area>;
};

/// The spin function is proportional to the imaginary part of the
/// Strahlkorper’s complex scalar curvature.

struct SpinFunction : db::SimpleTag {
  using type = Scalar<DataVector>;
};

/// Calculates the spin function which is proportional to the imaginary part of
/// the Strahlkorper’s complex scalar curvature.
template <typename Frame>
struct SpinFunctionCompute : SpinFunction, db::ComputeTag {
  using base = SpinFunction;
  static constexpr auto function = static_cast<void (*)(
      gsl::not_null<Scalar<DataVector>*>,
      const ylm::Tags::aliases::Jacobian<Frame>&,
      const ylm::Strahlkorper<Frame>&, const tnsr::I<DataVector, 3, Frame>&,
      const Scalar<DataVector>&, const tnsr::ii<DataVector, 3, Frame>&)>(
      &gr::surfaces::spin_function<Frame>);
  using argument_tags =
      tmpl::list<ylm::Tags::Tangents<Frame>, ylm::Tags::Strahlkorper<Frame>,
                 ylm::Tags::UnitNormalVector<Frame>, AreaElement<Frame>,
                 gr::Tags::ExtrinsicCurvature<DataVector, 3, Frame>>;
  using return_type = Scalar<DataVector>;
};

/// The approximate-Killing-Vector quasilocal spin magnitude of a Strahlkorper
/// (see Sec. 2.2 of \cite Boyle2019kee and references therein).
struct DimensionfulSpinMagnitude : db::SimpleTag {
  using type = double;
};

/// Computes the approximate-Killing-Vector quasilocal spin magnitude of a
/// Strahlkorper
template <typename Frame>
struct DimensionfulSpinMagnitudeCompute : DimensionfulSpinMagnitude,
                                          db::ComputeTag {
  using base = DimensionfulSpinMagnitude;
  using return_type = double;
  static constexpr auto function = static_cast<void (*)(
      const gsl::not_null<double*>, const Scalar<DataVector>&,
      const Scalar<DataVector>&, const tnsr::ii<DataVector, 3, Frame>&,
      const ylm::Tags::aliases::Jacobian<Frame>&,
      const ylm::Strahlkorper<Frame>&, const Scalar<DataVector>&)>(
      &gr::surfaces::dimensionful_spin_magnitude<Frame>);
  using argument_tags =
      tmpl::list<ylm::Tags::RicciScalar, SpinFunction,
                 gr::Tags::SpatialMetric<DataVector, 3, Frame>,
                 ylm::Tags::Tangents<Frame>, ylm::Tags::Strahlkorper<Frame>,
                 AreaElement<Frame>>;
};

/// The dimensionful spin angular momentum vector.
template <typename Frame>
struct DimensionfulSpinVector : db::SimpleTag {
  using type = std::array<double, 3>;
};

/// Computes the dimensionful spin angular momentum vector.
template <typename MeasurementFrame, typename MetricDataFrame>
struct DimensionfulSpinVectorCompute : DimensionfulSpinVector<MeasurementFrame>,
                                       db::ComputeTag {
  using base = DimensionfulSpinVector<MeasurementFrame>;
  using return_type = std::array<double, 3>;
  static constexpr auto function = static_cast<void (*)(
      const gsl::not_null<std::array<double, 3>*>, double,
      const Scalar<DataVector>&, const Scalar<DataVector>&,
      const Scalar<DataVector>&, const ylm::Strahlkorper<MetricDataFrame>&,
      const tnsr::I<DataVector, 3, MeasurementFrame>&)>(
      &gr::surfaces::spin_vector<MetricDataFrame, MeasurementFrame>);
  using argument_tags =
      tmpl::list<DimensionfulSpinMagnitude, AreaElement<MetricDataFrame>,
                 ylm::Tags::RicciScalar, SpinFunction,
                 ylm::Tags::Strahlkorper<MetricDataFrame>,
                 ylm::Tags::CartesianCoords<MeasurementFrame>>;
};

/// The Christodoulou mass, which is a function of the dimensionful spin
/// angular momentum and the irreducible mass of a Strahlkorper.
struct ChristodoulouMass : db::SimpleTag {
  using type = double;
};

/// Computes the Christodoulou mass from the dimensionful spin angular momentum
/// and the irreducible mass of a Strahlkorper.
template <typename Frame>
struct ChristodoulouMassCompute : ChristodoulouMass, db::ComputeTag {
  using base = ChristodoulouMass;
  using return_type = double;
  static void function(const gsl::not_null<double*> result,
                       const double dimensionful_spin_magnitude,
                       const double irreducible_mass) {
    *result = ::gr::surfaces::christodoulou_mass(dimensionful_spin_magnitude,
                                                 irreducible_mass);
  }

  using argument_tags = tmpl::list<DimensionfulSpinMagnitude, IrreducibleMass>;
};

/// The dimensionless spin magnitude of a `Strahlkorper`.
template <typename Frame>
struct DimensionlessSpinMagnitude : db::SimpleTag {
  using type = double;
};

/// Computes the dimensionless spin magnitude \f$\chi = \frac{S}{M^2}\f$
/// from the dimensionful spin magnitude \f$S\f$ and the christodoulou
/// mass \f$M\f$ of a black hole.
template <typename Frame>
struct DimensionlessSpinMagnitudeCompute : DimensionlessSpinMagnitude<Frame>,
                                           db::ComputeTag {
  using base = DimensionlessSpinMagnitude<Frame>;
  using return_type = double;
  static constexpr auto function = static_cast<void (*)(
      const gsl::not_null<double*>, const double, const double)>(
      &gr::surfaces::dimensionless_spin_magnitude);
  using argument_tags =
      tmpl::list<DimensionfulSpinMagnitude, ChristodoulouMass>;
};

/// The dimensionless spin angular momentum vector
template <typename Frame>
struct DimensionlessSpinVector : db::SimpleTag {
  using type = std::array<double, 3>;
};

/// Computes the dimensionless spin angular momentum vector
template <typename MeasurementFrame, typename MetricDataFrame>
struct DimensionlessSpinVectorCompute
    : DimensionlessSpinVector<MeasurementFrame>,
      db::ComputeTag {
  using base = DimensionlessSpinVector<MeasurementFrame>;
  using argument_tags =
      tmpl::list<DimensionlessSpinMagnitude<MeasurementFrame>,
                 AreaElement<MetricDataFrame>, ylm::Tags::RicciScalar,
                 SpinFunction, ylm::Tags::Strahlkorper<MetricDataFrame>,
                 ylm::Tags::CartesianCoords<MeasurementFrame>>;
  using return_type = std::array<double, 3>;
  static constexpr auto function = static_cast<void (*)(
      const gsl::not_null<std::array<double, 3>*>, double,
      const Scalar<DataVector>&, const Scalar<DataVector>&,
      const Scalar<DataVector>&, const ylm::Strahlkorper<MetricDataFrame>&,
      const tnsr::I<DataVector, 3, MeasurementFrame>&)>(
      &gr::surfaces::spin_vector<MetricDataFrame, MeasurementFrame>);
};

}  // namespace Tags
}  // namespace gr::surfaces
