// Distributed under the MIT License.
// See LICENSE.txt for details.

///\file
/// Defines Functions for calculating spacetime tensors from 3+1 quantities

#pragma once

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <utility>

#include "DataStructures/DataBox/Prefixes.hpp"
#include "DataStructures/DataBox/Tag.hpp"
#include "DataStructures/Tensor/EagerMath/DeterminantAndInverse.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "DataStructures/VariablesTag.hpp"
#include "NumericalAlgorithms/LinearOperators/PartialDerivatives.hpp"
#include "PointwiseFunctions/GeneralRelativity/Tags.hpp"
#include "Utilities/TMPL.hpp"
#include "Utilities/TaggedTuple.hpp"

/// \cond
namespace gsl {
template <class T>
class not_null;
}  // namespace gsl
/// \endcond

/// \ingroup GeneralRelativityGroup
/// Holds functions related to general relativity.
namespace gr {
// @{
/*!
 * \ingroup GeneralRelativityGroup
 * \brief Computes the spacetime metric from the spatial metric, lapse, and
 * shift.
 * \details The spacetime metric \f$ \psi_{ab} \f$ is calculated as
 * \f{align}{
 *   \psi_{tt} &= - N^2 + N^m N^n g_{mn} \\
 *   \psi_{ti} &= g_{mi} N^m  \\
 *   \psi_{ij} &= g_{ij}
 * \f}
 * where \f$ N, N^i\f$ and \f$ g_{ij}\f$ are the lapse, shift and spatial metric
 * respectively
 */
template <size_t Dim, typename Frame, typename DataType>
void spacetime_metric(
    gsl::not_null<tnsr::aa<DataType, Dim, Frame>*> spacetime_metric,
    const Scalar<DataType>& lapse, const tnsr::I<DataType, Dim, Frame>& shift,
    const tnsr::ii<DataType, Dim, Frame>& spatial_metric) noexcept;

template <size_t SpatialDim, typename Frame, typename DataType>
tnsr::aa<DataType, SpatialDim, Frame> spacetime_metric(
    const Scalar<DataType>& lapse,
    const tnsr::I<DataType, SpatialDim, Frame>& shift,
    const tnsr::ii<DataType, SpatialDim, Frame>& spatial_metric) noexcept;
// @}

// @{
/*!
 * \ingroup GeneralRelativityGroup
 * \brief Compute spatial metric from spacetime metric.
 * \details Simply pull out the spatial components.
 */
template <size_t SpatialDim, typename Frame, typename DataType>
tnsr::ii<DataType, SpatialDim, Frame> spatial_metric(
    const tnsr::aa<DataType, SpatialDim, Frame>& spacetime_metric) noexcept;

template <size_t SpatialDim, typename Frame, typename DataType>
void spatial_metric(
    gsl::not_null<tnsr::ii<DataType, SpatialDim, Frame>*> spatial_metric,
    const tnsr::aa<DataType, SpatialDim, Frame>& spacetime_metric) noexcept;
// @}

//@{
/*!
 * \ingroup GeneralRelativityGroup
 * \brief Compute inverse spacetime metric from inverse spatial metric, lapse
 * and shift
 *
 * \details The inverse spacetime metric \f$ \psi^{ab} \f$ is calculated as
 * \f{align}
 *    \psi^{tt} &= -  1/N^2 \\
 *    \psi^{ti} &= N^i / N^2 \\
 *    \psi^{ij} &= g^{ij} - N^i N^j / N^2
 * \f}
 * where \f$ N, N^i\f$ and \f$ g^{ij}\f$ are the lapse, shift and inverse
 * spatial metric respectively
 */
template <size_t SpatialDim, typename Frame, typename DataType>
void inverse_spacetime_metric(
    gsl::not_null<tnsr::AA<DataType, SpatialDim, Frame>*>
        inverse_spacetime_metric,
    const Scalar<DataType>& lapse,
    const tnsr::I<DataType, SpatialDim, Frame>& shift,
    const tnsr::II<DataType, SpatialDim, Frame>&
        inverse_spatial_metric) noexcept;

template <size_t SpatialDim, typename Frame, typename DataType>
tnsr::AA<DataType, SpatialDim, Frame> inverse_spacetime_metric(
    const Scalar<DataType>& lapse,
    const tnsr::I<DataType, SpatialDim, Frame>& shift,
    const tnsr::II<DataType, SpatialDim, Frame>&
        inverse_spatial_metric) noexcept;
//@}

// @{
/*!
 * \ingroup GeneralRelativityGroup
 * \brief Compute shift from spacetime metric and inverse spatial metric.
 *
 * \details Computes
 * \f{align}
 *    N^i &= g^{ij} \psi_{jt}
 * \f}
 * where \f$ N^i\f$, \f$ g^{ij}\f$, and \f$\psi_{ab}\f$ are the shift, inverse
 * spatial metric, and spacetime metric.
 * This can be derived, e.g., from Eqs. 2.121--2.122 of Baumgarte & Shapiro.
 */
template <size_t SpatialDim, typename Frame, typename DataType>
tnsr::I<DataType, SpatialDim, Frame> shift(
    const tnsr::aa<DataType, SpatialDim, Frame>& spacetime_metric,
    const tnsr::II<DataType, SpatialDim, Frame>&
        inverse_spatial_metric) noexcept;

template <size_t SpatialDim, typename Frame, typename DataType>
void shift(gsl::not_null<tnsr::I<DataType, SpatialDim, Frame>*> shift,
           const tnsr::aa<DataType, SpatialDim, Frame>& spacetime_metric,
           const tnsr::II<DataType, SpatialDim, Frame>&
               inverse_spatial_metric) noexcept;
// @}

// @{
/*!
 * \ingroup GeneralRelativityGroup
 * \brief Compute lapse from shift and spacetime metric
 *
 * \details Computes
 * \f{align}
 *    N &= \sqrt{N^i \psi_{it}-\psi_{tt}}
 * \f}
 * where \f$ N \f$, \f$ N^i\f$, and \f$\psi_{ab}\f$ are the lapse, shift,
 * and spacetime metric.
 * This can be derived, e.g., from Eqs. 2.121--2.122 of Baumgarte & Shapiro.
 */
template <size_t SpatialDim, typename Frame, typename DataType>
Scalar<DataType> lapse(
    const tnsr::I<DataType, SpatialDim, Frame>& shift,
    const tnsr::aa<DataType, SpatialDim, Frame>& spacetime_metric) noexcept;

template <size_t SpatialDim, typename Frame, typename DataType>
void lapse(
    gsl::not_null<Scalar<DataType>*> lapse,
    const tnsr::I<DataType, SpatialDim, Frame>& shift,
    const tnsr::aa<DataType, SpatialDim, Frame>& spacetime_metric) noexcept;
// @}

// @{
/*!
 * \ingroup GeneralRelativityGroup
 * \brief Computes the time derivative of the spacetime metric from spatial
 * metric, lapse, shift, and their time derivatives.
 *
 * \details Computes the derivative as:
 *
 * \f{align}{
 * \partial_t g_{tt} &= - 2 \alpha \partial_t \alpha
 * - 2 \gamma_{i j} \beta^i \partial_t \beta^j
 * + \beta^i \beta^j \partial_t \gamma_{i j}\\
 * \partial_t g_{t i} &= \gamma_{j i} \partial_t \beta^j
 * + \beta^j \partial_t \gamma_{j i}\\
 * \partial_t g_{i j} &= \partial_t \gamma_{i j},
 * \f}
 *
 * where \f$\alpha, \beta^i, \gamma_{ij}\f$ are the lapse, shift, and spatial
 * metric respectively.
 */
template <size_t SpatialDim, typename Frame, typename DataType>
void time_derivative_of_spacetime_metric(
    gsl::not_null<tnsr::aa<DataType, SpatialDim, Frame>*> dt_spacetime_metric,
    const Scalar<DataType>& lapse, const Scalar<DataType>& dt_lapse,
    const tnsr::I<DataType, SpatialDim, Frame>& shift,
    const tnsr::I<DataType, SpatialDim, Frame>& dt_shift,
    const tnsr::ii<DataType, SpatialDim, Frame>& spatial_metric,
    const tnsr::ii<DataType, SpatialDim, Frame>& dt_spatial_metric) noexcept;

template <size_t SpatialDim, typename Frame, typename DataType>
tnsr::aa<DataType, SpatialDim, Frame> time_derivative_of_spacetime_metric(
    const Scalar<DataType>& lapse, const Scalar<DataType>& dt_lapse,
    const tnsr::I<DataType, SpatialDim, Frame>& shift,
    const tnsr::I<DataType, SpatialDim, Frame>& dt_shift,
    const tnsr::ii<DataType, SpatialDim, Frame>& spatial_metric,
    const tnsr::ii<DataType, SpatialDim, Frame>& dt_spatial_metric) noexcept;
//@}

//@{
/*!
 * \ingroup GeneralRelativityGroup
 * \brief Computes spacetime derivative of spacetime metric from spatial metric,
 * lapse, shift, and their space and time derivatives.
 *
 * \details Computes the derivatives as:
 * \f{align}
 *     \partial_\mu \psi_{tt} &= - 2 N \partial_\mu N
 *                 + 2 g_{mn} N^m \partial_\mu N^n
 *                 + N^m N^n \partial_\mu g_{mn} \\
 *     \partial_\mu \psi_{ti} &= g_{mi} \partial_\mu N^m
 *                 + N^m \partial_\mu g_{mi} \\
 *     \partial_\mu \psi_{ij} &= \partial_\mu g_{ij}
 * \f}
 * where \f$ N, N^i, g \f$ are the lapse, shift, and spatial metric
 * respectively.
 */
template <size_t SpatialDim, typename Frame, typename DataType>
void derivatives_of_spacetime_metric(
    gsl::not_null<tnsr::abb<DataType, SpatialDim, Frame>*>
        spacetime_deriv_spacetime_metric,
    const Scalar<DataType>& lapse, const Scalar<DataType>& dt_lapse,
    const tnsr::i<DataType, SpatialDim, Frame>& deriv_lapse,
    const tnsr::I<DataType, SpatialDim, Frame>& shift,
    const tnsr::I<DataType, SpatialDim, Frame>& dt_shift,
    const tnsr::iJ<DataType, SpatialDim, Frame>& deriv_shift,
    const tnsr::ii<DataType, SpatialDim, Frame>& spatial_metric,
    const tnsr::ii<DataType, SpatialDim, Frame>& dt_spatial_metric,
    const tnsr::ijj<DataType, SpatialDim, Frame>&
        deriv_spatial_metric) noexcept;

template <size_t SpatialDim, typename Frame, typename DataType>
tnsr::abb<DataType, SpatialDim, Frame> derivatives_of_spacetime_metric(
    const Scalar<DataType>& lapse, const Scalar<DataType>& dt_lapse,
    const tnsr::i<DataType, SpatialDim, Frame>& deriv_lapse,
    const tnsr::I<DataType, SpatialDim, Frame>& shift,
    const tnsr::I<DataType, SpatialDim, Frame>& dt_shift,
    const tnsr::iJ<DataType, SpatialDim, Frame>& deriv_shift,
    const tnsr::ii<DataType, SpatialDim, Frame>& spatial_metric,
    const tnsr::ii<DataType, SpatialDim, Frame>& dt_spatial_metric,
    const tnsr::ijj<DataType, SpatialDim, Frame>&
        deriv_spatial_metric) noexcept;
//@}

//@{
/*!
 * \brief Computes spacetime normal one-form from lapse.
 *
 * \details If \f$N\f$ is the lapse, then
 * \f{align} n_t &= - N \\
 * n_i &= 0 \f}
 * is computed.
 */
template <size_t SpatialDim, typename Frame, typename DataType>
void spacetime_normal_one_form(
    gsl::not_null<tnsr::a<DataType, SpatialDim, Frame>*> normal_one_form,
    const Scalar<DataType>& lapse) noexcept;

template <size_t SpatialDim, typename Frame, typename DataType>
tnsr::a<DataType, SpatialDim, Frame> spacetime_normal_one_form(
    const Scalar<DataType>& lapse) noexcept;
//@}

// @{
/*!
 * \ingroup GeneralRelativityGroup
 * \brief  Computes spacetime normal vector from lapse and shift.
 * \details If \f$N, N^i\f$ are the lapse and shift respectively, then
 * \f{align} n^t &= 1/N \\
 * n^i &= -\frac{N^i}{N} \f}
 * is computed.
 */
template <size_t SpatialDim, typename Frame, typename DataType>
tnsr::A<DataType, SpatialDim, Frame> spacetime_normal_vector(
    const Scalar<DataType>& lapse,
    const tnsr::I<DataType, SpatialDim, Frame>& shift) noexcept;

template <size_t SpatialDim, typename Frame, typename DataType>
void spacetime_normal_vector(
    gsl::not_null<tnsr::A<DataType, SpatialDim, Frame>*>
        spacetime_normal_vector,
    const Scalar<DataType>& lapse,
    const tnsr::I<DataType, SpatialDim, Frame>& shift) noexcept;
// @}

/*!
 * \ingroup GeneralRelativityGroup
 * \brief  Computes extrinsic curvature from metric and derivatives.
 * \details Uses the ADM evolution equation for the spatial metric,
 * \f[ K_{ij} = \frac{1}{2N} \left ( -\partial_0 g_{ij}
 * + N^k \partial_k g_{ij} + g_{ki} \partial_j N^k
 * + g_{kj} \partial_i N^k \right ) \f]
 * where \f$K_{ij}\f$ is the extrinsic curvature, \f$N\f$ is the lapse,
 * \f$N^i\f$ is the shift, and \f$g_{ij}\f$ is the spatial metric. In terms
 * of the Lie derivative of the spatial metric with respect to a unit timelike
 * vector \f$t^a\f$ normal to the spatial slice, this corresponds to the sign
 * convention
 * \f[ K_{ab} = - \frac{1}{2} \mathcal{L}_{\mathbf{t}} g_{ab} \f]
 */
template <size_t SpatialDim, typename Frame, typename DataType>
tnsr::ii<DataType, SpatialDim, Frame> extrinsic_curvature(
    const Scalar<DataType>& lapse,
    const tnsr::I<DataType, SpatialDim, Frame>& shift,
    const tnsr::iJ<DataType, SpatialDim, Frame>& deriv_shift,
    const tnsr::ii<DataType, SpatialDim, Frame>& spatial_metric,
    const tnsr::ii<DataType, SpatialDim, Frame>& dt_spatial_metric,
    const tnsr::ijj<DataType, SpatialDim, Frame>&
        deriv_spatial_metric) noexcept;

namespace Tags {
/*!
 * \brief Compute item for spacetime normal oneform \f$n_a\f$ from
 * the lapse \f$N\f$.
 *
 * \details Can be retrieved using `gr::Tags::SpacetimeNormalOneForm`.
 */
template <size_t SpatialDim, typename Frame, typename DataType>
struct SpacetimeNormalOneFormCompute
    : SpacetimeNormalOneForm<SpatialDim, Frame, DataType>,
      db::ComputeTag {
  using argument_tags = tmpl::list<Lapse<DataType>>;

  using return_type = tnsr::a<DataType, SpatialDim, Frame>;

  static constexpr auto function =
      static_cast<void (*)(gsl::not_null<tnsr::a<DataType, SpatialDim, Frame>*>,
                           const Scalar<DataType>&) noexcept>(
          &spacetime_normal_one_form<SpatialDim, Frame, DataType>);

  using base = SpacetimeNormalOneForm<SpatialDim, Frame, DataType>;
};

/*!
 * \brief Compute item for spacetime normal vector \f$n^a\f$ from
 * the lapse \f$N\f$ and the shift \f$N^i\f$.
 *
 * \details Can be retrieved using `gr::Tags::SpacetimeNormalVector`.
 */
template <size_t SpatialDim, typename Frame, typename DataType>
struct SpacetimeNormalVectorCompute
    : SpacetimeNormalVector<SpatialDim, Frame, DataType>,
      db::ComputeTag {
  using argument_tags =
      tmpl::list<Lapse<DataType>, Shift<SpatialDim, Frame, DataType>>;
  static constexpr auto function =
      static_cast<tnsr::A<DataType, SpatialDim, Frame> (*)(
          const Scalar<DataType>&,
          const tnsr::I<DataType, SpatialDim, Frame>&)>(
          &spacetime_normal_vector<SpatialDim, Frame, DataType>);
  using base = SpacetimeNormalVector<SpatialDim, Frame, DataType>;
};

/*!
 * \brief Compute item for spacetime metric \f$\psi_{ab}\f$ from the
 * lapse \f$N\f$, shift \f$N^i\f$, and spatial metric \f$g_{ij}\f$.
 *
 * \details Can be retrieved using `gr::Tags::SpacetimeMetric`.
 */
template <size_t SpatialDim, typename Frame, typename DataType>
struct SpacetimeMetricCompute : SpacetimeMetric<SpatialDim, Frame, DataType>,
                                db::ComputeTag {
  using argument_tags =
      tmpl::list<Lapse<DataType>, Shift<SpatialDim, Frame, DataType>,
                 SpatialMetric<SpatialDim, Frame, DataType>>;
  using base = SpacetimeMetric<SpatialDim, Frame, DataType>;
  static constexpr tnsr::aa<DataType, SpatialDim, Frame> (*function)(
      const Scalar<DataType>&, const tnsr::I<DataType, SpatialDim, Frame>&,
      const tnsr::ii<DataType, SpatialDim, Frame>&) =
      &spacetime_metric<SpatialDim, Frame, DataType>;
};

/*!
 * \brief Compute item for inverse spacetime metric \f$\psi^{ab}\f$
 * in terms of the lapse \f$N\f$, shift \f$N^i\f$, and inverse
 * spatial metric \f$g^{ij}\f$.
 *
 * \details Can be retrieved using `gr::Tags::InverseSpacetimeMetric`.
 */
template <size_t SpatialDim, typename Frame, typename DataType>
struct InverseSpacetimeMetricCompute
    : InverseSpacetimeMetric<SpatialDim, Frame, DataType>,
      db::ComputeTag {
  using argument_tags =
      tmpl::list<Lapse<DataType>, Shift<SpatialDim, Frame, DataType>,
                 InverseSpatialMetric<SpatialDim, Frame, DataType>>;

  using return_type = tnsr::AA<DataType, SpatialDim, Frame>;

  static constexpr auto function = static_cast<void (*)(
      gsl::not_null<tnsr::AA<DataType, SpatialDim, Frame>*>,
      const Scalar<DataType>&, const tnsr::I<DataType, SpatialDim, Frame>&,
      const tnsr::II<DataType, SpatialDim, Frame>&) noexcept>(
      &inverse_spacetime_metric<SpatialDim, Frame, DataType>);

  using base = InverseSpacetimeMetric<SpatialDim, Frame, DataType>;
};

/*!
 * \brief Compute item for spatial metric \f$g_{ij}\f$ from the
 * spacetime metric \f$\psi_{ab}\f$.
 *
 * \details Can be retrieved using `gr::Tags::SpatialMetric`.
 */
template <size_t SpatialDim, typename Frame, typename DataType>
struct SpatialMetricCompute : SpatialMetric<SpatialDim, Frame, DataType>,
                              db::ComputeTag {
  using argument_tags =
      tmpl::list<SpacetimeMetric<SpatialDim, Frame, DataType>>;
  static constexpr auto function =
      static_cast<tnsr::ii<DataType, SpatialDim, Frame> (*)(
          const tnsr::aa<DataType, SpatialDim, Frame>&)>(
          &spatial_metric<SpatialDim, Frame, DataType>);
  using base = SpatialMetric<SpatialDim, Frame, DataType>;
};

/*!
 * \brief Compute item for spatial metric determinant \f$g\f$
 * and inverse \f$g^{ij}\f$ in terms of the spatial metric \f$g_{ij}\f$.
 *
 * \details Can be retrieved using `gr::Tags::DetSpatialMetric` and
 * `gr::Tags::InverseSpatialMetric`.
 */
template <size_t SpatialDim, typename Frame, typename DataType>
struct DetAndInverseSpatialMetricCompute
    : ::Tags::Variables<
          tmpl::list<DetSpatialMetric<DataType>,
                     InverseSpatialMetric<SpatialDim, Frame, DataType>>>,
      db::ComputeTag {
  using argument_tags = tmpl::list<SpatialMetric<SpatialDim, Frame, DataType>>;
  using base = ::Tags::Variables<
      tmpl::list<DetSpatialMetric<DataType>,
                 InverseSpatialMetric<SpatialDim, Frame, DataType>>>;
  static constexpr auto function = &determinant_and_inverse<
      DetSpatialMetric<DataType>,
      InverseSpatialMetric<SpatialDim, Frame, DataType>, DataType,
      tmpl::integral_list<std::int32_t, 1, 1>,
      SpatialIndex<SpatialDim, UpLo::Lo, Frame>,
      SpatialIndex<SpatialDim, UpLo::Lo, Frame>>;
};

/*!
 * \brief Compute item to get spacetime derivative of spacetime metric from
 * spatial metric, lapse, shift, and their space and time derivatives.
 *
 * \details See `derivatives_of_spacetime_metric()`. Can be retrieved using
 * `gr::Tags::DerivativesOfSpacetimeMetric`.
 */
template <size_t SpatialDim, typename Frame>
struct DerivativesOfSpacetimeMetricCompute
    : gr::Tags::DerivativesOfSpacetimeMetric<SpatialDim, Frame, DataVector>,
      db::ComputeTag {
  using argument_tags = tmpl::list<
      gr::Tags::Lapse<DataVector>, ::Tags::dt<gr::Tags::Lapse<DataVector>>,
      ::Tags::deriv<gr::Tags::Lapse<DataVector>, tmpl::size_t<SpatialDim>,
                    Frame>,
      gr::Tags::Shift<SpatialDim, Frame, DataVector>,
      ::Tags::dt<gr::Tags::Shift<SpatialDim, Frame, DataVector>>,
      ::Tags::deriv<gr::Tags::Shift<SpatialDim, Frame, DataVector>,
                    tmpl::size_t<SpatialDim>, Frame>,
      gr::Tags::SpatialMetric<SpatialDim, Frame, DataVector>,
      ::Tags::dt<gr::Tags::SpatialMetric<SpatialDim, Frame, DataVector>>,
      ::Tags::deriv<gr::Tags::SpatialMetric<SpatialDim, Frame, DataVector>,
                    tmpl::size_t<SpatialDim>, Frame>>;

  using return_type = tnsr::abb<DataVector, SpatialDim, Frame>;

  static constexpr auto function = static_cast<void (*)(
      gsl::not_null<tnsr::abb<DataVector, SpatialDim, Frame>*>
          spacetime_deriv_spacetime_metric,
      const Scalar<DataVector>&, const Scalar<DataVector>&,
      const tnsr::i<DataVector, SpatialDim, Frame>&,
      const tnsr::I<DataVector, SpatialDim, Frame>&,
      const tnsr::I<DataVector, SpatialDim, Frame>&,
      const tnsr::iJ<DataVector, SpatialDim, Frame>&,
      const tnsr::ii<DataVector, SpatialDim, Frame>&,
      const tnsr::ii<DataVector, SpatialDim, Frame>&,
      const tnsr::ijj<DataVector, SpatialDim, Frame>&) noexcept>(
      &gr::derivatives_of_spacetime_metric<SpatialDim, Frame, DataVector>);

  using base =
      gr::Tags::DerivativesOfSpacetimeMetric<SpatialDim, Frame, DataVector>;
};

/*!
 * \brief Compute item to get spatial derivative of spacetime metric from
 * spatial metric, lapse, shift, and their space and time derivatives.
 *
 * \details Extracts spatial derivatives from spacetime derivatives computed
 * with `derivatives_of_spacetime_metric()`. Can be retrieved using
 * `gr::Tags::SpacetimeMetric` wrapped in `Tags::deriv`.
 */
template <size_t SpatialDim, typename Frame>
struct DerivSpacetimeMetricCompute
    : gr::Tags::DerivSpacetimeMetric<SpatialDim, Frame, DataVector>,
      db::ComputeTag {
  using argument_tags = tmpl::list<
      gr::Tags::DerivativesOfSpacetimeMetric<SpatialDim, Frame, DataVector>>;
  static constexpr auto function(
      const tnsr::abb<DataVector, SpatialDim, Frame>&
          spacetime_deriv_of_spacetime_metric) noexcept {
    auto deriv_spacetime_metric =
        make_with_value<tnsr::iaa<DataVector, SpatialDim, Frame>>(
            spacetime_deriv_of_spacetime_metric, 0.);
    for (size_t i = 0; i < SpatialDim; ++i) {
      for (size_t a = 0; a < SpatialDim + 1; ++a) {
        for (size_t b = a; b < SpatialDim + 1; ++b) {
          deriv_spacetime_metric.get(i, a, b) =
              spacetime_deriv_of_spacetime_metric.get(i + 1, a, b);
        }
      }
    }
    return deriv_spacetime_metric;
  }
  using base = gr::Tags::DerivSpacetimeMetric<SpatialDim, Frame, DataVector>;
};

/*!
 * \brief Compute item to get the square root of the determinant of the spatial
 * metric \f$\sqrt{g}\f$ via `gr::Tags::DetAndInverseSpatialMetric`.
 *
 * \details Can be retrieved using `gr::Tags::SqrtDetSpatialMetric`.
 */
template <size_t SpatialDim, typename Frame, typename DataType>
struct SqrtDetSpatialMetricCompute : SqrtDetSpatialMetric<DataType>,
                                     db::ComputeTag {
  using argument_tags = tmpl::list<DetSpatialMetric<DataType>>;
  static Scalar<DataType> function(const Scalar<DataType>& det_spatial_metric) {
    return Scalar<DataType>{sqrt(get(det_spatial_metric))};
  }
  using base = SqrtDetSpatialMetric<DataType>;
};

/*!
 * \brief Compute item for shift \f$N^i\f$ from the spacetime metric
 * \f$\psi_{ab}\f$ and the inverse spatial metric \f$g^{ij}\f$.
 *
 * \details Can be retrieved using `gr::Tags::Shift`.
 */
template <size_t SpatialDim, typename Frame, typename DataType>
struct ShiftCompute : Shift<SpatialDim, Frame, DataType>, db::ComputeTag {
  using argument_tags =
      tmpl::list<SpacetimeMetric<SpatialDim, Frame, DataType>,
                 InverseSpatialMetric<SpatialDim, Frame, DataType>>;
  static constexpr auto function =
      static_cast<tnsr::I<DataType, SpatialDim, Frame> (*)(
          const tnsr::aa<DataType, SpatialDim, Frame>&,
          const tnsr::II<DataType, SpatialDim, Frame>&)>(
          &shift<SpatialDim, Frame, DataType>);
  using base = Shift<SpatialDim, Frame, DataType>;
};

/*!
 * \brief Compute item for lapse \f$N\f$ from the spacetime metric
 * \f$\psi_{ab}\f$ and the shift \f$N^i\f$.
 *
 * \details Can be retrieved using `gr::Tags::Lapse`.
 */
template <size_t SpatialDim, typename Frame, typename DataType>
struct LapseCompute : Lapse<DataType>, db::ComputeTag {
  using argument_tags =
      tmpl::list<Shift<SpatialDim, Frame, DataType>,
                 SpacetimeMetric<SpatialDim, Frame, DataType>>;
  static constexpr auto function = static_cast<Scalar<DataType> (*)(
      const tnsr::I<DataType, SpatialDim, Frame>&,
      const tnsr::aa<DataType, SpatialDim, Frame>&)>(
      &lapse<SpatialDim, Frame, DataType>);
  using base = Lapse<DataType>;
};
}  // namespace Tags
}  // namespace gr
