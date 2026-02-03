// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>

#include "DataStructures/DataBox/Tag.hpp"
#include "DataStructures/Tensor/TypeAliases.hpp"
#include "PointwiseFunctions/GeneralRelativity/Tags.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/TMPL.hpp"

namespace gr {

/// @{
/*!
 * Computes the real part of the cubic invariant of the Weyl tensor from the
 * electric and magnetic parts. The cubic invariant is e.g. given in Equation
 * (10) of \cite Dennison:2012vf
 * \f[
 *   \mathcal{J} = \left(-\frac{1}{6} E^i_j E^j_k E^k_i + \frac{1}{2} E^i_j
 *    B^j_k B^k_i\right) + i\left(\frac{1}{6} B^i_j B^j_k B^k_i - \frac{1}{2}
 *    B^i_j E^j_k E^k_i\right)
 * \f]
 */
template <typename DataType, size_t Dim, typename Frame>
void cubic_invariant_real(
    gsl::not_null<Scalar<DataType>*> result,
    const tnsr::ii<DataType, Dim, Frame>& weyl_electric,
    const tnsr::ii<DataType, Dim, Frame>& weyl_magnetic,
    const tnsr::II<DataType, Dim, Frame>& inverse_spatial_metric);

template <typename DataType, size_t Dim, typename Frame>
Scalar<DataType> cubic_invariant_real(
    const tnsr::ii<DataType, Dim, Frame>& weyl_electric,
    const tnsr::ii<DataType, Dim, Frame>& weyl_magnetic,
    const tnsr::II<DataType, Dim, Frame>& inverse_spatial_metric);
/// @}

/// @{
/*!
 * Computes the imaginary part of the cubic invariant of the Weyl tensor from
 * the electric and magnetic parts. The cubic invariant is e.g. given in
 * Equation (10) of \cite Dennison:2012vf
 * \f[
 *   \mathcal{J} = \left(-\frac{1}{6} E^i_j E^j_k E^k_i + \frac{1}{2} E^i_j
 *    B^j_k B^k_i\right) + i\left(\frac{1}{6} B^i_j B^j_k B^k_i - \frac{1}{2}
 *    B^i_j E^j_k E^k_i\right)
 * \f]
 */
template <typename DataType, size_t Dim, typename Frame>
void cubic_invariant_imag(
    gsl::not_null<Scalar<DataType>*> result,
    const tnsr::ii<DataType, Dim, Frame>& weyl_electric,
    const tnsr::ii<DataType, Dim, Frame>& weyl_magnetic,
    const tnsr::II<DataType, Dim, Frame>& inverse_spatial_metric);

template <typename DataType, size_t Dim, typename Frame>
Scalar<DataType> cubic_invariant_imag(
    const tnsr::ii<DataType, Dim, Frame>& weyl_electric,
    const tnsr::ii<DataType, Dim, Frame>& weyl_magnetic,
    const tnsr::II<DataType, Dim, Frame>& inverse_spatial_metric);
/// @}

}  // namespace gr

namespace gr::Tags {
// Simple and compute tags for cubic invariants
template <typename DataType>
struct CubicInvariantReal : db::SimpleTag {
  using type = Scalar<DataType>;
};

template <typename DataType, size_t Dim, typename Frame>
struct CubicInvariantRealCompute : CubicInvariantReal<DataType>,
                                   db::ComputeTag {
  using argument_tags =
      tmpl::list<gr::Tags::WeylElectric<DataType, Dim, Frame>,
                 gr::Tags::WeylMagnetic<DataType, Dim, Frame>,
                 gr::Tags::InverseSpatialMetric<DataType, Dim, Frame>>;
  using return_type = Scalar<DataType>;
  static constexpr auto function = static_cast<void (*)(
      gsl::not_null<Scalar<DataType>*>, const tnsr::ii<DataType, Dim, Frame>&,
      const tnsr::ii<DataType, Dim, Frame>&,
      const tnsr::II<DataType, Dim, Frame>&)>(
      &gr::cubic_invariant_real<DataType, Dim, Frame>);
  using base = CubicInvariantReal<DataType>;
};

template <typename DataType>
struct CubicInvariantImag : db::SimpleTag {
  using type = Scalar<DataType>;
};

template <typename DataType, size_t Dim, typename Frame>
struct CubicInvariantImagCompute : CubicInvariantImag<DataType>,
                                   db::ComputeTag {
  using argument_tags =
      tmpl::list<gr::Tags::WeylElectric<DataType, Dim, Frame>,
                 gr::Tags::WeylMagnetic<DataType, Dim, Frame>,
                 gr::Tags::InverseSpatialMetric<DataType, Dim, Frame>>;
  using return_type = Scalar<DataType>;
  static constexpr auto function = static_cast<void (*)(
      gsl::not_null<Scalar<DataType>*>, const tnsr::ii<DataType, Dim, Frame>&,
      const tnsr::ii<DataType, Dim, Frame>&,
      const tnsr::II<DataType, Dim, Frame>&)>(
      &gr::cubic_invariant_imag<DataType, Dim, Frame>);
  using base = CubicInvariantImag<DataType>;
};
}  // namespace gr::Tags
