// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>

#include "DataStructures/DataBox/Prefixes.hpp"
#include "DataStructures/DataBox/Tag.hpp"
#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Evolution/Systems/GeneralizedHarmonic/Tags.hpp"
#include "NumericalAlgorithms/LinearOperators/PartialDerivatives.hpp"
#include "PointwiseFunctions/GeneralRelativity/Tags.hpp"
#include "Utilities/ContainerHelpers.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/TMPL.hpp"

// IWYU pragma: no_forward_declare Tags::deriv

/// \cond
namespace domain {
namespace Tags {
template <size_t Dim, typename Frame>
struct Coordinates;
}  // namespace Tags
}  // namespace domain
class DataVector;
template <typename X, typename Symm, typename IndexList>
class Tensor;
/// \endcond

namespace GeneralizedHarmonic {
/// @{
/*!
 * \ingroup GeneralRelativityGroup
 * \brief Computes spatial derivatives of lapse (N) from the generalized
 *        harmonic variables and spacetime unit normal 1-form.
 *
 * \details If the generalized harmonic conjugate momentum and spatial
 * derivative variables are \f$\Pi_{ab} = -t^c \partial_c \psi_{ab} \f$ and
 * \f$\Phi_{iab} = \partial_i \psi_{ab} \f$, the spatial derivatives of N
 * can be obtained from:
 * \f{align*}
 *  t^a t^b \Phi_{iab} = -\frac{1}{2N} [\partial_i (-N^2 + N_jN^j)-
 *                               2 N^j \partial_i N_j
 *                               + N^j N^k \partial_i g_{jk}]
 *                     = -\frac{2}{N} \partial_i N,
 * \f}
 * since
 * \f[
 * \partial_i (N_jN^j) = 2N^j \partial_i N_j - N^j N^k \partial_i g_{jk}.
 * \f]
 *
 * \f[
 * \Longrightarrow \partial_i N = -(N/2) t^a \Phi_{iab} t^b
 * \f]
 */
template <size_t SpatialDim, typename Frame, typename DataType>
void spatial_deriv_of_lapse(
    gsl::not_null<tnsr::i<DataType, SpatialDim, Frame>*> deriv_lapse,
    const Scalar<DataType>& lapse,
    const tnsr::A<DataType, SpatialDim, Frame>& spacetime_unit_normal,
    const tnsr::iaa<DataType, SpatialDim, Frame>& phi);

template <size_t SpatialDim, typename Frame, typename DataType>
tnsr::i<DataType, SpatialDim, Frame> spatial_deriv_of_lapse(
    const Scalar<DataType>& lapse,
    const tnsr::A<DataType, SpatialDim, Frame>& spacetime_unit_normal,
    const tnsr::iaa<DataType, SpatialDim, Frame>& phi);
/// @}

namespace Tags {
/*!
 * \brief Compute item to get spatial derivatives of lapse from the
 * generalized harmonic variables and spacetime unit normal one-form.
 *
 * \details See `spatial_deriv_of_lapse()`. Can be retrieved using
 * `gr::Tags::Lapse` wrapped in `::Tags::deriv`.
 */
template <size_t SpatialDim, typename Frame>
struct DerivLapseCompute : ::Tags::deriv<gr::Tags::Lapse<DataVector>,
                                         tmpl::size_t<SpatialDim>, Frame>,
                           db::ComputeTag {
  using argument_tags =
      tmpl::list<gr::Tags::Lapse<DataVector>,
                 gr::Tags::SpacetimeNormalVector<SpatialDim, Frame, DataVector>,
                 Phi<SpatialDim, Frame>>;

  using return_type = tnsr::i<DataVector, SpatialDim, Frame>;

  static constexpr auto function = static_cast<void (*)(
      gsl::not_null<tnsr::i<DataVector, SpatialDim, Frame>*>,
      const Scalar<DataVector>&, const tnsr::A<DataVector, SpatialDim, Frame>&,
      const tnsr::iaa<DataVector, SpatialDim, Frame>&)>(
      &spatial_deriv_of_lapse<SpatialDim, Frame>);

  using base = ::Tags::deriv<gr::Tags::Lapse<DataVector>,
                             tmpl::size_t<SpatialDim>, Frame>;
};
}  // namespace Tags
}  // namespace GeneralizedHarmonic
