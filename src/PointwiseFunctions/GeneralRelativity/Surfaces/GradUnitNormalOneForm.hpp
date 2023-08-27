// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include "DataStructures/Tensor/TypeAliases.hpp"

/// \cond
class DataVector;
namespace gsl {
template <typename>
struct not_null;
}  // namespace gsl
/// \endcond

namespace gr::surfaces {
/// @{
/// \ingroup SurfacesGroup
/// \brief Computes 3-covariant gradient \f$D_i S_j\f$ of a
/// Strahlkorper's normal.
///
/// \details See Eqs. (1--9) of \cite Baumgarte1996hh.
/// Here \f$S_j\f$ is the (normalized) unit one-form to the surface,
/// and \f$D_i\f$ is the spatial covariant derivative.  Note that this
/// object is symmetric, even though this is not obvious from the
/// definition.  The input arguments `r_hat`, `radius`, and
/// `d2x_radius` depend on the Strahlkorper but not on the metric, and
/// can be computed from a Strahlkorper using ComputeItems in
/// `StrahlkorperTags`.  The input argument
/// `one_over_one_form_magnitude` is \f$1/\sqrt{g^{ij}n_i n_j}\f$,
/// where \f$n_i\f$ is `StrahlkorperTags::NormalOneForm` (i.e.  the
/// unnormalized one-form to the Strahlkorper); it can be computed
/// using (one over) the `magnitude` function.  The input argument
/// `unit_normal_one_form` is \f$S_j\f$,the normalized one-form.
template <typename Frame>
void grad_unit_normal_one_form(
    gsl::not_null<tnsr::ii<DataVector, 3, Frame>*> result,
    const tnsr::i<DataVector, 3, Frame>& r_hat,
    const Scalar<DataVector>& radius,
    const tnsr::i<DataVector, 3, Frame>& unit_normal_one_form,
    const tnsr::ii<DataVector, 3, Frame>& d2x_radius,
    const DataVector& one_over_one_form_magnitude,
    const tnsr::Ijj<DataVector, 3, Frame>& christoffel_2nd_kind);

template <typename Frame>
tnsr::ii<DataVector, 3, Frame> grad_unit_normal_one_form(
    const tnsr::i<DataVector, 3, Frame>& r_hat,
    const Scalar<DataVector>& radius,
    const tnsr::i<DataVector, 3, Frame>& unit_normal_one_form,
    const tnsr::ii<DataVector, 3, Frame>& d2x_radius,
    const DataVector& one_over_one_form_magnitude,
    const tnsr::Ijj<DataVector, 3, Frame>& christoffel_2nd_kind);
/// @}
}  // namespace gr::surfaces
