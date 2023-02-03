// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>
#include <string>

#include "DataStructures/Tensor/TypeAliases.hpp"
#include "Domain/Tags.hpp"
#include "NumericalAlgorithms/Spectral/Mesh.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/TMPL.hpp"

/// \cond
class DataVector;
/// \endcond

namespace domain {
/*!
 * \ingroup ComputationalDomainGroup
 * \brief A diagnostic comparing the analytic and numerical Jacobians for a
 * map.
 *
 * Specifically, returns
 * \f[
 * C_{\hat{i}} = 1 -
 * \frac{\sum_i |\partial_{\hat{i}} x^i|}{\sum_i |D_{\hat{i}} x^i|}
 * \f], where \f$x^{\hat{i}}\f$ are the logical
 * coordinates, \f$x^i\f$ are the coordinates in the target frame,
 * \f$\partial_{\hat{i}}x^i\f$ is the analytic Jacobian, and \f$D_{\hat{i}}
 * x^i\f$ is the numerical Jacobian.
 *
 * \note This function accepts the transpose of the numeric Jacobian as a
 * parameter, since the numeric Jacobian will typically be computed via
 * logical_partial_derivative(), which prepends the logical (source frame)
 * derivative index. Tensors of type Jacobian, in contrast, have the derivative
 * index second.
 */
template <size_t Dim, typename Fr>
void jacobian_diagnostic(
    const gsl::not_null<
        tnsr::i<DataVector, Dim, typename Frame::ElementLogical>*>
        jacobian_diag,
    const Jacobian<DataVector, Dim, typename Frame::ElementLogical, Fr>&
        analytic_jacobian,
    const TensorMetafunctions::prepend_spatial_index<
        tnsr::I<DataVector, Dim, Fr>, Dim, UpLo::Lo,
        typename Frame::ElementLogical>& numeric_jacobian_transpose);

/// @{
/*!
 * \ingroup ComputationalDomainGroup
 * \brief A diagnostic comparing the analytic and numerical Jacobians for a
 * map.
 *
 * Specifically, returns
 * \f[
 * C_{\hat{i}} = 1 -
 * \frac{\sum_i |\partial_{\hat{i}} x^i|}{\sum_i |D_{\hat{i}} x^i|}
 * \f],
 * where \f$x^{\hat{i}}\f$ are the logical coordinates, \f$x^i\f$ are the
 * coordinates in the target frame, \f$\partial_{\hat{i}}x^i\f$ is the analytic
 * Jacobian, and \f$D_{\hat{i}} x^i\f$ is the numerical Jacobian.
 *
 * \note This function accepts the analytic jacobian, mapped coordinates, and
 * mesh as a parameter. The numeric jacobian is computed
 * internally by differentiating the mapped coordinates with respect to the
 * logical coordinates.
 */
template <size_t Dim, typename Fr>
void jacobian_diagnostic(
    const gsl::not_null<tnsr::i<DataVector, Dim, Frame::ElementLogical>*>
        jacobian_diag,
    const ::Jacobian<DataVector, Dim, Frame::ElementLogical, Fr>&
        analytic_jacobian,
    const tnsr::I<DataVector, Dim, Fr>& mapped_coords, const ::Mesh<Dim>& mesh);

template <size_t Dim, typename Fr>
tnsr::i<DataVector, Dim, Frame::ElementLogical> jacobian_diagnostic(
    const ::Jacobian<DataVector, Dim, Frame::ElementLogical, Fr>&
        analytic_jacobian,
    const tnsr::I<DataVector, Dim, Fr>& mapped_coords, const ::Mesh<Dim>& mesh);
/// @}

namespace Tags {
/// \ingroup DataBoxTagsGroup
/// \ingroup ComputationalDomainGroup
/// \brief A diagnostic comparing the analytic and numerical Jacobians for a
/// map. See `domain::jacobian_diagnostic` for details.
template <size_t Dim>
struct JacobianDiagnostic : db::SimpleTag {
  using type = tnsr::i<DataVector, Dim, typename Frame::ElementLogical>;
};

/// \ingroup DataBoxTagsGroup
/// \ingroup ComputationalDomainGroup
/// \brief Computes the Jacobian diagnostic, which compares the analytic
/// Jacobian (provided by some coordinate map) to a numerical Jacobian computed
/// using numerical partial derivatives. The coordinates must be in the target
/// frame of the map. See `domain::jacobian_diagnostic` for details of the
/// calculation.
template <size_t Dim, typename TargetFrame>
struct JacobianDiagnosticCompute : JacobianDiagnostic<Dim>, db::ComputeTag {
  using base = JacobianDiagnostic<Dim>;
  using return_type = typename base::type;
  using argument_tags = tmpl::list<
      ::domain::Tags::Jacobian<Dim, Frame::ElementLogical, TargetFrame>,
      ::domain::Tags::Coordinates<Dim, TargetFrame>, Mesh<Dim>>;
  static constexpr auto function = static_cast<void (*)(
      const gsl::not_null<
          tnsr::i<DataVector, Dim, typename Frame::ElementLogical>*>,
      const ::Jacobian<DataVector, Dim, Frame::ElementLogical, TargetFrame>&,
      const tnsr::I<DataVector, Dim, TargetFrame>&, const ::Mesh<Dim>&)>(
      &::domain::jacobian_diagnostic<Dim, TargetFrame>);
};
}  // namespace Tags
}  // namespace domain
