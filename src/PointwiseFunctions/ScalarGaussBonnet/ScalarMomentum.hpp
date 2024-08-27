// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>

#include "DataStructures/DataBox/Tag.hpp"
#include "Elliptic/Systems/ScalarGaussBonnet/Tags.hpp"
#include "PointwiseFunctions/GeneralRelativity/Tags.hpp"

namespace sgb {
/// @{
/*!
 * \brief Compute the momentum $\Pi$ assuming quasi-stationarity in sGB gravity.
 *
 * This expression can be obtained by simply demanding $\partial_t \Psi = 0$,
 * yielding
 *
 * \begin{equation}
 * \Pi \equiv -n^{a} \partial_a = \alpha^{-1} \beta^{i} \partial_i \Psi
 * \end{equation}
 */

void scalar_momentum(gsl::not_null<Scalar<DataVector>*> result,
                     const tnsr::i<DataVector, 3, Frame::Inertial>& deriv,
                     const tnsr::I<DataVector, 3>& shift,
                     const Scalar<DataVector>& lapse);

Scalar<DataVector> scalar_momentum(
    const tnsr::i<DataVector, 3, Frame::Inertial>& deriv,
    const tnsr::I<DataVector, 3>& shift, const Scalar<DataVector>& lapse);
/// @{

namespace Tags {

/*!
 * \brief Re-compute the momentum Pi, assuming quasi-stationarity.
 */
template <typename ShiftTag, typename OutputTag>
struct PiCompute : OutputTag, db::ComputeTag {
 public:
  using base = OutputTag;
  using return_type = typename base::type;
  static constexpr auto function =
      static_cast<void (*)(gsl::not_null<return_type*>,
                           const tnsr::i<DataVector, 3, Frame::Inertial>&,
                           const tnsr::I<DataVector, 3>&,
                           const Scalar<DataVector>&)>(&scalar_momentum);
  using argument_tags =
      tmpl::list<::Tags::deriv<Psi, tmpl::size_t<3>, Frame::Inertial>, ShiftTag,
                 gr::Tags::Lapse<DataVector>>;
};

}  // namespace Tags
}  // namespace sgb
