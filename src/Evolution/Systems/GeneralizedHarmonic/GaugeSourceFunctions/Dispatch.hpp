// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include "DataStructures/Tensor/TypeAliases.hpp"
#include "Evolution/Systems/GeneralizedHarmonic/GaugeSourceFunctions/Gauges.hpp"
#include "Utilities/Gsl.hpp"

/// \cond
class DataVector;
template <size_t Dim>
class Mesh;
/// \endcond

namespace gh::gauges {
/*!
 * \brief Dispatch to the derived gauge condition.
 *
 * Which of the arguments to this function are used will depend on the gauge
 * condition, but since that is a runtime choice we need support for all gauge
 * conditions.
 */
template <size_t Dim>
void dispatch(
    gsl::not_null<tnsr::a<DataVector, Dim, Frame::Inertial>*> gauge_h,
    gsl::not_null<tnsr::ab<DataVector, Dim, Frame::Inertial>*> d4_gauge_h,
    const Scalar<DataVector>& lapse,
    const tnsr::I<DataVector, Dim, Frame::Inertial>& shift,
    const tnsr::a<DataVector, Dim, Frame::Inertial>&
        spacetime_unit_normal_one_form,
    const tnsr::A<DataVector, Dim, Frame::Inertial>& spacetime_unit_normal,
    const Scalar<DataVector>& sqrt_det_spatial_metric,
    const tnsr::II<DataVector, Dim, Frame::Inertial>& inverse_spatial_metric,
    const tnsr::abb<DataVector, Dim, Frame::Inertial>& d4_spacetime_metric,
    const Scalar<DataVector>& half_pi_two_normals,
    const tnsr::i<DataVector, Dim, Frame::Inertial>& half_phi_two_normals,
    const tnsr::aa<DataVector, Dim, Frame::Inertial>& spacetime_metric,
    const tnsr::aa<DataVector, Dim, Frame::Inertial>& pi,
    const tnsr::iaa<DataVector, Dim, Frame::Inertial>& phi,
    const Mesh<Dim>& mesh, double time,
    const tnsr::I<DataVector, Dim, Frame::Inertial>& inertial_coords,
    const InverseJacobian<DataVector, Dim, Frame::ElementLogical,
                          Frame::Inertial>& inverse_jacobian,
    const GaugeCondition& gauge_condition);
}  // namespace gh::gauges
