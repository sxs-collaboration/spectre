// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>

#include "DataStructures/DataBox/Prefixes.hpp"
#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "DataStructures/Variables.hpp"
#include "Evolution/Systems/CurvedScalarWave/Tags.hpp"
#include "NumericalAlgorithms/LinearOperators/PartialDerivatives.hpp"
#include "Utilities/Gsl.hpp"

namespace CurvedScalarWave::Worldtube {
/*!
 * \brief Computes the puncture/singular field \f$\Psi^\mathcal{P}\f$ of a
 * scalar charge in circular orbit around a Schwarzschild black hole as
 * described in \cite Detweiler2003.
 *
 * \details The field is computed using a Detweiler-Whiting singular
 * Green's function and perturbatively expanded in the geodesic distance from
 * the particle. It solves the inhomogeneous wave equation
 *
 * \f{align*}{
 * \Box \Psi^\mathcal{P} = -4 \pi q \int \sqrt{-g} \delta^4(x^i, z(\tau)) d \tau
 * \f}
 *
 * where \f$q\f$ is the scalar charge and \f$z(\tau)\f$ is the worldline of the
 * particle. The expression is expanded up to a certain order in geodesic
 * distance and transformed to Kerr-Schild coordinates.
 *
 * The function given here assumes that the particle has scalar charge \f$q=1\f$
 * and is on a fixed geodesic orbit  with
 * \f$z(t) = (R \cos{\omega t}, R \sin{\omega t}, 0)\f$ around a Schwarzschild
 * black hole, where \f$R\f$ is the orbital radius and
 * \f$ \omega = R^{-3/2} \f$. It returns the singular field at the requested
 * coordinates as well as its time and spatial derivative.
 *
 * \note The expressions were computed with Mathematica and optimized by
 * applying common subexpression elimination with sympy. The memory allocations
 * of temporaries were optimized manually.
 */
void puncture_field(
    const gsl::not_null<Variables<tmpl::list<
        CurvedScalarWave::Tags::Psi, ::Tags::dt<CurvedScalarWave::Tags::Psi>,
        ::Tags::deriv<CurvedScalarWave::Tags::Psi, tmpl::size_t<3>,
                      Frame::Inertial>>>*>
        result,
    const tnsr::I<DataVector, 3, Frame::Inertial>& coords, const double time,
    const double orbital_radius, const double bh_mass, size_t order);

/*!
 * \brief Computes the puncture/singular field \f$\Psi^\mathcal{P}\f$ of a
 * scalar charge in circular orbit around a Schwarzschild black hole as
 * described in \cite Detweiler2003 expanded to order 0 in geodesic distance.
 */
void puncture_field_0(
    const gsl::not_null<Variables<tmpl::list<
        CurvedScalarWave::Tags::Psi, ::Tags::dt<CurvedScalarWave::Tags::Psi>,
        ::Tags::deriv<CurvedScalarWave::Tags::Psi, tmpl::size_t<3>,
                      Frame::Inertial>>>*>
        result,
    const tnsr::I<DataVector, 3, Frame::Inertial>& coords, const double time,
    const double orbital_radius, const double bh_mass);

/*!
 * \brief Computes the puncture/singular field \f$\Psi^\mathcal{P}\f$ of a
 * scalar charge in circular orbit around a Schwarzschild black hole as
 * described in \cite Detweiler2003 expanded to order 1 in geodesic distance.
 */
void puncture_field_1(
    const gsl::not_null<Variables<tmpl::list<
        CurvedScalarWave::Tags::Psi, ::Tags::dt<CurvedScalarWave::Tags::Psi>,
        ::Tags::deriv<CurvedScalarWave::Tags::Psi, tmpl::size_t<3>,
                      Frame::Inertial>>>*>
        result,
    const tnsr::I<DataVector, 3, Frame::Inertial>& coords, const double time,
    const double orbital_radius, const double bh_mass);

/*!
 * \brief Computes the puncture/singular field \f$\Psi^\mathcal{P}\f$ of a
 * scalar charge in circular orbit around a Schwarzschild black hole as
 * described in \cite Detweiler2003 expanded to order 2 in geodesic distance.
 */
void puncture_field_2(
    const gsl::not_null<Variables<tmpl::list<
        CurvedScalarWave::Tags::Psi, ::Tags::dt<CurvedScalarWave::Tags::Psi>,
        ::Tags::deriv<CurvedScalarWave::Tags::Psi, tmpl::size_t<3>,
                      Frame::Inertial>>>*>
        result,
    const tnsr::I<DataVector, 3, Frame::Inertial>& coords, const double time,
    const double orbital_radius, const double bh_mass);
}  // namespace CurvedScalarWave::Worldtube
