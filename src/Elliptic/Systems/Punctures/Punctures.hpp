// Distributed under the MIT License.
// See LICENSE.txt for details.

/// \file
/// Documents the `Punctures` namespace

#pragma once

/*!
 * \ingroup EllipticSystemsGroup
 * \brief Items related to solving the puncture equation
 *
 * The puncture equation
 *
 * \begin{equation}\label{eq:puncture_eqn}
 * -\nabla^2 u = \beta \left(\alpha \left(1 + u\right) + 1\right)^{-7}
 * \end{equation}
 *
 * is a nonlinear Poisson-type elliptic PDE for the "puncture field" $u$. See
 * Eq. (12.52) and surrounding discussion in \cite BaumgarteShapiro, or
 * \cite BrandtBruegmann1997 for an introduction. To arrive at the puncture
 * equation we assume conformal flatness and maximal slicing in vacuum so the
 * Einstein momentum constraint becomes homogeneous:
 *
 * \begin{equation}\label{eq:mom_constraint}
 * \nabla_j \bar{A}^{ij} = 0
 * \end{equation}
 *
 * Here, $\nabla$ is the flat-space covariant derivate. $\bar{A}^{ij}$ is the
 * conformal traceless extrinsic curvature that composes the extrinsic curvature
 * as
 *
 * \begin{equation}
 * K_{ij} = \psi^{-2} \bar{A}_{ij} + \frac{1}{3} \gamma_{ij} K
 * \end{equation}
 *
 * (where $K=0$ under maximal slicing). $\psi$ is the conformal factor that
 * composes the spatial metric as
 *
 * \begin{equation}
 * \gamma_{ij} = \psi^4 \bar{\gamma}_{ij}
 * \end{equation}
 *
 * (where $\bar{\gamma}_{ij} = \delta{ij}$ under conformal flatness and in
 * Cartesian coordinates).
 *
 * The momentum constraint ($\ref{eq:mom_constraint}$) is solved analytically by
 * the Bowen-York extrinsic curvature
 *
 * \begin{equation}
 * \bar{A}^{ij} = \frac{3}{2} \frac{1}{r_C} \left(
 * 2 P^{(i} n^{j)} - (\delta^{ij} - n^i n^j) P^k n^k
 * + \frac{4}{r_C} n^{(i} \epsilon^{j)kl} S^k n^l\right)
 * \end{equation}
 *
 * representing a black hole with linear momentum $\mathbf{P}$ and angular
 * momentum $\mathbf{S}$ at position $\mathbf{C}$. The quantity
 * $r_C=||\mathbf{x}-\mathbf{C}||$ is the Euclidean coordinate distance to the
 * black hole, and $\mathbf{n}=(\mathbf{x}-\mathbf{C})/r_C$ is the radial unit
 * normal to the black hole. Since the momentum constraint is linear, any
 * superposition of $\bar{A}^{ij}$ is also a solution to the momentum
 * constraint, allowing to represent multiple black holes.
 *
 * Only the Einstein Hamiltonian constraint remains to be solved numerically for
 * the conformal factor:
 *
 * \begin{equation}
 * \nabla^2 \psi = \frac{1}{8} \psi^{-7} \bar{A}_{ij} \bar{A}^{ij}
 * \end{equation}
 *
 * It reduces to the puncture equation ($\ref{eq:puncture_eqn}$) when we
 * decompose the conformal factor as:
 *
 * \begin{equation}
 * \psi = 1 + \frac{1}{\alpha} + u
 * \end{equation}
 *
 * where we define
 *
 * \begin{equation}
 * \frac{1}{\alpha} = \sum_I \frac{M_I}{2 r_I}
 * \end{equation}
 *
 * and
 *
 * \begin{equation}
 * \beta = \frac{1}{8} \alpha^7 \bar{A}_{ij} \bar{A}^{ij}.
 * \end{equation}
 *
 * Here, $M_I$ is the "puncture mass" (or "bare mass") parameter for the $I$th
 * black hole at position $\mathbf{C}_I$, and $\bar{A}_{ij}$ is the
 * superposition of the Bowen-York extrinsic curvature of the black holes with
 * the parameters defined above. Note that the definition of $\frac{1}{\alpha}$
 * in Eq. (12.51) in \cite BaumgarteShapiro is missing factors of $\frac{1}{2}$,
 * but their Eq. (3.23) includes them, as does Eq. (8) in
 * \cite BrandtBruegmann1997 (though the latter includes the unit offset in
 * the definition of $u$).
 */
namespace Punctures {}
