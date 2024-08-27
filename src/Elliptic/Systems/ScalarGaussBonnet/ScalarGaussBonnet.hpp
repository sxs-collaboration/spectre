// Distributed under the MIT License.
// See LICENSE.txt for details.

/// \file
/// Documents the `sgb` namespace

#pragma once

/*!
 * \ingroup EllipticSystemsGroup
 * \brief Items related to solving the sGB scalar equation
 *
 * The quasi-stationary scalar equation in sGB gravity
 * \begin{equation}\label{eq:sGB}
 * -\partial_i \left[ \left( \gamma^{ij} - \alpha^{-2} \beta^i \beta^j \right)
 * \partial_j \Psi \right] + \partial_j \Psi \left( \gamma^{ij} - \alpha^{-2}
 * \beta^i \beta^j \right) \left( \Gamma_i + \partial_i \ln \alpha \right) =
 * \ell^2 f' \left( \Psi \right) \mathcal{G}
 * \end{equation}
 *
 * is a nonlinear Poisson-type elliptic PDE for the scalar field $\Psi$. To
 * obtain this equation, one begins by considering the action:
 *
 * \begin{equation}
 * S\left[g_{ab}, \Psi \right] \equiv \int \, d^4 x \sqrt{-g}
 * \Big[ \dfrac{R}{2 \kappa } - \dfrac{1}{2}  \nabla_{a} \Psi \nabla^{a} \Psi +
 * \ell^2 f(\Psi) \, \mathcal{G} \Big],
 * \end{equation}
 *
 * where $\mathcal{G} \equiv R_{abcd}R^{abcd} - 4 R_{ab}R^{ab} + R^2$. Varying
 * the action with respect to $\Psi$, one obtains the wave-like equation
 *
 * \begin{equation}
 * \Box \Psi = - \ell^2 f'(\Psi) \mathcal{G}
 * \end{equation}
 *
 * In the spirit of quasi-stationarity, we set $\partial_t \Psi = \partial_t^2
 * \Psi = \partial_t \alpha = \partial_t \beta^{i} = 0$, where $\alpha$ and
 * $\beta^i$ are the lapse and shift respectively. This yields ($\ref{eq:sGB}$),
 * with $\gamma^{ij}$ being the spatial metric, and $\Gamma_i$ its associated
 * contracted christoffel symbol of the second kind.
 *
 * Currently, we have implemented the coupling function
 *
 * \begin{equation}
 * \ell^2 f(\Psi) = \epsilon_2 \frac{\Psi^2}{8} + \epsilon_4 \frac{\Psi^4}{16}
 * \end{equation}
 *
 * Note that the principal part of the master equation will generically turn
 * singular at black hole horizon's for stationary initial data. Typically, this
 * requires a boundary condition ensuring regularity of the solution. However,
 * as detailed in \cite Nee2024bur, this is already enforced by the chosen
 * spectral decomposition, and so instead one should impose the DoNothing
 * boundary condition for excision surfaces within black hole apparent horizons.
 *
 * As is currently implemented, one must provide numeric data corresponding to
 * the full metric $g_{ab}$. All of this can be generated using SolveXcts,
 * with a glob for the volume files being specified in the
 * SolveScalarGaussBonnet input file.
 *
 * Specifically, one must ensure the following is in the provided volume file:
 *
 * - the conformal factor $\psi$,
 * - the spatial metric $\gamma_{ij}$,
 * - the lapse $\alpha$,
 * - the shift $\beta^i$,
 * - the shift excess $\beta_{exc}^i$,
 * - the extrinsic curvature $K^{ij}$, and
 * - the inverse conformal metric $\bar{\gamma}^{ij}$.
 *
 * One must also ensure that the Background specified in the input file is the
 * same one that was used in SolveXcts. This is checked using
 * $\bar{\gamma}^{ij}$.
 */
namespace sgb {}
