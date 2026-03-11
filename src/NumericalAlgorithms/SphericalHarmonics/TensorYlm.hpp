// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

/*!
 * \brief Holds functions that converts spatial tensors between scalar-Ylm
 * and tensor-Ylm basis, and functions that filter by doing such a conversion.
 *
 * \details We expand a spatial tensor of arbitrary rank
 * in terms of Cartesian components as follows:
 * \begin{align}
 * {\mathbf T} &= \sum_{\ell,m,\tilde{A}} {}_0 Y_{\ell m}
 *                T^{\tilde A}_{\ell m}\mathbf{e}_{\tilde A},
 * \end{align}
 * where ${}_0 Y_{\ell m}$ are the usual scalar spherical harmonics,
 * $T^{\tilde A}_{\ell m}$ are complex-valued expansion coefficients,
 * $\tilde A$ is a multi-index
 * $\tilde{A}=(\tilde{a}_1,\tilde{a}_2,\tilde{a}_3,\ldots)$, where each
 * of the $\tilde{a}_i$ take on values from 0 to 2, referring to
 * either $\mathbf{e}_x$, $\mathbf{e}_y$, or $\mathbf{e}_z$.
 * The sum over $\tilde A$ goes over all $\tilde A$ of the same rank.
 *
 * Similarly, we can expand the same spatial
 * tensor in terms of a complex tetrad:
 * \begin{align}
 * {\mathbf T} &= \sum_{\ell,m,A} {}_{s(\!A\!)}Y_{\ell m}
 *                T^A_{\ell m} \mathbf{e}_A,
 * \end{align}
 * where ${}_sY_{\ell m}$ are the spin-weighted harmonics,
 * $A$ is a multi-index $A=(a_1,a_2,a_3,\ldots)$, where each of the
 * $a_i$ refer to either $\mathbf{l}$,$\mathbf{m}$, or $\mathbf{\bar{m}}$,
 * and $s(\!A\!)$ is the spin weight, which is a function of
 * $A$: each $\mathbf{l}$ in $A$ adds the value zero, each $\mathbf{m}$
 * adds the value -1, and each $\mathbf{\bar{m}}$ adds the value +1.
 * (These values are the spin weights of the complex conjugates
 * of the basis vectors.)
 *
 * To allow for a compact notation where we don't need to write different
 * equations for different combinations of basis vectors,
 * we define two auxiliary quantities $k_j$ and $m_j$
 * that are functions of the Cartesian basis vectors:
 * \begin{align}
 *  k_j(\mathbf{e}_x) &= -j,\\
 *  k_j(\mathbf{e}_y) &=  i,\\
 *  k_j(\mathbf{e}_z) &= 1/\sqrt{2},\\
 *  m_j(\mathbf{e}_x) &= j,\\
 *  m_j(\mathbf{e}_y) &= j,\\
 *  m_j(\mathbf{e}_z) &= 0,
 * \end{align}
 * where the $i$ above is the imaginary unit $i=\sqrt{-1}$.
 * These quantities will be used in explicit formulas for transformations
 * between expansion coefficients.
 *
 * ### Spherepack coefficients
 *
 * Instead of using the actual spherical-harmonic or
 * spin-weighted spherical-harmonic coefficients $T^{B}_{\ell' m'}$ and
 * $T^{\tilde A}_{\ell'm'}$, we often work with
 * coefficients computed by Spherepack, which have a different normalization
 * and phase than the standard coefficients.  While Spherepack internally
 * stores real matrices it calls $a_{lm}$ and $b_{lm}$ (see Spherepack
 * documentation for details), we simplify things by defining
 * complex Spherepack coefficients
 * ${\breve T}^{B}_{\ell' m'}$ and ${\breve T}^{\tilde A}_{\ell'm'}$ that
 * have Spherepack normalization. These are related to the standard coefficients
 * by
 * \begin{align}
 *  {\breve T}^{A}_{\ell m} &= (-1)^m \sqrt{\frac{2}{\pi}} T^{B}_{\ell m},\\
 *  {\breve T}^{\tilde A}_{\ell m} &= (-1)^m \sqrt{\frac{2}{\pi}}
 *  T^{\tilde A}_{\ell m}.
 * \end{align}
 * Some of the formulas below are slightly different when acting on
 * Spherepack coefficients compared to standard coefficients.
 *
 * ## Transformations
 *
 * We can define the following transformations between the expansion
 * coefficients $T^A_{\ell m}$ and $T^{\tilde A}_{\ell m}$:
 * \begin{align}
 * T^{\tilde B}_{\ell' m'} &=
 * \sum_{\ell,m,A} C_{\ell' m' A}^{\ell m\tilde{B}} T^A_{\ell m},\\
 * T^{B}_{\ell' m'} &= \sum_{\ell m \tilde{A}}
 *                  C_{\ell' m'\tilde{A}}^{\ell mB}
 *                  T^{\tilde A}_{\ell m},
 * \end{align}
 * where the above transformations define the coefficients
 * $C_{\ell' m' A}^{\ell m\tilde{B}}$ and $C_{\ell' m'\tilde{A}}^{\ell mB}$.
 * Analytic expressions for these coefficients are derived in Klinger
 * and Scheel (in prep) and will be used here to compute the coefficients.
 *
 * For real-valued tensors, the coefficients $T^A_{\ell m}$ and
 * $T^{\tilde A}_{\ell m}$ for negative $m$ can be computed from the
 * complex conjugates of the same coefficients with positive $m$.  Therefore
 * we store and loop over only nonnegative $m$. This means we can write
 * \begin{align}
 * T^{\tilde B}_{\ell m} &= \sum_{A, \ell',m'\geq 0}
 *                  \left[
 *              C_{\ell m A}^{\ell' m'\tilde{B}} T^A_{\ell' m'}
 *               +{\hat C}_{\ell m A}^{\ell' m'\tilde{B}} T^A_{\ell' m'}{}^\star
 *                \right]\label{eq:S2C},\\
 * T^{B}_{\ell m} &= \sum_{\tilde{A},\ell',m'\geq 0}
 *                  \left[
 *                  C_{\ell m\tilde{A}}^{\ell' m' B}
 *                  T^{\tilde A}_{\ell' m'}
 *                  +{\hat C}_{\ell m\tilde{A}}^{\ell' m' B}
 *                  T^{\tilde A}_{\ell' m'}{}^\star
 *                  \right]\label{eq:C2S},
 * \end{align}
 * where
 * \begin{align}
 * {\hat C}_{\ell m A}^{\ell' m'\tilde{B}} &=
 *                 (1-\delta_{0m'})C_{\ell m A^\star}^{\ell' -m'\tilde{B}}
 *                 (-1)^{S(A)+m'},\\
 * {\hat C}_{\ell m\tilde{A}}^{\ell' m' B} &=
 *                (1-\delta_{0m'})C_{\ell m \tilde{A}}^{\ell' -m'B}
 *                 (-1)^{m'}.
 * \end{align}
 *
 * The functions FillCartToSphere and FillSphereToCart fill
 * sparse matrices that encode Eqs. $(\ref{eq:C2S})$ and
 * $(\ref{eq:S2C})$. When working on Spherepack coefficients, the functions
 * FillCartToSphere and FillSphereToCart instead
 * fill sparse matrices that encode
 * \begin{align}
 *  {\breve T}^{\tilde B}_{\ell m} &= (-1)^m\sum_{A, \ell',m'\geq 0}(-1)^{m'}
 *                  \left[
 *              C_{\ell m A}^{\ell' m'\tilde{B}} T^A_{\ell' m'}
 *               +{\hat C}_{\ell m A}^{\ell' m'\tilde{B}}
 *                {\breve T}^A_{\ell' m'}{}^\star
 *                \right]\label{eq:S2CSpherepack},\\
 *  {\breve T}^{B}_{\ell m} &= (-1)^m\sum_{\tilde{A},\ell',m'\geq 0}(-1)^{m'}
 *                  \left[
 *                  C_{\ell m\tilde{A}}^{\ell' m' B}
 *                  T^{\tilde A}_{\ell' m'}
 *                  +{\hat C}_{\ell m\tilde{A}}^{\ell' m' B}
 *                  {\breve T}^{\tilde A}_{\ell' m'}{}^\star
 *                  \right]\label{eq:C2SSpherepack}.
 * \end{align}
 *
 * ## Filtering
 *
 * Consider starting with coefficients $T^{\tilde A}_{\ell' m'}$,
 * transforming to spin-weighted harmonics, applying a filter to
 * the spin-weighted harmonic coefficients, and transforming back.
 * We can write this entire operation as
 * \begin{align}
 *   F_{\ell m \tilde{D}}^{\ell'' m''\tilde{A}} &=
 *   \sum_{\ell'=0}^{\ell_{\mathrm{cut}}^--1} C^{\ell' m' \tilde{A}}_{\ell m B}
 *   C^{\ell'' m'' B}_{\ell'  m' \tilde{D}}
 * + \sum_{\ell'=\ell_{\mathrm{cut}}^-}^{\ell_{\mathrm{cut}}^+}
 *   C^{\ell' m' \tilde{A}}_{\ell m B}
 *   C^{\ell'' m'' B}_{\ell'  m' \tilde{D}} f(\ell'),
 * \end{align}
 * where $f(\ell')$ is a filter function,
 * $\ell_{\mathrm{cut}}^{-}$ is the smallest $\ell$ mode that is filtered,
 * and
 * $\ell_{\mathrm{cut}}^{+}$ is the largest $\ell$ mode that is retained.
 *
 * There are two cases we care about.
 * The first is simple "Heaviside filtering", where
 * $\ell_{\mathrm{cut}}^{-} = \ell_{\mathrm{cut}}^{+}$ and $f(\ell')=1$.
 *
 * The second case is
 * \begin{align}
 *  f(\ell') &= \exp\left[-\alpha
 * \left(\frac{\ell'}{\ell_{\mathrm{cut}}^++1}\right)^{2 \sigma}\right],\\
 * \ell_{\mathrm{cut}}^- &= \mathrm{ceil}\left[(\ell_{\mathrm{cut}}^++1)
 *    \left(\frac{\epsilon}{\alpha}\right)^{1/(2\sigma)}\right],
 * \end{align}
 * where $\alpha$ is a parameter we choose to be 36, $\sigma$ is an integer
 * parameter typically between 28 and 32, and
 * $\epsilon$ is machine roundoff.
 * Note that the filter remains smooth at $\ell' = \ell_{\mathrm{cut}}^-$ and
 * it reduces to the simple filter as $\sigma \to \infty$.
 *
 * As with the transformations, we sum over only nonnegative $m$, so we can
 * write the action of the filter as
 * \begin{align}
 * T^{\tilde B}_{\ell m} &= \sum_{{\tilde A}, \ell',m'\geq 0}
 *                  \left[
 *              F_{\ell m {\tilde A}}^{\ell' m'\tilde{B}}
 *                T^{\tilde A}_{\ell' m'}
 *               +{\hat F}_{\ell m {\tilde A}}^{\ell' m'\tilde{B}}
 *               T^{\tilde A}_{\ell' m'}{}^\star
 *                \right]\label{eq:Filter},
 * \end{align}
 * where
 * \begin{align}
 * {\hat F}_{\ell m\tilde{A}}^{\ell' m' \tilde{B}} &=
 *                (1-\delta_{0m'})F_{\ell m \tilde{A}}^{\ell' -m' \tilde{B}}
 *                 (-1)^{m'}.
 * \end{align}
 *
 * When filtering Spherepack coefficients, the expression is
 * \begin{align}
 * {\breve T}^{\tilde B}_{\ell m} &=
 *           (-1)^m\sum_{{\tilde A}, \ell',m'\geq 0}(-1)^{m'}
 *                  \left[
 *              F_{\ell m {\tilde A}}^{\ell' m'\tilde{B}}
 *                {\breve T}^{\tilde A}_{\ell' m'}
 *               +{\hat F}_{\ell m {\tilde A}}^{\ell' m'\tilde{B}}
 *               {\breve T}^{\tilde A}_{\ell' m'}{}^\star
 *                \right]\label{eq:FilterSpherepack}.
 * \end{align}
 *
 * The function FillFilter encapsulates Eq. $(\ref{eq:Filter})$ or
 * Eq. $(\ref{eq:FilterSpherepack})$,
 * depending what type of coefficients are being filtered.
 */
namespace ylm::TensorYlm {

/// Instances of this class are passed to FillFilter,
/// FillCartToSphere, and FillSphereToCart to identify how the
/// coefficients are normalized.  This makes a difference because the
/// normalization is not just a constant factor.
enum class CoefficientNormalization {
  /// The standard normalization of the spherical-harmonic coefficients.
  Standard,
  /// The Spherepack normalization. Spherepack coefficients are standard
  /// coefficients multiplied by $(-1)^m \sqrt{2/\pi}$.
  Spherepack
};
}  // namespace ylm::TensorYlm
