// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include "NumericalAlgorithms/SphericalHarmonics/TensorYlm.hpp"

#include <cstddef>

#include "Utilities/Gsl.hpp"

namespace ylm::TensorYlm {

/*!
 * \brief Fills a sparse matrix that does a TensorYlm Cartesian
 * to Spherical operation.
 *
 * Assumes that the input, $T^B_{\ell m}$, is stored in a
 * Tensor<DataVector>.  Multiplying the resulting
 * sparse matrix by the Tensor<DataVector> is equivalent to
 * evaluating the right-hand side of Eq. $(\ref{eq:S2C})$.
 *
 * We assume that the independent components of the Tensor<DataVector>
 * are stored contiguously in memory, in order of the `storage_index` of
 * the Tensor.  However, we do allow for a stride.  This means that we
 * can point to memory starting with the first element of
 * $T^B_{\ell m}$ and multiply that by the sparse matrix we compute
 * here, and get the result.
 *
 * The memory layout here is different than in SpEC.  In SpEC, each
 * tensor component is stored in separately-allocated memory, so the
 * SpEC equivalent of the fill_sphere_to_cart function fills $N^2$ sparse
 * matrices, where $N$ is the number of independent components of the
 * Tensor. The advantage of the SpEC method is that each sparse matrix
 * is smaller, so sorting elements into the correct order while
 * constructing each sparse matrix is faster (sorting is > linear in
 * the number of matrix elements).  The disadvantage of the SpEC
 * method is that evaluating the coefficients for a single Tensor involves
 * $N^2$ matrix-vector multiplications, whereas here evaluating the
 * coefficients involves only one matrix-vector multiplication, which should
 * have more efficient memory access.  It is not clear which method is
 * faster overall without more profiling.
 *
 * ## Explicit formulas
 *
 * The following formulas come from Klinger and Scheel, in prep.
 *
 * For rank-1 tensors, the expression for
 * $C^{\ell' m' \tilde{A}}_{\ell m B}$ is
 * \begin{align}
 * C^{\ell' m' \tilde{A}}_{\ell m B} &=
 * (-1)^{\delta(s_B,-1)-m}\sqrt{\frac{(2 \ell+1)(2 \ell'  + 1)}{2}}
 * \sum_j k_j(\tilde{A})
 *    \left(\begin{array}{ccc}
 *     \ell & \ell' & 1 \cr
 *      0 & -s_B & s_B
 *    \end{array}\right)
 *    \left(\begin{array}{ccc}
 *     \ell & \ell' & 1 \cr
 *      -m & m' & m_j(\tilde{A})
 *    \end{array}\right),
 * \end{align}
 * where the 6-element "matrices"
 * in parentheses are Wigner 3-J symbols.
 *
 * For second-rank tensors, the expression for
 * $C^{\ell' m' \tilde{A}}_{\ell m B}$ is
 * \begin{align}
 *  C^{\ell' m' \tilde{A}}_{\ell m B}
 *    &=
 *    \frac{1}{2}(-1)^{\delta(s_{B_1},-1)}(-1)^{\delta(s_{B_2},-1)}(-1)^{m'}
 *    \sqrt{(2\ell+1)(2\ell'+1)}
 *    \nonumber \\
 *    &\times
 *    \sum_{p,q,\bar{\ell},\bar{m},\bar{s}}
 *    (2 \bar{\ell}+1) (-1)^{\bar{s}}
 *    k_p(\tilde{A}_1) k_q(\tilde{A}_2) S(\bar{\ell},B)
 *    \left(\begin{array}{ccc}
 *     \ell & \ell' & \bar{\ell} \cr
 *      0&-s_{B_1}-s_{B_2}&-\bar{s}
 *    \end{array}\right)
 *    \nonumber \\
 *    &\qquad\times
 *    \left(\begin{array}{ccc}
 *     \ell & \ell' & \bar{\ell} \cr
 *      -m & m' & \bar{m}
 *    \end{array}\right)
 *    \left(\begin{array}{ccc}
 *     1 & 1 & \bar{\ell} \cr
 *      m_p(\tilde{A}_1)&m_q(\tilde{A}_2)&-\bar{m}
 *    \end{array}\right)
 *    \left(\begin{array}{ccc}
 *     1 & 1 & \bar{\ell} \cr
 *     s_{B_1}&s_{B_2}&\bar{s}
 *    \end{array}\right),
 * \end{align}
 * where the factor $S(\bar{\ell},B)$ is a symmetry factor.
 * For a 2nd-rank tensor with no symmetries, $S(\bar{\ell},B)$ is
 * unity, but for a tensor symmetric in $(B_1,B_2)$
 * the matrix elements $C^{\ell' m' \tilde{A}}_{\ell m B}$
 * are multiplied only by tensor components with $B_1\geq B_2$
 * and the symmetry is accounted for by setting
 * \begin{align}
 *   S(\bar{\ell},B) &=
 *   (1+(-1)^{\bar{\ell}})\frac{2-\delta(B_1,B_2)}{2}.
 * \end{align}
 *
 * For third-rank tensors, the expression for
 * $C^{\ell' m' \tilde{A}}_{\ell m B}$ is
 * \begin{align}
 *  C^{\ell' m' \tilde{A}}_{\ell m B}
 *    &=
 *       (-1)^{m'}
 *       (-1)^{\delta(s_{B_1},-1)}
 *       (-1)^{\delta(s_{B_2},-1)}
 *       (-1)^{\delta(s_{B_3},-1)}
 *    \sqrt{\frac{(2\ell+1)(2\ell'+1)}{8}}
 *    \nonumber \\
 *    &\times
 *     \sum_{p,q,r,\bar{\ell},\bar{m},\bar{s},\check{\ell},\check{m},\check{s}}
 *    (2 \bar{\ell}+1) (2 \check{\ell}+1) (-1)^{\bar{s}+\check{s}-\bar{m}}
 *    k_p(\tilde{A}_2) k_q(\tilde{A}_3) k_r(\tilde{A}_1) S(\bar{\ell},B)
 *    \nonumber \\
 *    &\qquad\times
 *    \left(\begin{array}{ccc}
 *     \ell & \ell' & \check{\ell} \cr
 *      -m & m' & \check{m}
 *    \end{array}\right)
 *    \left(\begin{array}{ccc}
 *     \ell & \ell' & \check{\ell} \cr
 *      0&-s_{B_1}-s_{B_2}-s_{B_3}&-\check{s}
 *    \end{array}\right)
 *    \nonumber \\
 *    &\qquad\times
 *    \left(\begin{array}{ccc}
 *     1 & 1 & \bar{\ell} \cr
 *     m_p(\tilde{A}_2)&m_q(\tilde{A}_3)&-\bar{m}
 *    \end{array}\right)
 *    \left(\begin{array}{ccc}
 *     1 & 1 & \bar{\ell} \cr
 *     s_{B_2}&s_{B_3}&\bar{s}
 *    \end{array}\right)
 *    \nonumber \\
 *    &\qquad\times
 *    \left(\begin{array}{ccc}
 *     1 & \bar{\ell} & \check{\ell} \cr
 *     m_r(\tilde{A}_1)&\bar{m}&-\check{m}
 *    \end{array}\right)
 *    \left(\begin{array}{ccc}
 *     1 & \bar{\ell} & \check{\ell} \cr
 *     s_{B_1}&-\bar{s}&\check{s}
 *    \end{array}\right),
 * \end{align}
 * As in the rank-2 case,
 * $S(\bar{\ell},B)$ is a symmetry factor that is unity
 * for a tensor with no symmetries and is
 * \begin{align}
 *   S(\bar{\ell},B) &=
 *   (1+(-1)^{\bar{\ell}})\frac{2-\delta(B_2,B_3)}{2}.
 * \end{align}
 * for a tensor symmetric on its last two indices. We don't consider
 * other symmetries because we don't find them in the cases we care
 * about.
 *
 * \tparam TensorStructure A Tensor_detail::Structure
 * \tparam SparseMatrixType A sparse matrix fillable by SparseMatrixFiller
 *
 * \param matrix The sparse matrix to fill
 * \param ell_max The maximum ylm ell value
 *
 */
template <typename TensorStructure, typename SparseMatrixType>
void fill_sphere_to_cart(gsl::not_null<SparseMatrixType*> matrix,
                         size_t ell_max);

}  // namespace ylm::TensorYlm
