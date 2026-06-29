// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include "NumericalAlgorithms/TensorYlm/TensorYlm.hpp"

#include <cstddef>

#include "Utilities/Gsl.hpp"

namespace ylm::TensorYlm {

/*!
 * \brief Fills a sparse matrix that does a TensorYlm Cartesian
 * to Spherical operation.
 *
 * If the CoefficientNormalization parameter is set to Standard, then
 * assumes that the input, $T^{\tilde A}_{\ell' m'}$, is stored in a
 * Tensor<DataVector>.  Multiplying the resulting
 * sparse matrix by the Tensor<DataVector> is equivalent to
 * evaluating the right-hand side of Eq. $(\ref{eq:C2S})$.
 * If the CoefficientNormalization parameter is set to Spherepack, then
 * assumes the matrix will multiply a Tensor<DataVector> containing
 * ${\breve T}^{\tilde A}_{\ell' m'}$ and the transform operation is
 * equivalent to Eq. $(\ref{eq:C2SSpherepack})$.
 *
 * We assume that the independent components of the Tensor<DataVector>
 * are stored contiguously in memory, in order of the `storage_index` of
 * the Tensor.  However, we do allow for a stride.  This means that we
 * can point to memory starting with the first element of
 * $T^{\tilde A}_{\ell' m'}$ and multiply that by the sparse matrix we compute
 * here, and get the result.
 *
 * The memory layout here is different than in SpEC.  In SpEC, each
 * tensor component is stored in separately-allocated memory, so the
 * SpEC equivalent of the fill_cart_to_sphere function fills $N^2$ sparse
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
 * $C_{\ell' m'\tilde{D}}^{\ell'' m'' B}$ is
 * \begin{align}
 *   C_{\ell' m'\tilde{D}}^{\ell'' m'' B} &=
 *  \sqrt{\frac{(2\ell'+1)(2\ell''+1)}{2}}
 *  (-1)^{m''} (-1)^{\delta(\tilde{D},\mathbf{e}_y)} (-1)^{\delta(s_B,-1)}
 *    \left(\begin{array}{ccc}
 *     \ell' & \ell'' & 1 \cr
 *      s_B & 0 & -s_B
 *    \end{array}\right)
 *  \nonumber \\
 *  &\times \sum_j k_j(\tilde{D})
 *    \left(\begin{array}{ccc}
 *     \ell' & \ell'' & 1 \cr
 *      -m' & m'' & -m_j(\tilde{D})
 *    \end{array}\right),
 * \end{align}
 * where the 6-element "matrices"
 * in parentheses are Wigner 3-J symbols.
 *
 * For second-rank tensors, the expression for
 * $C_{\ell' m'\tilde{D}}^{\ell'' m'' B}$ is
 * \begin{align}
 *  C_{\ell' m'\tilde{D}}^{\ell'' m'' B}
 *      &=
 *    \frac{1}{2}(-1)^{\delta(s_{B_1},-1)}(-1)^{\delta(s_{B_2},-1)}(-1)^{m'}
 *    (-1)^{\delta(\tilde{D}_1,\mathbf{e}_y)}
 *    (-1)^{\delta(\tilde{D}_2,\mathbf{e}_y)}
 *    \sqrt{(2\ell''+1)(2\ell'+1)}
 *    \nonumber \\
 *    &\times
 *    \sum_{u,v,\tilde{\ell},\tilde{m},\tilde{s}}
 *    (2 \tilde{\ell}+1) (-1)^{\tilde{s}} S(\tilde{\ell},\tilde{D})
 *    k_u(\tilde{D}_1) k_v(\tilde{D}_2)
 *    \left(\begin{array}{ccc}
 *     \ell' & \ell'' & \tilde{\ell} \cr
 *      -m' & m'' & \tilde{m}
 *    \end{array}\right)
 *    \nonumber \\
 *    &\qquad\times
 *    \left(\begin{array}{ccc}
 *     \ell' & \ell'' & \tilde{\ell} \cr
 *      s_{B_1}+s_{B_2} & 0 & -\tilde{s}
 *    \end{array}\right)
 *    \left(\begin{array}{ccc}
 *      1 & 1 & \tilde{\ell} \cr
 *      -m_u(\tilde{D}_1)&-m_v(\tilde{D}_2)&-\tilde{m}
 *    \end{array}\right)
 *    \left(\begin{array}{ccc}
 *      1 & 1 & \tilde{\ell} \cr
 *      -s_{B_1}&-s_{B_2}&\tilde{s}
 *    \end{array}\right),
 * \end{align}
 * where the factor $S(\tilde{\ell},\tilde{D})$ is a symmetry factor.
 * For a 2nd-rank tensor with no symmetries, $S(\tilde{\ell},\tilde{D})$ is
 * unity, but for a tensor symmetric in $(\tilde{D_1},\tilde{D_2})$
 * the matrix elements $C_{\ell' m'\tilde{D}}^{\ell'' m'' B}$ are
 * multiplied only by tensor components with $\tilde{D_1}\geq\tilde{D_2}$
 * and the symmetry is accounted for by setting
 * \begin{align}
 *   S(\tilde{\ell},\tilde{D}) &=
 *   (1+(-1)^{\tilde{\ell}})\frac{2-\delta(\tilde{D_1},\tilde{D_2})}{2}.
 * \end{align}
 *
 * For third-rank tensors, the expression for
 * $C_{\ell' m'\tilde{D}}^{\ell'' m'' B}$ is
 * \begin{align}
 *   C_{\ell' m'\tilde{D}}^{\ell'' m'' B}
 *     &=
 *        (-1)^{m'}
 *        (-1)^{\delta(s_{B_1},-1)}
 *        (-1)^{\delta(s_{B_2},-1)}
 *        (-1)^{\delta(s_{B_3},-1)}
 *     (-1)^{\delta(\tilde{D}_1,\mathbf{e}_y)}
 *     (-1)^{\delta(\tilde{D}_2,\mathbf{e}_y)}
 *     (-1)^{\delta(\tilde{D}_3,\mathbf{e}_y)}
 *     \nonumber \\
 *     &\times
 *     \sqrt{\frac{(2\ell''+1)(2\ell'+1)}{8}}
 *     \sum_{u,v,w,\tilde{\ell},\tilde{m},\tilde{s},\hat{\ell},\hat{m},\hat{s}}
 *     (2 \tilde{\ell}+1) (2 \hat{\ell}+1) (-1)^{\tilde{s}+\hat{s}+\tilde{m}}
 *     k_u(\tilde{D}_2) k_v(\tilde{D}_3) k_w(\tilde{D}_1)
 *     \nonumber \\
 *     &\qquad\times
 *    S(\tilde{\ell},\tilde{D})
 *    \left(\begin{array}{ccc}
 *     \ell' & \ell'' & \hat{\ell} \cr
 *      -m' & m'' & \hat{m}
 *    \end{array}\right)
 *     \left(\begin{array}{ccc}
 *     \ell' & \ell'' & \hat{\ell} \cr
 *      s_{B_1}+s_{B_2}+s_{B_3} & 0 & -\hat{s}
 *    \end{array}\right)
 *     \nonumber \\
 *     &\qquad\times
 *    \left(\begin{array}{ccc}
 *      1 & 1 & \tilde{\ell} \cr
 *      -m_u(\tilde{D}_2)&-m_v(\tilde{D}_3)&-\tilde{m}
 *    \end{array}\right)
 *     \left(\begin{array}{ccc}
 *      1 & 1 & \tilde{\ell} \cr
 *      -s_{B_2}&-s_{B_3}&\tilde{s}
 *     \end{array}\right)
 *     \nonumber \\
 *     &\qquad\times
 *     \left(\begin{array}{ccc}
 *      1 & \tilde{\ell} & \hat{\ell} \cr
 *      -m_w(\tilde{D}_1)&\tilde{m}&-\hat{m}
 *     \end{array}\right)
 *     \left(\begin{array}{ccc}
 *      1 & \tilde{\ell} & \hat{\ell} \cr
 *      -s_{B_1}&-\tilde{s}&\hat{s}
 *     \end{array}\right).
 * \end{align}
 * As in the rank-2 case,
 * $S(\tilde{\ell},\tilde{D})$ is a symmetry factor that is unity
 * for a tensor with no symmetries and is
 * \begin{align}
 *   S(\tilde{\ell},\tilde{D}) &=
 *   (1+(-1)^{\tilde{\ell}})\frac{2-\delta(\tilde{D_2},\tilde{D_3})}{2}
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
 * \param coefficient_normalization Describes the normalization of coefficients.
 *
 */
template <typename TensorStructure, typename SparseMatrixType>
void fill_cart_to_sphere(gsl::not_null<SparseMatrixType*> matrix,
                         size_t ell_max,
                         CoefficientNormalization coefficient_normalization);

}  // namespace ylm::TensorYlm
