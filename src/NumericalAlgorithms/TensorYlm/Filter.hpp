// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>
#include <optional>

#include "DataStructures/SimpleSparseMatrix.hpp"
#include "DataStructures/Tensor/TypeAliases.hpp"
#include "DataStructures/VariablesDeclaration.hpp"
#include "NumericalAlgorithms/TensorYlm/TensorYlm.hpp"
#include "Utilities/Gsl.hpp"

/// \cond
class DataVector;
/// \endcond

namespace ylm::TensorYlm {

/*!
 * \brief Fills a sparse matrix that does a TensorYlm filter operation.
 *
 * If the CoefficientNormalization parameter is set to Standard, then
 * assumes that $T^{\tilde A}_{\ell' m'}$ is stored in a
 * Tensor<DataVector>.  Multiplying (one plus) the resulting
 * sparse matrix by the Tensor<DataVector> is equivalent to
 * evaluating the right-hand side of Eq. $(\ref{eq:Filter})$.
 * If the CoefficientNormalization parameter is set to Spherepack, then
 * assumes the matrix will multiply a Tensor<DataVector> containing
 * ${\breve T}^{\tilde A}_{\ell' m'}$ and the filter operation is equivalent to
 * Eq. $(\ref{eq:FilterSpherepack})$.
 *
 * We assume that the independent components of the Tensor<DataVector>
 * are stored contiguously in memory, in order of the storage_index of
 * the Tensor.  However, we do allow for a stride.  This means that we
 * can point to memory starting with the first element of
 * $T^{\tilde A}_{\ell' m'}$ and multiply that by the sparse matrix we compute
 * here, and get the filtered result.
 *
 * The memory layout here is different than in SpEC.  In SpEC, each
 * tensor component is stored in separately-allocated memory, so the
 * SpEC equivalent of the fill_filter function fills $N^2$ sparse
 * matrices, where $N$ is the number of independent components of the
 * Tensor. The advantage of the SpEC method is that each sparse matrix
 * is smaller, so sorting elements into the correct order while
 * constructing each sparse matrix is faster (sorting is > linear in
 * the number of matrix elements).  The disadvantage of the SpEC
 * method is that evaluating the filter for a single Tensor involves
 * $N^2$ matrix-vector multiplications, whereas here evaluating the
 * filter involves only one matrix-vector multiplication, which should
 * have more efficient memory access.  It is not clear which method is
 * faster overall without more profiling.
 *
 * ## Explicit formulas
 *
 * The following formulas come from Klinger and Scheel, in prep.
 *
 * For rank-0 tensors, the expression is simple because there
 * is no change of basis, only a filter based on $\ell$.
 * \begin{align}
 *  F_{l m \tilde{D}}^{\ell'' m''\tilde{A}} &=
 *  \delta(\tilde{D},\tilde{A})\delta_{\ell \ell''}\delta_{m m''}
 *  \left[1-
 *  \delta(\ell_{\mathrm{cut}}^-\leq \ell \leq \ell_{\mathrm{max}}) g(\ell)
 *  \right].
 * \end{align}
 *
 * For rank-1 tensors, the expression for
 * $F_{l m \tilde{D}}^{\ell'' m''\tilde{A}}$ is
 * \begin{align}
 *  F_{l m \tilde{D}}^{\ell'' m''\tilde{A}} &=
 *  \delta(\tilde{D},\tilde{A})\delta_{\ell \ell''}\delta_{m m''}
 *  \nonumber \\
 *  &- (-1)^{m+m''} (-1)^{\delta(\tilde{D},\mathbf{e}_y)}
 *  \delta(\ell,\ell'')
 *  \sum_{\ell'=\ell_{\mathrm{cut}}^-}^{\ell_{\mathrm{max}}+1}
 *  \frac{2\ell'+1}{2} g(\ell') \nonumber \\
 *  &\times \sum_{j,p,m'} k_j(\tilde{D})k_p(\tilde{A})
 *    \left(\begin{array}{rrr}
 *     \ell&\ell'&1\cr
 *      -m''&m'&m_j(\tilde{D})
 *    \end{array}\right)
 *    \left(\begin{array}{rrr}
 *     \ell&\ell'&1\cr
 *      -m&m'&m_p(\tilde{A})
 *    \end{array}\right),
 * \end{align}
 * where the 6-element "matrices"
 * in parentheses are Wigner 3-J symbols, and where
 * \begin{align}
 *  g(\ell') &=
 *  \left\{\begin{array}{lr}
 *      1-f(\ell') & \ell' \leq \ell_{\mathrm{cut}}^+,\\
 *      1 & \ell' > \ell_{\mathrm{cut}}^+.
 *  \end{array}\right.
 * \end{align}
 *
 * For second-rank tensors, the expression for
 * $F_{l m \tilde{D}}^{\ell'' m''\tilde{A}}$ is
 * \begin{align}
 *  F_{l m \tilde{D}}^{\ell'' m''\tilde{A}}
 *  &=
 *  \delta(\tilde{D},\tilde{A})\delta_{\ell \ell''}\delta_{m m''}
 * \nonumber \\
 * &-
 *  \frac{1}{4}
 * (-1)^{\delta(\tilde{D}_1,\mathbf{e}_y)}
 * (-1)^{\delta(\tilde{D}_2,\mathbf{e}_y)}
 *  \delta_{\ell \ell''}
 * \nonumber \\
 * &\times
 * \sum_{\ell'=\ell_{\mathrm{cut}}^-}^{\ell_{\mathrm{max}}+2}
 * (2\ell'+1) g(\ell')
 * \sum_{u,v,p,q,\bar{\ell},\tilde{m},\bar{m},m'}
 * (2 \bar{\ell}+1) S(\bar{\ell},\tilde{D})
 * k_u(\tilde{D}_1) k_v(\tilde{D}_2)
 * k_p(\tilde{A}_1) k_q(\tilde{A}_2)
 * \nonumber \\
 * &\times
 *    \left(\begin{array}{rrr}
 *     \ell&\ell'&\bar{\ell}\cr
 *      -m&m'&\bar{m}
 *    \end{array}\right)
 *    \left(\begin{array}{rrr}
 *     1&1&\bar{\ell}\cr
 *      m_p(\tilde{A}_1)&m_q(\tilde{A}_2)&-\bar{m}
 *    \end{array}\right)
 * \nonumber \\
 * &\times
 *    \left(\begin{array}{rrr}
 *     \ell'&\ell&\bar{\ell}\cr
 *      -m'&m''&\tilde{m}
 *    \end{array}\right)
 *    \left(\begin{array}{rrr}
 *     1&1&\bar{\ell}\cr
 *      m_u(\tilde{D}_1)&m_v(\tilde{D}_2)&\tilde{m}
 *    \end{array}\right),
 *  \label{eq:RankTwoTransformWithCut}
 * \end{align}
 * where the factor $S(\bar{\ell},\tilde{D})$ is a symmetry factor.
 * For a 2nd-rank tensor with no symmetries, $S(\bar{\ell},\tilde{D})$ is
 * unity, but for a tensor symmetric in $(\tilde{D_1},\tilde{D_2})$
 * the matrix elements $F_{l m \tilde{D}}^{\ell'' m''\tilde{A}}$ are
 * multiplied only by tensor components with $\tilde{D_1}\geq\tilde{D_2}$
 * and the symmetry is accounted for by setting
 * \begin{align}
 *   S(\bar{\ell},\tilde{D}) &=
 *   (1+(-1)^{\bar{\ell}})\frac{2-\delta(\tilde{D_1},\tilde{D_2})}{2}.
 * \end{align}
 *
 * For third-rank tensors, the expression for
 * $F_{l m \tilde{D}}^{\ell'' m''\tilde{A}}$ is
 * \begin{align}
 *  F_{l m \tilde{D}}^{\ell'' m''\tilde{A}}
 *  &=
 *  \delta(\tilde{D},\tilde{A})\delta_{\ell \ell''}\delta_{m m''}
 *  \nonumber \\
 * &-
 *  \frac{1}{8}
 *  (-1)^{\delta(\tilde{D}_1,\mathbf{e}_y)}
 *  (-1)^{\delta(\tilde{D}_2,\mathbf{e}_y)}
 *  (-1)^{\delta(\tilde{D}_3,\mathbf{e}_y)}
 *  \delta_{\ell \ell''}
 *  \nonumber \\
 *  &\times
 *  \sum_{\ell'=\ell_{\mathrm{cut}}^-}^{\ell_{\mathrm{max}}+3}
 *  (2\ell'+1) g(\ell')
 *  \sum_{u,v,w,p,q,r,\tilde{m},\bar{\ell},\bar{m},
 *  \hat{\ell},\hat{m},\check{m},m'}
 *  (2 \bar{\ell}+1) S(\bar{\ell},\tilde{D})
 *  (2 \hat{\ell}+1)
 *  \nonumber \\
 *  &\qquad\times
 *  k_u(\tilde{D}_2) k_v(\tilde{D}_3) k_w(\tilde{D}_1)
 *  k_p(\tilde{A}_2) k_q(\tilde{A}_3) k_r(\tilde{A}_1)
 *  (-1)^{\tilde{m}-\bar{m}}
 *  \nonumber \\
 *  &\qquad\times
 *    \left(\begin{array}{rrr}
 *     1&1&\bar{\ell}\cr
 *     m_p(\tilde{A}_2)&m_q(\tilde{A}_3)&-\bar{m}
 *    \end{array}\right)
 *    \left(\begin{array}{rrr}
 *     1&1&\bar{\ell}\cr
 *     m_u(\tilde{D}_2)&m_v(\tilde{D}_3)&\tilde{m}
 *    \end{array}\right)
 *  \nonumber \\
 *  &\qquad\times
 *    \left(\begin{array}{rrr}
 *     \ell&\ell'&\hat{\ell}\cr
 *     -m&m'&\check{m}
 *    \end{array}\right)
 *    \left(\begin{array}{rrr}
 *     \ell'&\ell&\hat{\ell}\cr
 *     -m'&m''&\hat{m}
 *    \end{array}\right)
 *  \nonumber \\
 *  &\qquad\times
 *    \left(\begin{array}{rrr}
 *     1&\bar{\ell}&\hat{\ell}\cr
 *     m_r(\tilde{A}_1)&\bar{m}&-\check{m}
 *    \end{array}\right)
 *    \left(\begin{array}{rrr}
 *     1&\bar{\ell}&\hat{\ell}\cr
 *     m_w(\tilde{D}_1)&-\tilde{m}&\hat{m}
 *    \end{array}\right).
 *    \label{eq:RankThreeTransformWithCut}
 * \end{align}
 * As in the rank-2 case,
 * $S(\bar{\ell},\tilde{D})$ is a symmetry factor that is unity
 * for a tensor with no symmetries and is
 * \begin{align}
 *   S(\bar{\ell},\tilde{D}) &=
 *   (1+(-1)^{\bar{\ell}})\frac{2-\delta(\tilde{D_2},\tilde{D_3})}{2}
 * \end{align}
 * for a tensor symmetric on its last two indices. We don't consider
 * other symmetries because we don't find them in the cases we care
 * about.
 *
 * \tparam TensorStructure A Tensor_detail::Structure
 * \tparam SparseMatrixType A sparse matrix fillable by SparseMatrixFiller
 *
 * \param matrix The sparse matrix to fill
 * \param ell_max The maximum ylm ell value.
 * \param number_of_ell_modes_to_kill How many top ell modes to set to zero.
 * \param half_power The half power $\sigma$ for more complicated filtering.
 * \param coefficient_normalization Describes the normalization of coefficients.
 *
 *  If half_power is std::nullopt, implements a Heaviside filter.
 *  Otherwise, the filter is the more complicated one described in the
 *  TensorYlm namespace documentation, with $\sigma$ equal to
 *  half_power and $\ell_{\mathrm{cut}}^+$ equal to $\ell_{\rm max}$
 *  minus number_of_ell_modes_to_kill.
 *
 */
template <typename TensorStructure, typename SparseMatrixType>
void fill_filter(gsl::not_null<SparseMatrixType*> matrix, size_t ell_max,
                 size_t number_of_ell_modes_to_kill,
                 std::optional<size_t> half_power,
                 CoefficientNormalization coefficient_normalization);

/*!
 * \brief Holds the filtering matrices for different matrix ranks.
 *
 * This is used for caching the matrices that are needed for a particular
 * system.
 *
 * Since the data being cached depends on parameters, those parameters are also
 * cached.
 */
struct FilterMatrixHolder {
  size_t number_of_ell_modes_to_kill{};
  CoefficientNormalization coefficient_normalization{};
  std::optional<size_t> half_power;

  std::optional<SimpleSparseMatrix> scalar;
  std::optional<SimpleSparseMatrix> i;
  std::optional<SimpleSparseMatrix> ii;
  std::optional<SimpleSparseMatrix> ij;
  std::optional<SimpleSparseMatrix> kii;
};

/*!
 * \brief Fill the FilterMatrixHolder for the specific TagList (system).
 */
template <typename TagList>
void fill_tensor_ylm_filters(
    gsl::not_null<FilterMatrixHolder*> matrix, size_t ell_max,
    size_t number_of_ell_modes_to_kill, std::optional<size_t> half_power,
    CoefficientNormalization coefficient_normalization);

/*!
 * \brief Applies TensorYlm filter in place to variables.
 *
 * When \p radial_extents is 1, `vars` and `temp_storage` are assumed to
 * be defined on a spherical slice, with number of grid points
 * corresponding to a spherical-harmonic grid of `ell_max`, and the
 * filter happens only on that slice.
 *
 * When \p radial_extents is > 1, `vars` and `temp_storage` are assumed to
 * be defined on a spherical shell of topology I1 x S2. The filter
 * happens in the entire volume, internally iterating over each
 * spherical slice at a time.
 *
 * For performance reasons, `apply_tensor_ylm_filter` does not allocate
 * or deallocate memory, but takes a `temp_storage` buffer.  The size of
 * `temp_storage` should at least `radial_extents*spectral_size*num_components`,
 * where `num_components` is the total number of independent components in the
 * variable list (e.g. 5 for curved scalar wave and 50 for generalized
 * harmonic), and `spectral_size` is the size of the S2 Spherepack spectral
 * coefficient array for `ell_max`, as obtained from the member function
 * ylm::Spherepack::spectral_size().  Note that for S2 on Spherepack, the number
 * of collocation points is different than the number of spectral coefficients,
 * and both are different than the size of the Spherepack storage array.
 *
 * \param vars Variables at collocation points to filter.
 * \param temp_storage Temporary storage for the variables,
 *   allocated outside apply_tensor_ylm_filter. See above for size requirements.
 * \param jac_inertial_to_grid Jacobian taking V_x from inertial to grid.
 * \param jac_grid_to_inertial Jacobian taking V_x from grid to inertial.
 * \param filter_matrices The bundle of filter matrices computed by
 *   fill_filter. The members used depend on the ranks present in `TagList`.
 * \param ell_max The maximum ylm ell.
 * \param radial_extents The number of radial grid points, can be 1 for slices.
 */
template <typename TagList>
void apply_tensor_ylm_filter(
    gsl::not_null<Variables<TagList>*> vars,
    gsl::not_null<Variables<TagList>*> temp_storage,
    const InverseJacobian<DataVector, 3, Frame::Inertial, Frame::Grid>&
        jac_inertial_to_grid,
    const InverseJacobian<DataVector, 3, Frame::Grid, Frame::Inertial>&
        jac_grid_to_inertial,
    const FilterMatrixHolder& filter_matrices, size_t ell_max,
    size_t radial_extents);
}  // namespace ylm::TensorYlm
