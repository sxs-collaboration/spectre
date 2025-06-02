// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <array>
#include <cstddef>

#include "NumericalAlgorithms/Spectral/Mesh.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/TMPL.hpp"

/// \cond
class DataVector;
/// \endcond

/*!
 * \brief Items for assessing truncation error in spectral methods.
 */
namespace PowerMonitors {

/// @{
/*!
 * \ingroup SpectralGroup
 * \brief Returns array of power monitors in each spatial dimension.
 *
 * Computed following Sec. 5.1 of Ref. \cite Szilagyi2014fna.
 * For example, in the x dimension (indexed by \f$ k_0 \f$), we compute
 *
 * \f{align*}{
 *  P_{k_0}[\psi] = \sqrt{ \frac{1}{N_1 N_2}
 *   \sum_{k_1,k_2} \left| C_{k_0,k_1,k_2} \right|^2} ,
 * \f}
 *
 * where \f$ C_{k_0,k_1,k_2}\f$ are the modal coefficients
 * of variable \f$ \psi \f$.
 *
 */
template <typename VectorType, size_t Dim>
void power_monitors(gsl::not_null<std::array<DataVector, Dim>*> result,
                    const VectorType& u, const Mesh<Dim>& mesh);

template <typename VectorType, size_t Dim>
std::array<DataVector, Dim> power_monitors(const VectorType& u,
                                           const Mesh<Dim>& mesh);
/// @}

/// @{
/*!
 * \ingroup SpectralGroup
 * \brief Compute the relative truncation error.
 *
 * The negative logarithm of this quantity is defined by Eqs. (57) and
 * (58) of Ref. \cite Szilagyi2014fna, i.e.,
 *
 * \f{align*}{
 *  \mathcal{T}\left[P_k\right] = \log_{10} \max \left(P_0, P_1\right)
 *   - \dfrac{\sum_{j=0}^{j_{\text{max}, k}} \log_{10} \left(P_j\right) w_j}
 *   {\sum_{j=0}^{j_{\text{max}, k}} w_j} , \f}
 *
 * with weights
 *
 * \f{align*}{
 *  w_j = \exp\left[ - \left(j - j_{\text{max}, k}
 *          + \dfrac{1}{2}\right)^2 \right] .
 * \f}
 *
 * where \f$ j_{\text{max}, k}  = N_k - 1 \f$ and  \f$ N_k \f$ is the number of
 * modes or gridpoints in dimension k. Here the second term is a weighted
 * average with larger weights toward the highest modes.
 *
 * \note Modes below a cutoff of $100 \epsilon \mathrm{max}_k(P_k)$ are ignored
 * in the weighted average, where $\epsilon$ is the machine epsilon. This
 * ensures that we don't underestimate the truncation error if some modes are
 * zero (e.g. by symmetry). Furthermore, if the last two or more modes are zero,
 * we assume that the function is represented exactly and return a relative
 * truncation error of zero.
 *
 * \details The number of modes (`num_modes_to_use`) argument needs to be less
 * or equal than the total number of power monitors (`power_monitor.size()`).
 * In contrast with Ref. \cite Szilagyi2014fna, here we index the modes starting
 * from zero.
 *
 */
double relative_truncation_error(const DataVector& power_monitor,
                                 size_t num_modes_to_use);
/// @}

/*!
 * \brief The relative truncation error in each logical direction of the grid
 *
 * This overload is intended for visualization purposes only. It takes a tensor
 * component as input, so it can be used as a kernel to post-process volume data
 * with Python bindings (see `TransformVolumeData.py`).
 */
template <typename VectorType, size_t Dim>
std::array<double, Dim> relative_truncation_error(
    const VectorType& tensor_component, const Mesh<Dim>& mesh);

/// @{
/*!
 * \ingroup SpectralGroup
 * \brief Returns an estimate of the absolute truncation error in each
 * dimension.
 *
 * The estimate of the numerical error is given by
 *
 * \f{align*}{
 *  \mathcal{E}\left[P_k\right] = u_\mathrm{max} \times 10^{- \mathcal{T}[P_k]},
 * \f}
 *
 * where \f$ u_\mathrm{max} = \mathrm{max} |u|\f$ in the corresponding element
 * and \f$ \mathcal{T}[P_k] \f$ is the relative error estimate
 * computed from the power monitors \f$ P_k \f$.
 *
 * \warning This estimate is intended for visualization purposes only.
 */
template <typename VectorType, size_t Dim>
std::array<double, Dim> absolute_truncation_error(
    const VectorType& tensor_component, const Mesh<Dim>& mesh);
/// @}

/// Holds convergence rate and pile up modes of a power monitor
struct ConvergenceInfo {
  double convergence_rate{std::numeric_limits<double>::signaling_NaN()};
  double number_of_pile_up_modes{std::numeric_limits<double>::signaling_NaN()};
};

/*!
 * \ingroup SpectralGroup
 * \brief Returns the convergence rate and the number of pile up modes of a
 * power monitor as a ConvergenceInfo.
 *
 * \details Computes the convergence rate of a power monitor as a weighted
 * average of slopes measured using different subsets of spectral modes
 * in the power monitor. Equation (53) of \cite Szilagyi2014fna gives
 * the convergence rate $\mathcal{C}$ in terms of a power monitor $P_k$ as
 * \begin{equation}
 * \mathcal{C}(P_k) = -\frac{\sum_{k_1=0}^2\sum_{k_2=\tilde{k}_1}^{\tilde{N}-1}
 *   \frac{\mathcal{S}(k_1,k_2)}{\epsilon + \mathcal{E}(k_1,k_2)}}{
 *   \sum_{k_1=0}^2\sum_{k_2=\tilde{k}_1}^{\tilde{N}-1}
 *   \frac{1}{\epsilon + \mathcal{E}(k_1,k_2)}}.
 * \end{equation}
 * Here, $\mathcal{S}(k_1,k_2)$ is the slope of a linear regression fit of
 * $\log_{10}(P_k)$ with $k$ satisfying $k_1\leq k \leq k_2$,
 * $\mathcal{E}(k_1,k_2)$ is the error of the slope in that fit,
 * $\epsilon=\max\left(10^{-3}\max\left(\mathcal{E}(k_1,k_2)\right),
 * 10^{-15}\right)$ is a small number to avoid dividing by zero in the event the
 * fit errors vanish,
 * $\max\left(\mathcal{E}(k_1,k_2)\right)$ is the maximum fit error of each
 * fit whose slope is included in the summation,
 * $\tilde{k}_1 = \min\left(k_1+4,\tilde{N}-1\right)$,
 * $\tilde{N} = N-N_f$, $N$ is the number of modes in the
 * power monitor, and the highest $N_f$ modes are filtered. Note that the
 * way $\epsilon$ is defined is so that it matches SpEC's definition, while
 * also ensuring that it is nonzero even if the error in the slope fit is
 * exactly zero.
 *
 * Also computes the number of pile up modes in a power monitor. Pile up
 * modes are modes where the power is no longer converging at the overall
 * convergence rate. Following Eq. (56) of \cite Szilagyi2014fna, the number of
 * pile up modes $\mathcal{P}$ is defined as
 * \begin{equation}
 * \mathcal{P}(P_k) = \sum_{j=2}^{\tilde{N}-2}
 * \exp\left[-32\left(\frac{\tilde{\mathcal{C}}_j}
 * {\mathcal{C}(P_k)}\right)^2\right],
 * \end{equation}
 * where $\mathcal{C}(P_k)$ is the convergence rate of the power monitor $P_k$,
 * the local convergence rate $\tilde{\mathcal{C}}_j$ of mode $j$ is
 * \begin{equation}
 * \tilde{\mathcal{C}}_j = -\mathcal{S}(j,\min(\tilde{N}-1,j+4)),
 * \end{equation}
 * $\mathcal{S}(k_1,k_2)$ is the slope of a linear regression fit of
 * $\log_{10}(P_k)$ with $k$ satisfying $k_1\leq k \leq k_2$,
 * $\tilde{N} = N-N_f$, $N$ is the number of modes in the
 * power monitor, and the highest $N_f$ modes are filtered.
 * The motivation of this definition is the following: if the local
 * convergence rate $\tilde{\mathcal{C}}_j$ is comparable to the overall
 * convergence rate $\mathcal{C}(P_k)$, then the $j^{\rm th}$ term in the
 * summation becomes $\approx \exp(-32) \approx 10^{-14}$, while if
 * $\tilde{\mathcal{C}}_j \ll \mathcal{C}(P_k)$, then the $j^{\rm th}$ term in
 * the summation is $\approx \exp(0) = 1$. Note that the coefficient value 32
 * is chosen to agree with SpEC.
 * \note The summation goes up to $\tilde{N}-2$ so that there is it least one
 * larger unfiltered mode for use in computing the slope. The highest mode
 * used when computing the slope is the highest unfiltered mode, $\tilde{N}-1$.
 * Mode numbers in the power monitor are zero based. These choices are off by
 * one vs. Eqs. (55) and (56) of \cite Szilagyi2014fna, because those formulas
 * apparently assume one-based indexing.
 * \param power_monitor The power monitor.
 * \param number_of_filtered_modes How many of the highest modes of the
 * power monitor are filtered (default 0).
 */
ConvergenceInfo convergence_rate_and_number_of_pile_up_modes(
    const DataVector& power_monitor, size_t number_of_filtered_modes = 0);
}  // namespace PowerMonitors
