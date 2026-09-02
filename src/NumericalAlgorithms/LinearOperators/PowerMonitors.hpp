// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <array>
#include <cstddef>
#include <limits>
#include <vector>

#include "DataStructures/DataVector.hpp"
#include "NumericalAlgorithms/Spectral/Mesh.hpp"
#include "NumericalAlgorithms/TensorYlm/Helpers.hpp"
#include "Utilities/ConstantExpressions.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/TMPL.hpp"

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
 * For dimensions using a Fourier basis, the cosine and sine power for each
 * wavenumber \f$ k \f$ are combined via \f$ P_k = \sqrt{P_{\cos,k}^2 +
 * P_{\sin,k}^2} \f$, so the output size for a Fourier dimension of \f$ N \f$
 * points is \f$ N/2 + 1 \f$ (integer division) rather than \f$ N \f$.
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

/// @{
/*!
 * \brief Return the radial power monitor for a tensor component on a
 * spherical shell.
 *
 * The mesh dimensions are assumed to be ordered `(radial, theta, phi)`. The
 * radial grid points are contiguous, so each angular point supplies one
 * radial slice to the one-dimensional modal transform.
 */
void spherical_shell_radial_power_monitor(gsl::not_null<DataVector*> result,
                                          const DataVector& tensor_component,
                                          const Mesh<3>& mesh);

DataVector spherical_shell_radial_power_monitor(
    const DataVector& tensor_component, const Mesh<3>& mesh);
/// @}

/// @{
/*!
 * \brief Return the angular power monitor for one TensorYlm component on a
 * spherical shell.
 *
 * The mesh dimensions are assumed to be ordered `(radial, theta, phi)`, with
 * `l_max == m_max`. TensorYlm coefficients use the radial dimension as the
 * fastest-moving extent. As reviewed in Sec. II of \cite Boyle2023,
 * spin-weighted spherical harmonics with `l < |spin_weight|` vanish, so these
 * modes are omitted from both the sum and its normalization. Set
 * `zero_m_is_real` for real scalar coefficients, which have no imaginary
 * `m=0` coefficients in Spherepack storage.
 */
void spherical_shell_angular_power_monitor(
    gsl::not_null<DataVector*> result, const DataVector& tensor_ylm_component,
    const Mesh<3>& mesh, int spin_weight, bool zero_m_is_real);

DataVector spherical_shell_angular_power_monitor(
    const DataVector& tensor_ylm_component, const Mesh<3>& mesh,
    int spin_weight, bool zero_m_is_real);
/// @}

/*!
 * \brief Return the RMS radial power monitor across all components of a
 * tensor on a spherical shell.
 *
 * Combines `spherical_shell_radial_power_monitor` for each of
 * `tensor.size()` components in quadrature, normalized by the number of
 * components.
 */
template <typename TensorType>
DataVector spherical_shell_tensor_radial_power_monitor(const TensorType& tensor,
                                                       const Mesh<3>& mesh) {
  DataVector squared_power(mesh.extents(0), 0.0);
  DataVector component_power{};
  for (size_t component = 0; component < tensor.size(); ++component) {
    spherical_shell_radial_power_monitor(make_not_null(&component_power),
                                         tensor[component], mesh);
    squared_power += square(component_power);
  }
  squared_power = sqrt(squared_power / static_cast<double>(tensor.size()));
  return squared_power;
}

/*!
 * \brief Number of independent TensorYlm coefficients contributing to
 * angular degree `ell`, summed over `radial_extents` radial points.
 *
 * Returns 0 when `ell < |spin_weight|`, since spin-weighted spherical
 * harmonics vanish there (such terms are then excluded from
 * `accumulate_spherical_shell_tensor_angular_power()`'s sum and normalization).
 * See `SpherepackIterator` for the meaning of `zero_m_is_real`.
 */
size_t spherical_shell_number_of_angular_coefficients(size_t ell,
                                                      int spin_weight,
                                                      bool zero_m_is_real,
                                                      size_t radial_extents);

/*!
 * \brief Accumulate weighted-squared angular TensorYlm power and mode counts
 * for all components of a tensor on a spherical shell.
 *
 * Adds to `weighted_squared_power` and `counts` in place, so this can be
 * called repeatedly to combine several tensors into the same angular power
 * monitor before calling `normalize_spherical_shell_angular_power`.
 */
template <typename TensorType>
void accumulate_spherical_shell_tensor_angular_power(
    const gsl::not_null<DataVector*> weighted_squared_power,
    const gsl::not_null<std::vector<size_t>*> counts, const TensorType& tensor,
    const Mesh<3>& mesh) {
  const size_t radial_extents = mesh.extents(0);
  const size_t ell_max = mesh.extents(1) - 1;
  constexpr bool zero_m_is_real = TensorType::rank() == 0;
  DataVector component_power{};
  for (size_t component = 0; component < tensor.size(); ++component) {
    const int spin_weight = ylm::TensorYlm::helpers::component_spin_weight<
        typename TensorType::structure>(component);
    spherical_shell_angular_power_monitor(make_not_null(&component_power),
                                          tensor[component], mesh, spin_weight,
                                          zero_m_is_real);
    for (size_t ell = 0; ell <= ell_max; ++ell) {
      const size_t component_count =
          spherical_shell_number_of_angular_coefficients(
              ell, spin_weight, zero_m_is_real, radial_extents);
      (*weighted_squared_power)[ell] +=
          static_cast<double>(component_count) * square(component_power[ell]);
      (*counts)[ell] += component_count;
    }
  }
}

/*!
 * \brief Normalize an angular power monitor accumulator in place.
 *
 * Sets `power[ell] = sqrt(power[ell] / counts[ell])`, or 0 when
 * `counts[ell] == 0`.
 */
void normalize_spherical_shell_angular_power(gsl::not_null<DataVector*> power,
                                             const std::vector<size_t>& counts);

/*!
 * \brief Accumulate squared ZernikeB3 Jacobi spectral coefficients for one
 * TensorYlm component into radial and angular power bins.
 *
 * `spec_buf` has layout `spec_buf[s * n_r + i_r]` where `s` is the SPHEREPACK
 * offset and `i_r` is the radial collocation index. Modes are binned radially
 * via `radial_mode = (n_total + 1) / 2` where `n_total = l + 2 * k_spec`, and
 * angularly by degree \f$\ell\f$. Modes with `l < |spin_weight|` are skipped.
 *
 * `offsets_by_l` must be pre-computed for the correct `zero_m_is_real` value
 * of this component. `gathered` and `modal_buf` are caller-owned scratch
 * buffers of size `max_n_modes_l * n_r` each, where
 * `max_n_modes_l = 2 * (l_max + 1)`.
 */
void accumulate_b3_tensor_component_sums(
    gsl::not_null<DataVector*> sum_sq_radial,
    gsl::not_null<DataVector*> counts_radial,
    gsl::not_null<DataVector*> sum_sq_angular,
    gsl::not_null<DataVector*> counts_angular, const double* spec_buf,
    size_t n_r, size_t n_r_max, int spin_weight,
    const std::vector<std::vector<size_t>>& offsets_by_l, double* gathered,
    double* modal_buf);

/*!
 * \brief Accumulate squared ZernikeB3 spectral coefficients for all components
 * of a TensorYlm tensor into radial and angular power bins.
 *
 * Selects `offsets_by_l_real` for rank-0 (scalar) tensors and
 * `offsets_by_l_complex` for all higher-rank tensors. The spin weight for each
 * component is determined from the tensor structure.
 */
template <typename TensorType>
void accumulate_b3_tensor_sums(
    gsl::not_null<DataVector*> sum_sq_radial,
    gsl::not_null<DataVector*> counts_radial,
    gsl::not_null<DataVector*> sum_sq_angular,
    gsl::not_null<DataVector*> counts_angular, const TensorType& tensor,
    size_t n_r, size_t n_r_max,
    const std::vector<std::vector<size_t>>& offsets_by_l_real,
    const std::vector<std::vector<size_t>>& offsets_by_l_complex,
    double* gathered, double* modal_buf) {
  constexpr bool zero_m_is_real = TensorType::rank() == 0;
  const auto& offsets_by_l =
      zero_m_is_real ? offsets_by_l_real : offsets_by_l_complex;
  for (size_t component = 0; component < tensor.size(); ++component) {
    const int spin_weight = ylm::TensorYlm::helpers::component_spin_weight<
        typename TensorType::structure>(component);
    accumulate_b3_tensor_component_sums(
        sum_sq_radial, counts_radial, sum_sq_angular, counts_angular,
        tensor[component].data(), n_r, n_r_max, spin_weight, offsets_by_l,
        gathered, modal_buf);
  }
}

/*!
 * \brief Normalize a B3 power monitor accumulator in place.
 *
 * Sets `result[i] = sqrt(sum_sq[i] / counts[i])`, or 0 when `counts[i] == 0`.
 */
void normalize_b3_power(gsl::not_null<DataVector*> result,
                        const DataVector& sum_sq, const DataVector& counts);

}  // namespace PowerMonitors
