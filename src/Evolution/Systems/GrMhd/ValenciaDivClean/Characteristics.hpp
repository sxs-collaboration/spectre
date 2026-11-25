// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <array>
#include <cstddef>

#include "DataStructures/DataBox/Tag.hpp"
#include "DataStructures/Tensor/EagerMath/Magnitude.hpp"
#include "DataStructures/Tensor/TypeAliases.hpp"
#include "Domain/FaceNormal.hpp"
#include "Evolution/Systems/GrMhd/ValenciaDivClean/Tags.hpp"
#include "PointwiseFunctions/GeneralRelativity/TagsDeclarations.hpp"
#include "PointwiseFunctions/Hydro/EquationsOfState/EquationOfState.hpp"
#include "PointwiseFunctions/Hydro/TagsDeclarations.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/TMPL.hpp"

/// \cond
class DataVector;
namespace Tags {
template <typename Tag>
struct Normalized;
}  // namespace Tags
/// \endcond

namespace grmhd {
namespace ValenciaDivClean {
/// @{
/*!
 * \brief Compute the characteristic speeds for the Valencia formulation of
 * GRMHD with divergence cleaning.
 *
 * Obtaining the exact form of the characteristic speeds involves the solution
 * of a nontrivial quartic equation for the fast and slow modes. Here we make
 * use of a common approximation in the literature (e.g. \cite Gammie2003)
 * where the resulting characteristic speeds are analogous to those of the
 * Valencia formulation of the 3-D relativistic Euler system
 * (see RelativisticEuler::Valencia::characteristic_speeds),
 *
 * \f{align*}
 * \lambda_2 &= \alpha \Lambda^- - \beta_n,\\
 * \lambda_{3, 4, 5, 6, 7} &= \alpha v_n - \beta_n,\\
 * \lambda_{8} &= \alpha \Lambda^+ - \beta_n,
 * \f}
 *
 * with the substitution
 *
 * \f{align*}
 * c_s^2 \longrightarrow c_s^2 + v_A^2(1 - c_s^2)
 * \f}
 *
 * in the definition of \f$\Lambda^\pm\f$. Here \f$v_A\f$ is the Alfvén
 * speed. In addition, two more speeds corresponding to the divergence cleaning
 * mode and the longitudinal magnetic field are added,
 *
 * \f{align*}
 * \lambda_1 = -\alpha - \beta_n,\\
 * \lambda_9 = \alpha - \beta_n.
 * \f}
 *
 * \note The ordering assumed here is such that, in the Newtonian limit,
 * the exact expressions for \f$\lambda_{2, 8}\f$, \f$\lambda_{3, 7}\f$,
 * and \f$\lambda_{4, 6}\f$ should reduce to the
 * corresponding fast modes, Alfvén modes, and slow modes, respectively.
 * See \cite Dedner2002 for a detailed description of the hyperbolic
 * characterization of Newtonian MHD.  In terms of the primitive variables:
 *
 * \f{align*}
 * v^2 &= \gamma_{mn} v^m v^n \\
 * c_s^2 &= \frac{1}{h} \left[ \left( \frac{\partial p}{\partial \rho}
 * \right)_\epsilon +
 * \frac{p}{\rho^2} \left(\frac{\partial p}{\partial \epsilon}
 * \right)_\rho \right] \\
 * v_A^2 &= \frac{b^2}{b^2 + \rho h} \\
 * b^2 &= \frac{1}{W^2} \gamma_{mn} B^m B^n + \left( \gamma_{mn} B^m v^n
 * \right)^2
 * \f}
 *
 * where \f$\gamma_{mn}\f$ is the spatial metric, \f$\rho\f$ is the rest
 * mass density, \f$W = 1/\sqrt{1-v_i v^i}\f$ is the Lorentz factor, \f$h = 1 +
 * \epsilon + \frac{p}{\rho}\f$ is the specific enthalpy, \f$v^i\f$ is the
 * spatial velocity, \f$\epsilon\f$ is the specific internal energy, \f$p\f$ is
 * the pressure, and \f$B^i\f$ is the spatial magnetic field measured by an
 * Eulerian observer.
 */

template <size_t ThermodynamicDim>
std::array<DataVector, 9> characteristic_speeds_approximate_mhd(
    const Scalar<DataVector>& rest_mass_density,
    const Scalar<DataVector>& electron_fraction,
    const Scalar<DataVector>& specific_internal_energy,
    const Scalar<DataVector>& specific_enthalpy,
    const tnsr::I<DataVector, 3, Frame::Inertial>& spatial_velocity,
    const Scalar<DataVector>& lorentz_factor,
    const tnsr::I<DataVector, 3, Frame::Inertial>& magnetic_field,
    const Scalar<DataVector>& lapse, const tnsr::I<DataVector, 3>& shift,
    const tnsr::ii<DataVector, 3, Frame::Inertial>& spatial_metric,
    const tnsr::i<DataVector, 3>& unit_normal,
    const EquationsOfState::EquationOfState<true, ThermodynamicDim>&
        equation_of_state);

template <size_t ThermodynamicDim>
void characteristic_speeds_approximate_mhd(
    gsl::not_null<std::array<DataVector, 9>*> char_speeds,
    const Scalar<DataVector>& rest_mass_density,
    const Scalar<DataVector>& electron_fraction,
    const Scalar<DataVector>& specific_internal_energy,
    const Scalar<DataVector>& specific_enthalpy,
    const tnsr::I<DataVector, 3, Frame::Inertial>& spatial_velocity,
    const Scalar<DataVector>& lorentz_factor,
    const tnsr::I<DataVector, 3, Frame::Inertial>& magnetic_field,
    const Scalar<DataVector>& lapse, const tnsr::I<DataVector, 3>& shift,
    const tnsr::ii<DataVector, 3, Frame::Inertial>& spatial_metric,
    const tnsr::i<DataVector, 3>& unit_normal,
    const EquationsOfState::EquationOfState<true, ThermodynamicDim>&
        equation_of_state);
/// @}

/*!
 * \brief Labels for the characteristic speeds of the relativistic hydrodynamics
 * system.
 *
 * \see `grmhd::ValenciaDivClean::characteristic_speeds_hydro`
 */
namespace HydroSpeed {
/// The smallest speed, $\lambda_-$
constexpr size_t LambdaMinus = 0;
/// The normal velocity, $v_n = n_i v^i$
constexpr size_t NormalDotVelocity = 1;
/// The largest speed, $\lambda_+$
constexpr size_t LambdaPlus = 2;
}  // namespace HydroSpeed

/// @{
/*!
 * \brief Compute the characteristic speeds for the relativistic hydrodynamics
 * system in the Eulerian frame.
 *
 * These are the eigenvalues of the flux Jacobian projected along a spatial
 * direction with unit normal \f$ s_i\f$, measured by an Eulerian observer.
 * They consist of four degenerate modes and two non-degenerate modes:
 *
 * \f{align*}
 * \lambda_\mathrm{deg} &= v_n ,\\
 * \lambda_{\pm} &=
 *   \frac{ (1 - c_s^2)\,v_n \pm c_s\,d / W^2 }
 *        { 1 - v^2 c_s^2 }.
 * \f}
 *
 * where
 *
 * \f{align*}
 * d &\equiv
 *   W\,\sqrt{ 1 - v^2 c_s^2 - v_n^2(1 - c_s^2) } .
 * \f}
 *
 * The variables are defined as:
 *
 * \f{align*}
 * v^2 &= \gamma_{mn} v^m v^n ,\\
 * v_n &= v^a s_a ,\\
 * W &= \frac{1}{\sqrt{1 - v^a v_a}} ,\\
 * h &= 1 + \epsilon + \frac{p}{\rho} ,\\
 * c_s^2 &= \frac{1}{h}\left(\chi + \kappa \frac{p}{\rho^2}\right) ,
 * \f}
 *
 * The returned tensor has index 0..2, in increasing order (see `HydroSpeed`):
 * - `characteristic_speeds.get(0)`: acoustic speed \f$\lambda_-\f$
 * - `characteristic_speeds.get(1)`: degenerate speed \f$v_n\f$
 * - `characteristic_speeds.get(2)`: acoustic speed \f$\lambda_+\f$
 *
 * \see `grmhd::ValenciaDivClean::numerical_characteristics` for how these
 * speeds are paired with characteristic modes and characteristic projectors.
 */

template <size_t ThermodynamicDim>
void characteristic_speeds_hydro(
    gsl::not_null<tnsr::i<DataVector, 3>*> characteristic_speeds,

    /* primitive variables */
    const tnsr::I<DataVector, 3, Frame::Inertial>& spatial_velocity,
    const Scalar<DataVector>& rest_mass_density,
    const Scalar<DataVector>& specific_internal_energy,
    const Scalar<DataVector>& electron_fraction,

    /* other helpful quantities */
    const Scalar<DataVector>& lorentz_factor,
    const Scalar<DataVector>& specific_enthalpy,
    const tnsr::ii<DataVector, 3, Frame::Inertial>& spatial_metric,
    const tnsr::i<DataVector, 3>& unit_normal,
    const EquationsOfState::EquationOfState<true, ThermodynamicDim>&
        equation_of_state);

template <size_t ThermodynamicDim>
tnsr::i<DataVector, 3> characteristic_speeds_hydro(
    const tnsr::I<DataVector, 3, Frame::Inertial>& spatial_velocity,
    const Scalar<DataVector>& rest_mass_density,
    const Scalar<DataVector>& specific_internal_energy,
    const Scalar<DataVector>& electron_fraction,
    const Scalar<DataVector>& lorentz_factor,
    const Scalar<DataVector>& specific_enthalpy,
    const tnsr::ii<DataVector, 3, Frame::Inertial>& spatial_metric,
    const tnsr::i<DataVector, 3>& unit_normal,
    const EquationsOfState::EquationOfState<true, ThermodynamicDim>&
        equation_of_state);
/// @}

/*!
 * \brief Labels for the characteristic speeds of the relativistic
 * magnetohydrodynamics system, in increasing order.
 *
 * \see `grmhd::ValenciaDivClean::characteristic_speeds_mhd`
 */
namespace MhdSpeed {
/// The smallest fast magnetosonic speed, $\lambda_\text{fast}^-$
constexpr size_t FastMagnetosonicMinus = 0;
/// The smallest Alfven speed, $\lambda_\text{Alfven}^-$
constexpr size_t AlfvenMinus = 1;
/// The smallest slow magnetosonic speed, $\lambda_\text{slow}^-$
constexpr size_t SlowMagnetosonicMinus = 2;
/// The entropy speed, $\lambda_\text{entropy} = v_n = v^i s_i$
constexpr size_t Entropy = 3;
/// The largest slow magnetosonic speed, $\lambda_\text{slow}^+$
constexpr size_t SlowMagnetosonicPlus = 4;
/// The largest Alfven speed, $\lambda_\text{Alfven}^+$
constexpr size_t AlfvenPlus = 5;
/// The largest fast magnetosonic speed, $\lambda_\text{fast}^+$
constexpr size_t FastMagnetosonicPlus = 6;
}  // namespace MhdSpeed

/*!
 * \brief Compute the characteristic speeds for the relativistic
 * magnetohydrodynamics system with divergence cleaning in the Eulerian frame.
 *
 * These are the eigenvalues of the flux Jacobian projected along a spatial
 * direction with unit normal $s_i$, evaluated with unit lapse and zero shift.
 * With lapse $\alpha$ and shift $\beta^i$, the physical characteristic speeds
 * are $\alpha \lambda - \beta_n$ with $\beta_n = \beta^i s_i$.
 *
 * Following \cite Teukolsky2025yle (Eq. (4.13)), the characteristic speeds
 * $\lambda$ are the zeros of the determinant of the characteristic matrix,
 * which is, up to a constant factor,
 * \begin{equation}
 *   \det(A^a q_a) = a \, G \, (B^2 - \rho h^* a^2) \, N,
 * \end{equation}
 * where $q_a := \lambda n_a + s_a$, with $n_a$ the unit normal to the 3+1
 * slice, and
 * \begin{align}
 *   a &:= q^a u_a = W (v_n - \lambda), \\
 *   B &:= q^a b_a = \frac{B_n}{W} - W (B^i v_i) (\lambda - v_n), \\
 *   G &:= q^a q_a = 1 - \lambda^2, \\
 *   N &:= a^2 G b^2 - \rho h a^4
 *        + c_s^2 \left[\rho h a^2 (a^2 + G) - B^2 G\right].
 * \end{align}
 * Here $v_n := v^i s_i$, $B_n := B^i s_i$, $W$ is the Lorentz factor,
 * $b^2 := B^i B_i / W^2 + (B^i v_i)^2$ is the comoving magnetic field
 * squared, and $\rho h^* := \rho h + b^2$. Each factor gives a class of
 * speeds:
 * - Entropy ($a = 0$):
 *   \begin{equation}
 *     \lambda_\text{entropy} = v_n.
 *   \end{equation}
 * - %Scalar, from the divergence cleaning ($G = 0$):
 *   \begin{equation}
 *     \lambda_\text{scalar}^\pm = \pm 1.
 *   \end{equation}
 *   These are the speed of light at every point, so they are not returned.
 * - Alfven ($B = \pm \sqrt{\rho h^*} a$): the closed form (Eq.
 *   (2.74) of \cite Teukolsky2025yle)
 *   \begin{equation}
 *     \lambda_\text{Alfven} = v_n
 *       + \frac{B_n}{W^2 \left(B^i v_i \pm \sqrt{\rho h^*}\right)}.
 *   \end{equation}
 *   Since $\sqrt{\rho h^*} > |B^i v_i|$, the sign of the second term is
 *   known, which orders the two speeds as
 *   \begin{equation}
 *     \lambda_\text{Alfven}^\pm = v_n
 *       \pm \frac{|B_n|}{W^2 \left(\sqrt{\rho h^*}
 *       \pm \operatorname{sgn}(B_n) B^i v_i\right)}.
 *   \end{equation}
 * - Magnetosonic ($N = 0$): the four roots of a quartic polynomial, two fast
 *   and two slow.
 *
 * For an admissible state, the order of the speeds is always
 * \begin{equation}
 *   \lambda_\text{scalar}^- \leq \lambda_\text{fast}^-
 *   \leq \lambda_\text{Alfven}^- \leq \lambda_\text{slow}^-
 *   \leq \lambda_\text{entropy} \leq \lambda_\text{slow}^+
 *   \leq \lambda_\text{Alfven}^+ \leq \lambda_\text{fast}^+
 *   \leq \lambda_\text{scalar}^+.
 * \end{equation}
 * The scalar speeds are outermost because every other speed lies inside the
 * light cone. Without them, this is Eq. (42) of \cite Anton2009xq.
 *
 * In order to resolve magnetosonic speeds close to the entropy speed $v_n$,
 * we define
 * \begin{equation}
 *   \mu := \lambda - v_n.
 * \end{equation}
 * For the polynomial to depend only on dimensionless quantities, the magnetic
 * terms are divided by $\sqrt{\rho h}$,
 * \begin{align}
 *   \tilde{B}(v_n + \mu) &:= \frac{B(v_n + \mu)}{\sqrt{\rho h}}, \\
 *   \tilde{B}_n &:= \frac{B_n}{\sqrt{\rho h}}, \\
 *   \tilde{B}_v &:= \frac{B^i v_i}{\sqrt{\rho h}}, \\
 *   \tilde{b}^2 &:= \frac{b^2}{\rho h}.
 * \end{align}
 * With this, the factors of the determinant become
 * \begin{align}
 *   a(v_n + \mu) &= -W \mu, \\
 *   \tilde{B}(v_n + \mu) &= \frac{\tilde{B}_n}{W} - W \tilde{B}_v \mu, \\
 *   G(v_n + \mu) &= (1 - v_n - \mu)(1 + v_n + \mu), \\
 *   \tilde{N}(\mu) &:= \frac{N(v_n + \mu)}{\rho h} = a^2 G \tilde{b}^2 - a^4
 *     + c_s^2 \left[a^2 (a^2 + G) - \tilde{B}^2 G\right],
 * \end{align}
 * with the factors evaluated at $v_n + \mu$.
 * Writing the polynomial $\tilde{N}(\mu)$ explicitly, we have
 * \begin{equation}
 *   \tilde{N}(\mu) = \tilde{N}_{(4)} \mu^4 + \tilde{N}_{(3)} \mu^3
 *     + \tilde{N}_{(2)} \mu^2 + \tilde{N}_{(1)} \mu + \tilde{N}_{(0)},
 * \end{equation}
 * where the coefficients are
 * \begin{align}
 *   \tilde{N}_{(4)} &:= -W^2 \left[(1 - c_s^2) W^2 + \tilde{b}^2
 *     + c_s^2 (1 - \tilde{B}_v^2)\right], \\
 *   \tilde{N}_{(3)} &:= -2 \left\{v_n W^2 \left[\tilde{b}^2
 *     + c_s^2 (1 - \tilde{B}_v^2)\right]
 *     + c_s^2 \tilde{B}_n \tilde{B}_v\right\}, \\
 *   \tilde{N}_{(2)} &:= \frac{1}{W^2} \left\{c_s^2 \tilde{B}_n
 *     (\tilde{B}_n - 4 \tilde{B}_v v_n W^2)
 *     + \left[\tilde{b}^2 + c_s^2 (1 - \tilde{B}_v^2)\right]
 *     (1 - v_n^2) W^4\right\}, \\
 *   \tilde{N}_{(1)} &:= \frac{2 c_s^2 \tilde{B}_n}{W^2}
 *     \left[\tilde{B}_v (1 - v_n^2) W^2 + v_n \tilde{B}_n\right], \\
 *   \tilde{N}_{(0)} &:= -\frac{c_s^2 \tilde{B}_n^2 (1 - v_n^2)}{W^2}.
 * \end{align}
 * Since $\tilde{N}_{(4)} < 0$, dividing by it gives the monic quartic
 * polynomial
 * \begin{align}
 *   Q(\mu) &:= \frac{\tilde{N}(\mu)}{\tilde{N}_{(4)}} \\
 *   &= \mu^4 + Q_{(3)} \mu^3 + Q_{(2)} \mu^2 + Q_{(1)} \mu + Q_{(0)},
 * \end{align}
 * where the coefficients are $Q_{(i)} := \tilde{N}_{(i)} / \tilde{N}_{(4)}$.
 *
 * The fast speeds are found by Newton-Raphson safeguarded by bisection
 * (based on `rtsafe` of \cite NumericalRecipes, Sec. 9.4), in the brackets
 * \begin{align}
 *   \mu_\text{Alfven}^+ &\leq \mu_\text{fast}^+ < 1 - v_n, \\
 *   -1 - v_n &< \mu_\text{fast}^- \leq \mu_\text{Alfven}^-,
 * \end{align}
 * where $\mu_\text{Alfven}^\pm$ is the second term of the closed form
 * above. Newton stops once the step is below $10^{-14}$. $Q(\mu)$ and
 * $Q'(\mu)$ are evaluated from the factors $a(v_n + \mu)$,
 * $\tilde{B}(v_n + \mu)$ and $G(v_n + \mu)$ rather than from the
 * coefficients, which keeps $G(v_n + \mu)$ accurate near the speed of
 * light.
 *
 * The slow speeds are the roots of the quadratic
 * $q(\mu) := \mu^2 + q_{(1)} \mu + q_{(0)}$ left after dividing $Q(\mu)$
 * by the two fast roots. Since the fast speeds are the roots of largest
 * $|\mu|$, the division is done from the constant term up (backward
 * deflation, which is stable for these roots; \cite NumericalRecipes,
 * Sec. 9.5.1):
 * \begin{align}
 *   q_{(0)} &:= \frac{Q_{(0)}}{\mu_\text{fast}^- \mu_\text{fast}^+}, \\
 *   q_{(1)} &:= \frac{Q_{(1)} + (\mu_\text{fast}^- + \mu_\text{fast}^+)
 *     q_{(0)}}{\mu_\text{fast}^- \mu_\text{fast}^+}.
 * \end{align}
 * The roots are then computed without cancellation
 * (\cite NumericalRecipes, Sec. 5.6). Since $Q_{(0)} \propto B_n^2$ and
 * $Q_{(1)} \propto B_n$, the slow speeds are exactly $v_n$ for $B_n = 0$.
 *
 * Since the terms of $Q(\mu)$ shrink with $\mu$ and $B_n$, the speeds keep
 * their accuracy when the magnetosonic speeds crowd around $v_n$ (e.g., a
 * cold fluid with a weak magnetic field, or a large Lorentz factor), while a
 * solve in $\lambda$ loses accuracy. The fast speeds are accurate to the
 * absolute tolerance $10^{-14}$ of the Newton-Raphson steps.
 *
 * The returned tensor has index 0..6, in increasing order (see `MhdSpeed`):
 * - `characteristic_speeds.get(0)`: fast magnetosonic speed
 *   $\lambda_\text{fast}^-$
 * - `characteristic_speeds.get(1)`: Alfven speed $\lambda_\text{Alfven}^-$
 * - `characteristic_speeds.get(2)`: slow magnetosonic speed
 *   $\lambda_\text{slow}^-$
 * - `characteristic_speeds.get(3)`: entropy speed $v_n$
 * - `characteristic_speeds.get(4)`: slow magnetosonic speed
 *   $\lambda_\text{slow}^+$
 * - `characteristic_speeds.get(5)`: Alfven speed $\lambda_\text{Alfven}^+$
 * - `characteristic_speeds.get(6)`: fast magnetosonic speed
 *   $\lambda_\text{fast}^+$
 *
 * \note The sound speed must satisfy $0 \leq c_s^2 < 1$, the condition for
 * the system to be hyperbolic, and this function stops with an error
 * otherwise. Every causal equation of state satisfies it, provided the state
 * passed in is consistent with the equation of state (e.g., the specific
 * enthalpy $h = 1 + \epsilon + p / \rho$ with $p$ from the equation of state
 * at the given $\rho$ and $\epsilon$).
 *
 * \note The normal must be unit with respect to the spatial metric.
 */
template <size_t ThermodynamicDim>
void characteristic_speeds_mhd(
    gsl::not_null<tnsr::i<DataVector, 7>*> characteristic_speeds,

    /* primitive variables */
    const tnsr::I<DataVector, 3, Frame::Inertial>& spatial_velocity,
    const tnsr::I<DataVector, 3, Frame::Inertial>& magnetic_field,
    const Scalar<DataVector>& rest_mass_density,
    const Scalar<DataVector>& specific_internal_energy,
    const Scalar<DataVector>& electron_fraction,

    /* other helpful quantities */
    const Scalar<DataVector>& lorentz_factor,
    const Scalar<DataVector>& specific_enthalpy,
    const tnsr::ii<DataVector, 3, Frame::Inertial>& spatial_metric,
    const tnsr::i<DataVector, 3>& unit_normal,
    const EquationsOfState::EquationOfState<true, ThermodynamicDim>&
        equation_of_state);

/**
 * \brief Compute the characteristic matrix (flux Jacobian) for relativistic
 * hydrodynamics + composition ($Y_e$), in a given direction.
 *
 * The characteristic matrix is the Jacobian of the flux along the unit normal
 * $n_i$ with respect to the conserved variables $U_n$,
 * \begin{equation}
 *   A_{m}{}^{n} = \frac{\partial (n_i F^i_m)}{\partial U_n},
 * \end{equation}
 * evaluated with unit lapse and zero shift. With lapse $\alpha$ and shift
 * $\beta^i$, the physical characteristic speeds are $\alpha \lambda - \beta_n$,
 * where $\lambda$ are the eigenvalues of this matrix and
 * $\beta_n = n_i \beta^i$.
 *
 * The index $m$ labels the rows of the matrix (the flux components), and the
 * index $n$ labels its columns (the conserved variables). Both follow the order
 * $(D, S_i, \tau, D Y_e)$, which differs from the order of the system's
 * `variables_tag`.
 *
 * \note The normal must be unit with respect to the spatial metric.
 *
 * \note With a 2D equation of state, the pressure must be positive, since
 * $\kappa = \partial p / \partial \epsilon |_\rho$ is recovered from
 * $\kappa p / \rho^2$, the only form the 2D interface provides.
 *
 * \note The specific enthalpy must be consistent with the equation of state,
 * $h = 1 + \epsilon + p / \rho$. The matrix uses both $h$ and
 * $h - 1 = \epsilon + p / \rho$, which is computed from the equation of state
 * to avoid cancellation when $h \approx 1$.
 *
 * \see `grmhd::ValenciaDivClean::numerical_characteristics` for the
 * conventions relating this matrix to characteristic speeds, modes, and
 * projectors.
 *
 * \see `find_real_eigensystem` for the index conventions of the matrix.
 */
template <size_t ThermodynamicDim>
void flux_jacobian_hydro(
    gsl::not_null<tnsr::iJ<DataVector, 6>*> characteristic_matrix,

    /* primitive variables */
    const tnsr::I<DataVector, 3, Frame::Inertial>& spatial_velocity,
    const Scalar<DataVector>& rest_mass_density,
    const Scalar<DataVector>& specific_internal_energy,
    const Scalar<DataVector>& electron_fraction,

    /* other helpful quantities */
    const Scalar<DataVector>& lorentz_factor,
    const Scalar<DataVector>& specific_enthalpy,
    const tnsr::ii<DataVector, 3, Frame::Inertial>& spatial_metric,
    const tnsr::II<DataVector, 3, Frame::Inertial>& inv_spatial_metric,
    const tnsr::i<DataVector, 3>& unit_normal,
    const EquationsOfState::EquationOfState<true, ThermodynamicDim>&
        equation_of_state);

/**
 * \brief Compute numerical characteristic speeds (eigenvalues), characteristic
 * projectors (left eigenvectors), and characteristic modes (right eigenvectors)
 * in a given direction, from the primitive variables.
 *
 * \note Currently, this function only supports building the eigensystem for
 * relativistic hydrodynamics (no magnetic field) with composition dependence
 * ($Y_e$).
 *
 * The characteristic speeds $\lambda_{(i)}$ are the eigenvalues of the
 * characteristic matrix in a given direction, $A_{m}{}^{n}$. The characteristic
 * projectors $L^{im}$ and modes $R_{im}$ are its left and right eigenvectors,
 * with the index conventions of `find_real_eigensystem`. We provide the
 * characteristics in the following types:
 * - Eigenvalues $\lambda_{(i)}$: `tnsr::i<DataVector, 6> characteristic_speeds`
 * - Right eigenvectors $R_{in}$: `tnsr::ij<DataVector, 6> characteristic_modes`
 * - Left eigenvectors $L^{im}$: `tnsr::IJ<DataVector, 6>
 *   characteristic_projectors`
 *
 * We refer to the left eigenvectors as "projectors" because they are used to
 * project a state into the characteristic basis. Similarly, we refer to the
 * right eigenvectors as "modes" because they are used to reconstruct the state.
 * For example, consider a state vector $U_{m}$ that can be represented in the
 * characteristic basis as
 * \begin{equation}
 *   U_{m} = \sum_i w^{i} R_{im}.
 * \end{equation}
 * Then, the characteristic fields $w^{i}$ can be obtained by projecting the
 * state vector $U_{m}$ into the characteristic basis using the left
 * eigenvectors:
 * \begin{equation}
 *   w^{i} = \sum_m L^{im} U_{m}.
 * \end{equation}
 *
 * The eigenvalues are in increasing order. The eigenvectors are normalized so
 * that $\sum_m L^{im} R_{jm} = \delta^{i}_{j}$. The right eigenvectors are not
 * unit vectors.
 *
 * \see `find_real_eigensystem` for the index conventions and for how this
 * eigensystem is computed at each grid point.
 */
template <size_t ThermodynamicDim>
void numerical_characteristics(
    gsl::not_null<tnsr::i<DataVector, 6>*> characteristic_speeds,
    gsl::not_null<tnsr::ij<DataVector, 6>*> characteristic_modes,
    gsl::not_null<tnsr::IJ<DataVector, 6>*> characteristic_projectors,

    /* primitive variables */
    const tnsr::I<DataVector, 3, Frame::Inertial>& spatial_velocity,
    const Scalar<DataVector>& rest_mass_density,
    const Scalar<DataVector>& specific_internal_energy,
    const Scalar<DataVector>& electron_fraction,

    /* other helpful quantities */
    const Scalar<DataVector>& lorentz_factor,
    const Scalar<DataVector>& specific_enthalpy,
    const tnsr::ii<DataVector, 3, Frame::Inertial>& spatial_metric,
    const tnsr::II<DataVector, 3, Frame::Inertial>& inv_spatial_metric,
    const tnsr::i<DataVector, 3>& unit_normal,
    const EquationsOfState::EquationOfState<true, ThermodynamicDim>&
        equation_of_state);

namespace Tags {
/// \brief Compute the characteristic speeds for the Valencia formulation of
/// GRMHD with divergence cleaning.
///
/// \details see grmhd::ValenciaDivClean::characteristic_speeds
struct CharacteristicSpeedsCompute : Tags::CharacteristicSpeeds,
                                     db::ComputeTag {
  using base = Tags::CharacteristicSpeeds;
  using argument_tags =
      tmpl::list<hydro::Tags::RestMassDensity<DataVector>,
                 hydro::Tags::ElectronFraction<DataVector>,
                 hydro::Tags::SpecificInternalEnergy<DataVector>,
                 hydro::Tags::SpecificEnthalpy<DataVector>,
                 hydro::Tags::SpatialVelocity<DataVector, 3>,
                 hydro::Tags::LorentzFactor<DataVector>,
                 hydro::Tags::MagneticField<DataVector, 3>,
                 gr::Tags::Lapse<DataVector>, gr::Tags::Shift<DataVector, 3>,
                 gr::Tags::SpatialMetric<DataVector, 3>,
                 ::Tags::Normalized<domain::Tags::UnnormalizedFaceNormal<3>>,
                 hydro::Tags::GrmhdEquationOfState>;

  using volume_tags = tmpl::list<hydro::Tags::GrmhdEquationOfState>;

  using return_type = std::array<DataVector, 9>;

  template <size_t ThermodynamicDim>
  void function(gsl::not_null<return_type*> result,
                const Scalar<DataVector>& rest_mass_density,
                const Scalar<DataVector>& /* electron_fraction */,
                const Scalar<DataVector>& specific_internal_energy,
                const Scalar<DataVector>& specific_enthalpy,
                const tnsr::I<DataVector, 3, Frame::Inertial>& spatial_velocity,
                const Scalar<DataVector>& lorentz_factor,
                const tnsr::I<DataVector, 3, Frame::Inertial>& magnetic_field,
                const Scalar<DataVector>& lapse,
                const tnsr::I<DataVector, 3>& shift,
                const tnsr::ii<DataVector, 3, Frame::Inertial>& spatial_metric,
                const tnsr::i<DataVector, 3>& unit_normal,
                const EquationsOfState::EquationOfState<true, ThermodynamicDim>&
                    equation_of_state);
};

struct LargestCharacteristicSpeed : db::SimpleTag {
  using type = double;
};

struct ComputeLargestCharacteristicSpeed : db::ComputeTag,
                                           LargestCharacteristicSpeed {
  using argument_tags =
      tmpl::list<gr::Tags::Lapse<DataVector>, gr::Tags::Shift<DataVector, 3>,
                 gr::Tags::SpatialMetric<DataVector, 3>>;
  using return_type = double;
  using base = LargestCharacteristicSpeed;
  static void function(gsl::not_null<double*> speed,
                       const Scalar<DataVector>& lapse,
                       const tnsr::I<DataVector, 3>& shift,
                       const tnsr::ii<DataVector, 3>& spatial_metric) {
    const auto shift_magnitude = magnitude(shift, spatial_metric);
    *speed = max(get(shift_magnitude) + get(lapse));
  }
};
}  // namespace Tags
}  // namespace ValenciaDivClean
}  // namespace grmhd
