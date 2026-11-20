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
 *
 * \see `grmhd::ValenciaDivClean::analytic_characteristics` for the same
 * characteristics from closed-form expressions.
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

/*!
 * \brief Compute analytic characteristic speeds (eigenvalues), characteristic
 * projectors (left eigenvectors), and characteristic modes (right eigenvectors)
 * in a given direction, from the primitive variables.
 *
 * \note Currently, this function only supports relativistic hydrodynamics (no
 * magnetic field) with composition dependence ($Y_e$).
 *
 * It returns the characteristic speeds, modes, and projectors with the same
 * types and ordering as `grmhd::ValenciaDivClean::numerical_characteristics`,
 * from the closed-form expressions of \cite Teukolsky2025erz. The eigenvectors
 * are normalized so that $\sum_m L^{im} R_{jm} = \delta^{i}_{j}$.
 *
 * The speeds are $\lambda_-$, $v_n$ (four times), and $\lambda_+$, with $v_n$,
 * $\lambda_\pm$, $W$, $h$, $c_s$, and $d$ as in `characteristic_speeds_hydro`.
 * In the order $(D, S_j, \tau, D Y_e)$ of `flux_jacobian_hydro`, the modes and
 * projectors are $R_{(0)} = R_-$, $R_{(5)} = R_+$, $L^{(0)} = L^-$,
 * $L^{(5)} = L^+$, and
 *
 * \begin{align}
 *   R_\pm &= \left(1,\ h W \left(v_j \pm \frac{c_s}{d} s_j\right),\
 *     h W \left(1 \pm \frac{c_s v_n}{d}\right) - 1,\ Y_e\right), \\
 *   R_{(1,2)} &= \left(W v_{(1,2)},\
 *     h \left(t^{(1,2)}_j + 2 W^2 v_{(1,2)} v_j\right),\
 *     W (2 h W - 1) v_{(1,2)},\ W v_{(1,2)} Y_e\right), \\
 *   R_{(3)} &= \left(\kappa,\ h W (\kappa - \rho c_s^2) v_j,\
 *     h W (\kappa - \rho c_s^2) - \kappa,\ \kappa Y_e\right), \\
 *   R_{(4)} &= (0,\ -\zeta W v_j / \kappa,\ -\zeta W / \kappa,\ 1), \\
 *   L^\pm &= \frac{1}{2 \rho h W c_s^2 (1 - v_n^2)} \left(b_\pm - h W
 *     \left(\kappa - \rho c_s^2 + \frac{\zeta Y_e}{h}\right) (1 - v_n^2),\
 *     -a v^j + \rho c_s (c_s v_n \pm d) s^j,\ b_\pm,\
 *     \zeta W (1 - v_n^2)\right), \\
 *   L^{(1,2)} &= \frac{1}{h (1 - v_n^2)} \left(-v_{(1,2)},\
 *     v_{(1,2)} v_n s^j + (1 - v_n^2) t^{(1,2)j},\ -v_{(1,2)},\ 0\right), \\
 *   L^{(3)} &= \frac{1}{\rho h c_s^2} \left(h - W + \frac{\zeta Y_e}{\kappa},\
 *     W v^j,\ -W,\ -\frac{\zeta}{\kappa}\right), \\
 *   L^{(4)} &= (-Y_e,\ 0,\ 0,\ 1),
 * \end{align}
 *
 * where $s_j$ is the unit normal, $t^{(1)}_j$ and $t^{(2)}_j$ are unit
 * one-forms orthogonal to each other and to $s_j$ (see `orthonormal_oneform`),
 * $v_{(1,2)} = t^{(1,2)}_j v^j$, $a = W^2 (1 - v_n^2) (\kappa + \rho c_s^2)$,
 * $b_\pm = a - \rho c_s (c_s \pm v_n d)$,
 * $\kappa = \partial p / \partial \epsilon |_{\rho, Y_e}$, and
 * $\zeta = \partial p / \partial Y_e |_{\rho, \epsilon}$.
 *
 * $R_{(4)}$ and $L^{(4)}$ are those of \cite Teukolsky2025erz times
 * $-\zeta W / \kappa$ and its inverse, which keeps them finite as
 * $\zeta \to 0$. Where $\zeta = 0$, as for any equation of state with fewer
 * than three thermodynamic variables, we take $\zeta / \kappa = 0$.
 *
 * \note The arguments must satisfy the conditions of `flux_jacobian_hydro`,
 * and $\kappa$ must be nonzero where $\zeta \neq 0$, or the function stops with
 * an error. Only `EquationsOfState::Tabulated3D` has $\zeta \neq 0$.
 * `ConvertComposeTable` floors negative $\kappa$ in its tables at a small
 * positive value, but writes $\kappa = 0$ if the CompOSE table lacks it.
 */
template <size_t ThermodynamicDim>
void analytic_characteristics(
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
