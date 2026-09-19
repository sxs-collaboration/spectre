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

namespace grmhd::ValenciaDivClean {
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
enum HydroSpeed : uint32_t {
  NormalDotVelocity = 0,
  LambdaPlus = 1,
  LambdaMinus = 2
};
enum HydroVectorR : uint32_t {
  R1 = 0,
  R2 = 1,
  R3 = 2,
  R4 = 3,
  Rplus = 4,
  Rminus = 5,
};

enum HydroVectorL : uint32_t {
  L1 = 0,
  L2 = 1,
  L3 = 2,
  L4 = 3,
  Lplus = 4,
  Lminus = 5,
};

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
 * The returned array has size 3 and contains the unique characteristic speeds:
 *  - `char_speeds[0]`: \f$v_n\f$ (degenerate eigenvalue, multiplicity 4 if the
 * electron fraction is included),
 *  - `char_speeds[1]`: \f$\lambda_+\f$ (right-propagating acoustic mode),
 *  - `char_speeds[2]`: \f$\lambda_-\f$ (left-propagating acoustic mode).
 */

template <size_t ThermodynamicDim>
void characteristic_speeds_hydro(
    gsl::not_null<std::array<DataVector, 3>*> pchar_speeds,
    const tnsr::I<DataVector, 3, Frame::Inertial>& spatial_velocity,
    const Scalar<DataVector>& rest_mass_density,
    const Scalar<DataVector>& specific_internal_energy,
    const Scalar<DataVector>& specific_enthalpy,
    const Scalar<DataVector>& electron_fraction,
    const Scalar<DataVector>& lorentz_factor,
    const tnsr::i<DataVector, 3>& unit_normal,
    const tnsr::ii<DataVector, 3, Frame::Inertial>& spatial_metric,
    const EquationsOfState::EquationOfState<true, ThermodynamicDim>&
        equation_of_state);

template <size_t ThermodynamicDim>
std::array<DataVector, 3> characteristic_speeds_hydro(
    const tnsr::I<DataVector, 3, Frame::Inertial>& spatial_velocity,
    const Scalar<DataVector>& rest_mass_density,
    const Scalar<DataVector>& specific_internal_energy,
    const Scalar<DataVector>& specific_enthalpy,
    const Scalar<DataVector>& electron_fraction,
    const Scalar<DataVector>& lorentz_factor,
    const tnsr::i<DataVector, 3>& unit_normal,
    const tnsr::ii<DataVector, 3, Frame::Inertial>& spatial_metric,
    const EquationsOfState::EquationOfState<true, ThermodynamicDim>&
        equation_of_state);
/// @}

/// @{
/*!
 * \brief Compute the left and right characteristic eigenvectors for the
 * relativistic hydrodynamics system in the Eulerian frame.
 *
 * These are the eigenvectors of the flux Jacobian projected along a spatial
 * direction with unit normal \f$ s_i \f$, measured by an Eulerian observer.
 * The eigenvectors correspond to the characteristic speeds returned by
 * `characteristic_speeds_hydro`.
 *
 * The ordering of the conserved variables for the eigenvectors is
 *
 * \f[
 *   (D,\; S_i,\; \tau,\; D Y_e).
 * \f]
 *
 * The right eigenvectors are:
 *
 * \f{align*}
 * \mathbf{R}_{1,2} &=
 * \begin{pmatrix}
 *   W v_{1,2} \\
 *   h\left(t^{(1,2)}_i + 2 W^2 v_{1,2} v_i\right) \\
 *   W(2hW - 1) v_{1,2} \\
 *   W v_{1,2} Y_e
 * \end{pmatrix}, \\
 *
 * \mathbf{R}_3 &=
 * \begin{pmatrix}
 *   \kappa \\
 *   hW(\kappa - \rho c_s^2) v_i \\
 *   hW(\kappa - \rho c_s^2) - \kappa \\
 *   \kappa Y_e
 * \end{pmatrix}, \\
 *
 * \mathbf{R}_\pm &=
 * \begin{pmatrix}
 *   1 \\
 *   hW\left(v_i \pm \dfrac{c_s}{d} s_i\right) \\
 *   hW\left(1 \pm \dfrac{c_s v_n}{d}\right) - 1 \\
 *   Y_e
 * \end{pmatrix},
 * \f}
 *
 * where
 *
 * \f[
 *   d \equiv W \sqrt{1 - v^2 c_s^2 - v_n^2 (1 - c_s^2)} .
 * \f]
 *
 * The additional degenerate right eigenvector \f$ \mathbf{R}_4 \f$ corresponds
 * to the electron fraction and reduces to
 *
 * \f[
 *   \mathbf{R}_4 =
 *   \begin{pmatrix}
 *     0 \\ 0 \\ 1 \\ -\kappa / (\zeta W)
 *   \end{pmatrix}
 * \f]
 *
 * when \f$ \zeta \neq 0 \f$. In the limit \f$ \zeta \to 0 \f$, a regularized
 * eigenvector is used with only a nonzero \f$ D Y_e \f$ component.
 *
 * The left eigenvectors are:
 *
 * \f{align*}
 * \mathbf{L}_{1,2} &=
 * \frac{1}{h(1 - v_n^2)}
 * \begin{pmatrix}
 *   -v_{1,2} \\
 *   v_{1,2} v_n s^i + (1 - v_n^2)\,t_{(1,2)}^i \\
 *   -v_{1,2} \\
 *   0
 * \end{pmatrix}, \\
 *
 * \mathbf{L}_3 &=
 * \frac{1}{\rho h c_s^2}
 * \begin{pmatrix}
 *   h - W + \dfrac{\zeta Y_e}{\kappa} \\
 *   W v^i \\
 *   -W \\
 *   -\dfrac{\zeta}{\kappa}
 * \end{pmatrix}, \\
 *
 * \mathbf{L}_\pm &=
 * \frac{1}{2 \rho h W c_s^2 (1 - v_n^2)}
 * \begin{pmatrix}
 *   b_\pm - hW\,k_\mathrm{term}(1 - v_n^2) \\
 *   -a v^i + \rho c_s (c_s v_n \pm d)\,s^i \\
 *   b_\pm \\
 *   \zeta W(1 - v_n^2)
 * \end{pmatrix},
 * \f}
 *
 * where
 *
 * \f{align*}
 * a &\equiv W^2(1 - v_n^2)(\kappa + \rho c_s^2),\\
 * c_\pm &\equiv \rho c_s(c_s \pm v_n d),\\
 * b_\pm &\equiv a - c_\pm,\\
 * k_\mathrm{term} &\equiv \kappa - \rho c_s^2 + \frac{\zeta Y_e}{h}.
 * \f}
 *
 * The additional degenerate left eigenvector is
 *
 * \f[
 *   \mathbf{L}_4 =
 *   \frac{\zeta W}{\kappa}
 *   \begin{pmatrix}
 *     -Y_e \\ 0 \\ 0 \\ 1
 *   \end{pmatrix},
 * \f]
 *
 * with a regularized form used in the limit \f$ \zeta \to 0 \f$.
 *
 * \note The transverse vectors \f$ t^{(1)}_i \f$ and \f$ t^{(2)}_i \f$ are
 * constructed to be orthonormal to each other and to the unit normal
 * \f$ s_i \f$.
 */
template <size_t ThermodynamicDim>
void eigenvectors_hydro(
    const gsl::not_null<std::array<tnsr::i<DataVector, 6, Frame::Inertial>,
                                   6>*>& right_eigenvectors,
    const gsl::not_null<std::array<tnsr::I<DataVector, 6, Frame::Inertial>,
                                   6>*>& left_eigenvectors,
    const tnsr::I<DataVector, 3, Frame::Inertial>& spatial_velocity,
    const Scalar<DataVector>& rest_mass_density,
    const Scalar<DataVector>& specific_internal_energy,
    const Scalar<DataVector>& specific_enthalpy,
    const Scalar<DataVector>& electron_fraction,
    const Scalar<DataVector>& lorentz_factor, const Scalar<DataVector>& kappa,
    const Scalar<DataVector>& zeta, const tnsr::i<DataVector, 3>& unit_normal,
    const tnsr::ii<DataVector, 3, Frame::Inertial>& spatial_metric,
    const EquationsOfState::EquationOfState<true, ThermodynamicDim>&
        equation_of_state);
/// @}

namespace detail {

/**
 * \brief Compute the flux Jacobian matrix (aka, characteristic matrix) for
 * relativistic hydrodynamics with composition dependence (electron fraction).
 *
 * We label the characteristic matrix in a given direction, $A_c^{\ b}$, such
 * that the index $b$ labels columns and the index $c$ labels rows.
 * With this choice of indices, the right eigenvectors $R_b$ satisfy
 *
 * \begin{equation}
 *   A_c^{\ b} R_b = \lambda R_c,
 * \end{equation}
 *
 * while the left eigenvectors $L^c$ satisfy $L^c A_c^{\ b} = \lambda L^b$.
 *
 * \begin{equation}
 *   L^c A_c^{\ b} = \lambda L^b,
 * \end{equation}
 *
 * where $\lambda$ is the corresponding eigenvalue (characteristic speed).
 *
 * \note The indices $b$ and $c$ are not tensorial. They simply label the rows
 * and columns of the characteristic matrix. We make the conventions presented
 * above to clearly distinguish which indices (first or second) get contracted
 * with which eigenvectors (left or right).
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

}  // namespace detail

/**
 * \brief Compute a numerical eigensystem (eigenvalues and left/right
 * eigenvectors) for a given characteristic matrix.
 *
 * \note Currently, this function only supports building the eigensystem for
 * relativistic hydrodynamics (no magnetic field) with composition dependence
 * (electron fraction).
 *
 * \see `grmhd::ValenciaDivClean::detail::flux_jacobian_hydro` for details on
 * how we choose our indices for the left/right eigenvectors.
 */
template <size_t ThermodynamicDim>
void numerical_eigensystem(
    gsl::not_null<std::array<Scalar<DataVector>, 6>*> all_eigenvalues,
    gsl::not_null<std::array<tnsr::i<DataVector, 6>, 6>*>
        all_right_eigenvectors,
    gsl::not_null<std::array<tnsr::I<DataVector, 6>, 6>*> all_left_eigenvectors,
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
}  // namespace grmhd::ValenciaDivClean
