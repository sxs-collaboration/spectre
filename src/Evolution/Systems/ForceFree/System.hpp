// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>

#include "DataStructures/Tensor/TypeAliases.hpp"
#include "DataStructures/VariablesTag.hpp"
#include "Evolution/Systems/ForceFree/BoundaryConditions/BoundaryCondition.hpp"
#include "Evolution/Systems/ForceFree/BoundaryCorrections/BoundaryCorrection.hpp"
#include "Evolution/Systems/ForceFree/Characteristics.hpp"
#include "Evolution/Systems/ForceFree/Tags.hpp"
#include "Evolution/Systems/ForceFree/TimeDerivativeTerms.hpp"
#include "PointwiseFunctions/GeneralRelativity/Tags.hpp"

///\cond
class DataVector;
///\endcond

/*!
 * \ingroup EvolutionSystemsGroup
 * \brief Items related to evolving the GRFFE system with divergence cleaning
 *
 */
namespace ForceFree {

/*!
 * \brief General relativistic force-free electrodynamics (GRFFE) system with
 * divergence cleaning.
 *
 * For electromagnetism in a curved spacetime, Maxwell equations are given as
 * \f{align*}{
 *  \nabla_a F^{ab} &= -J^b,\\
 *  \nabla_a ^* F^{ab} &= 0.
 * \f}
 *
 * where \f$F^{ab}\f$ is the electromagnetic field tensor, \f$^*F^{ab}\f$ is its
 * dual \f$^*F^{ab} = \epsilon^{abcd}F_{cd} / 2\f$, and \f$J^a\f$ is the
 * 4-current.
 *
 * \note
 * - We are using the electromagnetic variables with the scaling convention
 * that the factor \f$4\pi\f$ does not appear in Maxwell equations and the
 * stress-energy tensor of the EM fields (geometrized Heaviside-Lorentz units).
 * - We adopt following definition of the Levi-Civita tensor by
 * \cite Misner1973
 * \f{align*}{
 *  \epsilon_{abcd} &= \sqrt{-g} \, [abcd] ,\\
 *  \epsilon^{abcd} &= -\frac{1}{\sqrt{-g}} \, [abcd] ,
 * \f}
 * where \f$g\f$ is the determinant of spacetime metric, and \f$[abcd]=\pm 1\f$
 * is the antisymmetric symbol with \f$[0123]=+1\f$.
 *
 * In SpECTRE, we evolve 'extended' (or augmented) version of Maxwell equations
 * with two divergence cleaning scalar fields \f$\psi\f$ and \f$\phi\f$ :
 *
 * \f{align*}{
 *  \nabla_a(F^{ab}+g^{ab}\psi) & = -J^b + \kappa_\psi n^b \psi \\
 *  \nabla_a(^* F^{ab}+ g^{ab}\phi) & = \kappa_\phi n^b \phi
 * \f}
 *
 * which reduce to the original Maxwell equations when \f$\psi=\phi=0\f$. For
 * damping constants $\kappa_{\psi, \phi} > 0$, Gauss constraint violations are
 * damped with timescales $\kappa_{\psi,\phi}^{-1}$ and propagated away.
 *
 * We decompose the EM field tensor as follows
 * \f{align*}{
 *  F^{ab} = n^a E^b - n^b E^a - \epsilon^{abcd}B_c n_d,
 * \f}
 *
 * where $n^a$ is the normal to spatial hypersurface, $E^a$ and $B^a$ are
 * electric and magnetic fields.
 *
 * Evolved variables are
 *
 * \f{align*}{
 * \mathbf{U} = \sqrt{\gamma}\left[\,\begin{matrix}
 *      E^i \\
 *      B^i \\
 *      \psi \\
 *      \phi \\
 *      q \\
 * \end{matrix}\,\right] \equiv \left[\,\,\begin{matrix}
 *      \tilde{E}^i \\
 *      \tilde{B}^i \\
 *      \tilde{\psi} \\
 *      \tilde{\phi} \\
 *      \tilde{q} \\
 * \end{matrix}\,\,\right] \f}
 *
 * where \f$E^i\f$ is electric field, \f$B^i\f$ is magnetic field, $\psi$ is
 * electric divergence cleaning field, $\phi$ is magnetic divergence cleaning
 * field, \f$q\equiv-n_aJ^a\f$ is electric charge density, and \f$\gamma\f$ is
 * the determinant of spatial metric.
 *
 * Corresponding fluxes \f$\mathbf{F}^j\f$ are
 *
 * \f{align*}{
 *  F^j(\tilde{E}^i) & = -\beta^j\tilde{E}^i + \alpha
 *      (\gamma^{ij}\tilde{\psi} - \epsilon^{ijk}_{(3)}\tilde{B}_k) \\
 *  F^j(\tilde{B}^i) & = -\beta^j\tilde{B}^i + \alpha (\gamma^{ij}\tilde{\phi} +
 *      \epsilon^{ijk}_{(3)}\tilde{E}_k) \\
 *  F^j(\tilde{\psi}) & = -\beta^j \tilde{\psi} + \alpha \tilde{E}^j \\
 *  F^j(\tilde{\phi}) & = -\beta^j \tilde{\phi} + \alpha \tilde{B}^j \\
 *  F^j(\tilde{q}) & = \tilde{J}^j - \beta^j \tilde{q}
 * \f}
 *
 * and source terms are
 *
 * \f{align*}{
 *  S(\tilde{E}^i) &= -\tilde{J}^i - \tilde{E}^j \partial_j \beta^i
 *    + \tilde{\psi} ( \gamma^{ij} \partial_j \alpha - \alpha \gamma^{jk}
 *      \Gamma^i_{jk} ) \\
 *  S(\tilde{B}^i) &= -\tilde{B}^j \partial_j \beta^i + \tilde{\phi} (
 *      \gamma^{ij} \partial_j \alpha - \alpha \gamma^{jk} \Gamma^i_{jk} ) \\
 *  S(\tilde{\psi}) &= \tilde{E}^k \partial_k \alpha + \alpha \tilde{q} -
 *      \alpha \tilde{\phi} ( K + \kappa_\phi ) \\
 *  S(\tilde{\phi}) &= \tilde{B}^k \partial_k \alpha - \alpha \tilde{\phi}
 *      (K + \kappa_\phi ) \\
 *  S(\tilde{q}) &= 0
 * \f}
 *
 * where $\tilde{J}^i \equiv \alpha \sqrt{\gamma}J^i$.
 *
 * See the documentation of Fluxes and Sources for further details.
 *
 * In addition to Maxwell equations, general relativistic force-free
 * electrodynamics (GRFFE) assumes the following which are called the force-free
 * (FF) conditions.
 *
 * \f{align*}{
 *  F^{ab}J_b & = 0, \\
 *  ^*F^{ab}F_{ab} & = 0, \\
 *  F^{ab}F_{ab} & > 0.
 * \f}
 *
 * In terms of electric and magnetic fields, the FF conditions above read
 *
 * \f{align*}{
 *  E_iJ^i & = 0 , \\
 *  qE^i + \epsilon_{(3)}^{ijk} J_jB_k & = 0 , \\
 *  B_iE^i & = 0 , \\
 *  B^2 - E^2 & > 0.
 * \f}
 *
 * where \f$B^2=B^aB_a\f$ and \f$E^2 = E^aE_a\f$. Also,
 * \f$\epsilon_{(3)}^{ijk}\f$ is the spatial Levi-Civita tensor defined as
 *
 * \f{align*}
 *  \epsilon_{(3)}^{ijk} \equiv n_\mu \epsilon^{\mu ijk}
 *   = -\frac{1}{\sqrt{-g}} n_\mu [\mu ijk] = \frac{1}{\sqrt{\gamma}} [ijk]
 * \f}
 *
 * where \f$n^\mu\f$ is the normal to spatial hypersurface and \f$[ijk]\f$ is
 * the antisymmetric symbol with \f$[123] = +1\f$.
 *
 * There are a number of different ways in literature to numerically treat the
 * FF conditions. For the constraint $B_iE^i = 0$, cleaning of the parallel
 * electric field after every time step (e.g. \cite Palenzuela2010) or adopting
 * analytically determined parallel current density \cite Komissarov2011
 * were explored. On the magnetic dominance condition $B^2 - E^2 > 0$, there
 * have been approaches with modification of the drift current
 * \cite Komissarov2006 or manual rescaling of the electric field
 * \cite Palenzuela2010.
 *
 * We take the strategy that introduces special driver terms in the electric
 * current density \f$J^i\f$ following \cite Alic2012 :
 *
 * \f{align}{
 *  J^i = J^i_\mathrm{drift} + J^i_\mathrm{parallel}
 * \f}
 *
 * with
 *
 * \f{align}{
 *  J^i_\mathrm{drift} & = q \frac{\epsilon^{ijk}_{(3)}E_jB_k}{B_lB^l}, \\
 *  J^i_\mathrm{parallel} & = \eta \left[ \frac{E_jB^j}{B_lB^l}B^i
 *          + \frac{\mathcal{R}(E_lE^l-B_lB^l)}{B_lB^l}E^i \right] .
 * \f}
 *
 * where \f$\eta\f$ is the parallel conductivity and \f$\eta\rightarrow\infty\f$
 * corresponds to the ideal force-free limit. \f$\mathcal{R}(x)\f$ is the ramp
 * (or rectifier) function defined as
 *
 * \f{align*}
 *  \mathcal{R}(x) = \left\{\begin{array}{lc}
 *          x, & \text{if } x \geq 0 \\
 *          0, & \text{if } x < 0 \\
 * \end{array}\right\} = \max (x, 0) .
 * \f}
 *
 * Internally we handle each pieces \f$\tilde{J}^i_\mathrm{drift} \equiv
 * \alpha\sqrt{\gamma}J^i_\mathrm{drift}\f$ and \f$\tilde{J}^i_\mathrm{parallel}
 * \equiv \alpha\sqrt{\gamma}J^i_\mathrm{parallel}\f$ as two separate Tags
 * since the latter term is stiff and needs to be evolved in conjunction with
 * implicit time steppers.
 *
 */
struct System {
  static constexpr bool is_in_flux_conservative_form = true;
  static constexpr bool has_primitive_and_conservative_vars = false;
  static constexpr size_t volume_dim = 3;

  using boundary_conditions_base = BoundaryConditions::BoundaryCondition;
  using boundary_correction_base = BoundaryCorrections::BoundaryCorrection;

  using variables_tag =
      ::Tags::Variables<tmpl::list<Tags::TildeE, Tags::TildeB, Tags::TildePsi,
                                   Tags::TildePhi, Tags::TildeQ>>;

  using flux_variables = tmpl::list<Tags::TildeE, Tags::TildeB, Tags::TildePsi,
                                    Tags::TildePhi, Tags::TildeQ>;

  using non_conservative_variables = tmpl::list<>;
  using gradient_variables = tmpl::list<>;

  using spacetime_variables_tag =
      ::Tags::Variables<gr::tags_for_hydro<volume_dim, DataVector>>;

  using flux_spacetime_variables_tag = ::Tags::Variables<
      tmpl::list<gr::Tags::Lapse<DataVector>, gr::Tags::Shift<DataVector, 3>,
                 gr::Tags::SqrtDetSpatialMetric<DataVector>,
                 gr::Tags::SpatialMetric<DataVector, 3>,
                 gr::Tags::InverseSpatialMetric<DataVector, 3>>>;

  using compute_volume_time_derivative_terms = TimeDerivativeTerms;

  using compute_largest_characteristic_speed =
      Tags::LargestCharacteristicSpeedCompute;

  using inverse_spatial_metric_tag =
      gr::Tags::InverseSpatialMetric<DataVector, volume_dim>;
};
}  // namespace ForceFree
