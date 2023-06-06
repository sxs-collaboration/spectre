// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <memory>
#include <optional>
#include <pup.h>
#include <string>

#include "DataStructures/DataBox/Prefixes.hpp"
#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Evolution/BoundaryConditions/Type.hpp"
#include "Evolution/Systems/CurvedScalarWave/BoundaryConditions/BoundaryCondition.hpp"
#include "Evolution/Systems/CurvedScalarWave/Tags.hpp"
#include "NumericalAlgorithms/LinearOperators/PartialDerivatives.hpp"
#include "Options/String.hpp"
#include "PointwiseFunctions/GeneralRelativity/Tags.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/Serialization/CharmPupable.hpp"
#include "Utilities/TMPL.hpp"

/// \cond
namespace domain::Tags {
template <size_t Dim, typename Frame>
struct Coordinates;
}  // namespace domain::Tags
/// \endcond

namespace CurvedScalarWave::BoundaryConditions {

/*!
 * \brief Implements constraint-preserving boundary conditions with a second
 * order Bayliss-Turkel radiation boundary condition.
 *
 * \details The Bayliss-Turkel boundary conditions are technically only valid in
 * flat space and should therefore only be used at boundaries where the
 * background spacetime is approximately Minkwoski such as (sufficiently far
 * out) outer boundaries for asymptotically flat spacetimes. Small reflections
 * are still likely to occur.
 *
 * The constraint-preserving part of the boundary conditions are set on the time
 * derivatives of all evolved fields. The physical Bayliss-Turkel boundary
 * conditions are additionally set onto the time derivative of \f$\Pi\f$.
 *
 * The constraints are defined as follows:
 *
 * \f{align*}{
 *  \mathcal{C}_i&=\partial_i\Psi - \Phi_i=0, \\
 *  \mathcal{C}_{ij}&=\partial_{[i}\Phi_{j]}=0
 * \f}
 * Inspection of the constraint evolution system (Eqs. 29-30 in
 * \cite Holst2004wt) shows that the constraints themselves are characteristic
 * fields. We can derive constraint boundary conditions the same way
 * \cite Kidder2004rw does for the Einstein equations:
 *
 * We express the constraints in terms of (evolution) characeristic fields and
 * demand that the normal component of the constraint has to be zero when
 * flowing into the boundary i.e. there are no constraints flowing into our
 * numerical domain:
 *
 * \f{align*}{
 * 0 &= n^i \mathcal{C}_i &= n^i \partial_i w^\Psi - \frac{1}{2}(w^{+} - w^-) +
 * n^i
 * w_i^0 \\
 * (n^i \partial_i w^\Psi)_{BC}  &= \frac{1}{2}(w^{+} - w^-) - n^i w_i^0
 * \f}
 *
 * and
 *
 * \f{align*}{
 * 0 &= 2 n^i \mathcal{C}_{ij} = n^i \partial_i w^0_j + \frac{1}{2}n^i n_j
 * (\partial_i w^+ - \partial_i w^-) - \frac{1}{2}(\partial_j w^+ - \partial_j
 * w^-) - n^i \partial_j w^0_i \\
 * (n^i \partial_i w^0_j)_{BC} &= - \frac{1}{2}n^i
 * n_j (\partial_i w^+ - \partial_i w^-) + \frac{1}{2}(\partial_j w^+ +
 * \partial_j w^-) + n^i \partial_j w^0_i \f}
 *
 * This condition is applied to the time derivative using the Bjorhus condition
 * \cite Bjorhus1995 :
 * \f{align*}{
 * \partial_t u^\alpha + A^{i \alpha}_\beta \partial_i u^\beta &= F^\alpha \\
 *  e^{\hat{\alpha}}_\alpha (\partial_t u^\alpha + A^{i \alpha}_\beta
 * \partial_i u^\beta) &= e^{\hat{\alpha}}_\alpha F^\alpha  \\
 * d_t u^{\hat{\alpha}} + e^{\hat{\alpha}}_\alpha A^{i
 * \alpha}_\beta(P^k_i + n^k n_i) \partial_k u^\beta &= e^{\hat{\alpha}}_\alpha
 * F^\alpha  \\
 * d_t u^{\hat{\alpha}} + \lambda_{(\hat{\alpha})} n^k \partial_k
 * u^{\hat{\alpha}} + e^{\hat{\alpha}}_\alpha A^{i \alpha}_\beta P^k_i
 * \partial_k u^\beta &= e^{\hat{\alpha}}_\alpha F^\alpha
 * \f}
 *
 * Defining the volume time derivative of the characteristic fields as:
 * \f{equation*}{
 * D_t u^{\hat{\alpha}} \equiv e^{\hat{\alpha}}_\alpha (- A^{i \alpha}_\beta
 * \partial_i u^\beta + F^\alpha)
 * \f}
 *
 * The boundary conditions are now formulated as follows:
 *
 * \f{equation*}{
 *  d_t u^{\hat{\alpha}} = D_t u^{\hat{\alpha}} + \lambda_{(\hat{\alpha})}
 * (n^i\partial_i u^{\hat{\alpha}} - (n^i\partial_i u^{\hat{\alpha}})_{BC})
 * \f}
 *
 * Using the condition that there are no incoming constraint fields, this gives:
 *
 * \f{align*}{
 * d_t Z^1 &= D_t w^\Psi + \lambda_\Psi n^i \mathcal{C}_i \\
 * d_t Z^2_i &= D_t w^0_i + 2 \lambda_0 n^i \mathcal{C}_{ij}
 * \f}
 *
 * The Bayliss-Turkel boundary conditions are given by:
 *
 * \f{align*}{
 *  \prod_{l=1}^m\left(\partial_t + \partial_r + \frac{2l-1}{r}\right)\Psi=0,
 * \f}
 *
 * which we expand here to second order (\f$m=2\f$) to derive conditions for
 * \f$\partial_t\Pi^{\mathrm{BC}}\f$:
 *
 * \f{align*}{
 *  \partial_t\Pi^{\mathrm{BC}}
 *   &=\left(\partial_t\partial_r + \partial_r\partial_t +
 *     \partial_r^2+\frac{4}{r}\partial_t
 *     +\frac{4}{r}\partial_r + \frac{2}{r^2}\right)\Psi \notag \\
 *     &=\left((2n^i + \beta^i) \partial_t \Phi_i + n^i n^j\partial_i\Phi_j +
 *     \frac{4}{r}\partial_t\Psi + \frac{4}{r}n^i\Phi_i + \frac{2}{r^2}\Psi
 * \right) / \alpha.
 * \f}
 *
 *
 * This derivation makes the following assumptions:
 *
 * - The lapse, shift, normal vector and radius are time-independent,
 * \f$\partial_t \alpha = \partial_t \beta^i = \partial_t n^i = \partial_t r =
 * 0\f$. If necessary, these time derivatives can be accounted for in the future
 * by inserting the appropriate terms in a straightforward manner.
 *
 * - The outer boundary is spherical. It might be possible to generalize this
 * condition but we have not tried this.
 *
 *  * The boundary conditions to the time derivative of the evolved variables
 * are then given by:
 *
 * The full boundary conditions, as applied to the time derivative of each
 * evolved field are then given by:
 * \f{align*}{ \partial_{t}
 * \Psi&\to\partial_{t}\Psi +
 *                    \lambda_\Psi n^i \mathcal{C}_i, \\
 * \partial_{t}\Pi&\to\partial_{t}\Pi-\left(\partial_t\Pi
 *                  - \partial_t\Pi^{\mathrm{BC}}\right)
 *                  +\gamma_2\lambda_\Psi n^i \mathcal{C}_i
 *                  =\partial_t\Pi^{\mathrm{BC}}
 *                  +\gamma_2\lambda_\Psi n^i \mathcal{C}_i, \\
 * \partial_{t}\Phi_i&\to\partial_{t}\Phi_i+ 2 \lambda_0 n^j \mathcal{C}_{ji}.
 * \f}
 *
 */
template <size_t Dim>
class ConstraintPreservingSphericalRadiation final
    : public BoundaryCondition<Dim> {
 public:
  using options = tmpl::list<>;
  static constexpr Options::String help{
      "Constraint-preserving boundary conditions with a second order "
      "Bayliss-Turkel radiation boundary condition."};
  ConstraintPreservingSphericalRadiation() = default;
  ConstraintPreservingSphericalRadiation(
      ConstraintPreservingSphericalRadiation&&) = default;
  ConstraintPreservingSphericalRadiation& operator=(
      ConstraintPreservingSphericalRadiation&&) = default;
  ConstraintPreservingSphericalRadiation(
      const ConstraintPreservingSphericalRadiation&) = default;
  ConstraintPreservingSphericalRadiation& operator=(
      const ConstraintPreservingSphericalRadiation&) = default;
  ~ConstraintPreservingSphericalRadiation() override = default;

  explicit ConstraintPreservingSphericalRadiation(CkMigrateMessage* msg);

  WRAPPED_PUPable_decl_base_template(
      domain::BoundaryConditions::BoundaryCondition,
      ConstraintPreservingSphericalRadiation);

  auto get_clone() const -> std::unique_ptr<
      domain::BoundaryConditions::BoundaryCondition> override;

  static constexpr evolution::BoundaryConditions::Type bc_type =
      evolution::BoundaryConditions::Type::TimeDerivative;

  void pup(PUP::er& p) override;

  using dg_interior_evolved_variables_tags =
      tmpl::list<Tags::Psi, Tags::Phi<Dim>>;
  using dg_interior_temporary_tags =
      tmpl::list<domain::Tags::Coordinates<Dim, Frame::Inertial>,
                 Tags::ConstraintGamma1, Tags::ConstraintGamma2,
                 gr::Tags::Lapse<DataVector>, gr::Tags::Shift<DataVector, Dim>>;
  using dg_interior_dt_vars_tags =
      tmpl::list<::Tags::dt<Tags::Psi>, ::Tags::dt<Tags::Pi>,
                 ::Tags::dt<Tags::Phi<Dim>>>;
  using dg_interior_deriv_vars_tags = tmpl::list<
      ::Tags::deriv<Tags::Psi, tmpl::size_t<Dim>, Frame::Inertial>,
      ::Tags::deriv<Tags::Pi, tmpl::size_t<Dim>, Frame::Inertial>,
      ::Tags::deriv<Tags::Phi<Dim>, tmpl::size_t<Dim>, Frame::Inertial>>;
  using dg_gridless_tags = tmpl::list<>;

  std::optional<std::string> dg_time_derivative(
      gsl::not_null<Scalar<DataVector>*> dt_psi_correction,
      gsl::not_null<Scalar<DataVector>*> dt_pi_correction,
      gsl::not_null<tnsr::i<DataVector, Dim, Frame::Inertial>*>
          dt_phi_correction,
      const std::optional<tnsr::I<DataVector, Dim>>& face_mesh_velocity,
      const tnsr::i<DataVector, Dim>& normal_covector,
      const tnsr::I<DataVector, Dim>& normal_vector,
      const Scalar<DataVector>& psi, const tnsr::i<DataVector, Dim>& phi,
      const tnsr::I<DataVector, Dim, Frame::Inertial>& coords,
      const Scalar<DataVector>& gamma1, const Scalar<DataVector>& gamma2,
      const Scalar<DataVector>& lapse, const tnsr::I<DataVector, Dim>& shift,
      const Scalar<DataVector>& logical_dt_psi,
      const Scalar<DataVector>& logical_dt_pi,
      const tnsr::i<DataVector, Dim>& logical_dt_phi,
      const tnsr::i<DataVector, Dim>& d_psi,
      const tnsr::i<DataVector, Dim>& d_pi,
      const tnsr::ij<DataVector, Dim>& d_phi) const;
};
}  // namespace CurvedScalarWave::BoundaryConditions
