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
#include "Options/Options.hpp"
#include "Parallel/CharmPupable.hpp"
#include "PointwiseFunctions/GeneralRelativity/Tags.hpp"
#include "Utilities/Gsl.hpp"
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
 * derivatives of the evolved fields according to \cite Holst2004wt . The
 * physical Bayliss-Turkel boundary conditions are additionally set onto the
 * time derivative of \f$\Pi\f$.
 *
 * The constraints are defined as follows:
 *
 * \f{align*}{
 *  \mathcal{C}_i&=\partial_i\Psi - \Phi_i=0, \\
 *  \mathcal{C}_{ij}&=\partial_{[i}\Phi_{j]}=0
 * \f}
 *
 * The boundary conditions are then given by:
 *
 * \f{align*}{
 * \partial_{t} \Psi&\to\partial_{t}\Psi +
 *                    \lambda_\Psi n^i \mathcal{C}_i, \\
 * \partial_{t}\Pi&\to\partial_{t}\Pi-\left(\partial_t\Pi
 *                  - \partial_t\Pi^{\mathrm{BC}}\right)
 *                  +\gamma_2\lambda_\Psi n^i \mathcal{C}_i
 *                  =\partial_t\Pi^{\mathrm{BC}}
 *                  +\gamma_2\lambda_\Psi n^i \mathcal{C}_i, \\
 * \partial_{t}\Phi_i&\to\partial_{t}\Phi_i+
 *                     \lambda_0n^jP^k{}_i\mathcal{C}_{jk}
 *   = \partial_{t}\Phi_i+ \lambda_0n^j \mathcal{C}_{ji}.
 * \f}
 *
 * These conditions are equivalent to equations (40) and (41) of
 * \cite Holst2004wt if the shift vector is parallel to the normal of the outer
 * boundary. The Bayliss-Turkel boundary conditions are given by:
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
      ConstraintPreservingSphericalRadiation&&) noexcept = default;
  ConstraintPreservingSphericalRadiation& operator=(
      ConstraintPreservingSphericalRadiation&&) noexcept = default;
  ConstraintPreservingSphericalRadiation(
      const ConstraintPreservingSphericalRadiation&) = default;
  ConstraintPreservingSphericalRadiation& operator=(
      const ConstraintPreservingSphericalRadiation&) = default;
  ~ConstraintPreservingSphericalRadiation() override = default;

  explicit ConstraintPreservingSphericalRadiation(
      CkMigrateMessage* msg) noexcept;

  WRAPPED_PUPable_decl_base_template(
      domain::BoundaryConditions::BoundaryCondition,
      ConstraintPreservingSphericalRadiation);

  auto get_clone() const noexcept -> std::unique_ptr<
      domain::BoundaryConditions::BoundaryCondition> override;

  static constexpr evolution::BoundaryConditions::Type bc_type =
      evolution::BoundaryConditions::Type::TimeDerivative;

  void pup(PUP::er& p) override;

  using dg_interior_evolved_variables_tags = tmpl::list<Phi<Dim>, Psi>;
  using dg_interior_temporary_tags =
      tmpl::list<domain::Tags::Coordinates<Dim, Frame::Inertial>,
                 Tags::ConstraintGamma1, Tags::ConstraintGamma2,
                 gr::Tags::Lapse<DataVector>,
                 gr::Tags::Shift<Dim, Frame::Inertial, DataVector>>;
  using dg_interior_dt_vars_tags =
      tmpl::list<::Tags::dt<Pi>, ::Tags::dt<Phi<Dim>>, ::Tags::dt<Psi>>;
  using dg_interior_deriv_vars_tags =
      tmpl::list<::Tags::deriv<Psi, tmpl::size_t<Dim>, Frame::Inertial>,
                 ::Tags::deriv<Phi<Dim>, tmpl::size_t<Dim>, Frame::Inertial>>;
  using dg_gridless_tags = tmpl::list<>;

  std::optional<std::string> dg_time_derivative(
      gsl::not_null<Scalar<DataVector>*> dt_pi_correction,
      gsl::not_null<tnsr::i<DataVector, Dim, Frame::Inertial>*>
          dt_phi_correction,
      gsl::not_null<Scalar<DataVector>*> dt_psi_correction,
      const std::optional<tnsr::I<DataVector, Dim>>& face_mesh_velocity,
      const tnsr::i<DataVector, Dim>& normal_covector,
      const tnsr::I<DataVector, Dim>& normal_vector,
      const tnsr::i<DataVector, Dim>& phi, const Scalar<DataVector>& psi,
      const tnsr::I<DataVector, Dim, Frame::Inertial>& coords,
      const Scalar<DataVector>& gamma1, const Scalar<DataVector>& gamma2,
      const Scalar<DataVector>& lapse, const tnsr::I<DataVector, Dim>& shift,
      const Scalar<DataVector>& dt_pi, const tnsr::i<DataVector, Dim>& dt_phi,
      const Scalar<DataVector>& dt_psi, const tnsr::i<DataVector, Dim>& d_psi,
      const tnsr::ij<DataVector, Dim>& d_phi) const noexcept;
};
}  // namespace CurvedScalarWave::BoundaryConditions
