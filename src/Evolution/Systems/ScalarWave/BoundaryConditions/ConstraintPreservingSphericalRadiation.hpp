// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <memory>
#include <optional>
#include <pup.h>
#include <string>
#include <type_traits>

#include "DataStructures/DataBox/Prefixes.hpp"
#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "DataStructures/Variables.hpp"
#include "Evolution/BoundaryConditions/Type.hpp"
#include "Evolution/Systems/ScalarWave/BoundaryConditions/BoundaryCondition.hpp"
#include "Evolution/Systems/ScalarWave/Tags.hpp"
#include "NumericalAlgorithms/LinearOperators/PartialDerivatives.hpp"
#include "Options/Options.hpp"
#include "Parallel/CharmPupable.hpp"
#include "PointwiseFunctions/AnalyticData/Tags.hpp"
#include "PointwiseFunctions/AnalyticSolutions/AnalyticSolution.hpp"
#include "Time/Tags.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/TMPL.hpp"

/// \cond
namespace domain::Tags {
template <size_t Dim, typename Frame>
struct Coordinates;
}  // namespace domain::Tags
/// \endcond

namespace ScalarWave::BoundaryConditions {
namespace detail {
/// The type of spherical radiation boundary condition to impose
enum class ConstraintPreservingSphericalRadiationType {
  /// Impose \f$(\partial_t + \partial_r)\Psi=0\f$
  Sommerfeld,
  /// Impose \f$(\partial_t + \partial_r + r^{-1})\Psi=0\f$
  FirstOrderBaylissTurkel,
  /// Imposes a second-order Bayliss-Turkel boundary condition
  SecondOrderBaylissTurkel
};

ConstraintPreservingSphericalRadiationType
convert_constraint_preserving_spherical_radiation_type_from_yaml(
    const Options::Option& options);
}  // namespace detail

/*!
 * \brief Constraint-preserving spherical radiation boundary condition that
 * seeks to avoid ingoing constraint violations and radiation.
 *
 * The constraint-preserving part of the boundary condition imposes the
 * following condition on the time derivatives of the characteristic fields:
 *
 * \f{align*}{
 *   d_tw^\Psi&\to d_tw^{\Psi}+\lambda_{\Psi}n^i\mathcal{C}_i, \\
 *   d_tw^0_i&\to d_tw^{0}_i+\lambda_{0}n^jP^k_i\mathcal{C}_{ik},
 * \f}
 *
 * where
 *
 * \f{align*}{
 * P^k{}_i=\delta^k_i-n^kn_i
 * \f}
 *
 * projects a tensor onto the spatial surface to which \f$n_i\f$ is normal, and
 * \f$d_t w\f$ is the evolved to characteristic field transformation applied to
 * the time derivatives of the evolved fields. That is,
 *
 * \f{align*}{
 * d_t w^\Psi&=\partial_t \Psi, \\
 * d_t w_i^0&=(\delta^k_i-n^k n_i)\partial_t \Phi_k, \\
 * d_t w^{\pm}&=\partial_t\Pi\pm n^k\partial_t\Phi_k - \gamma_2\partial_t\Psi.
 * \f}
 *
 * The constraints are defined as:
 *
 * \f{align*}{
 *  \mathcal{C}_i&=\partial_i\Psi - \Phi_i=0, \\
 *  \mathcal{C}_{ij}&=\partial_{[i}\Phi_{j]}=0
 * \f}
 *
 * Radiation boundary conditions impose a condition on \f$\Pi\f$ or its time
 * derivative. We denote the boundary condition value of the time derivative of
 * \f$\Pi\f$ by \f$\partial_t\Pi^{\mathrm{BC}}\f$. With this, we can impose
 * boundary conditions on the time derivatives of the evolved variables as
 * follows:
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
 * Below we assume the normal vector \f$n^i\f$ is the radial unit normal vector.
 * That is, we assume the outer boundary is spherical. A Sommerfeld
 * \cite Sommerfeld1949 radiation condition is given by
 *
 * \f{align*}{
 *  \partial_t\Psi=n^i\Phi_i
 * \f}
 *
 * Or, assuming that \f$\partial_tn^i=0\f$ (or is very small),
 *
 * \f{align*}{
 *  \partial_t\Pi^{\mathrm{BC}}=n^i\partial_t\Phi_i
 * \f}
 *
 * The Bayliss-Turkel \cite BaylissTurkel boundary conditions are given by:
 *
 * \f{align*}{
 *  \prod_{l=1}^m\left(\partial_t + \partial_r + \frac{2l-1}{r}\right)\Psi=0
 * \f}
 *
 * The first-order form is
 *
 * \f{align*}{
 *  \partial_t\Pi^{\mathrm{BC}}=n^i\partial_t\Phi_i + \frac{1}{r}\partial_t\Psi,
 * \f}
 *
 * assuming \f$\partial_t n^i=0\f$ and \f$\partial_t r=0\f$.
 *
 * The second-order boundary condition is given by,
 *
 * \f{align*}{
 *  \partial_t\Pi^{\mathrm{BC}}
 *   &=\left(\partial_t\partial_r + \partial_r\partial_t +
 *     \partial_r^2+\frac{4}{r}\partial_t
 *     +\frac{4}{r}\partial_r + \frac{2}{r^2}\right)\Psi \notag \\
 *     &=n^i(\partial_t\Phi_i-\partial_i\Pi) + n^i n^j\partial_i\Phi_j -
 *     \frac{4}{r}\Pi+\frac{4}{r}n^i\Phi_i + \frac{2}{r^2}\Psi,
 * \f}
 *
 * assuming \f$\partial_t n^i=0\f$ and \f$\partial_t r=0\f$.
 *
 * The moving mesh can be accounted for by using
 *
 * \f{align*}{
 * \partial_t r = \frac{1}{r}x^i\delta_{ij}\partial_t x^j
 * \f}
 *
 * \note It is not clear if \f$\partial_t\Phi_i\f$ should be replaced by
 * \f$-\partial_i\Pi\f$, which is the evolution equation but without the
 * constraint.
 *
 * \note On a moving mesh the characteristic speeds change according to
 * \f$\lambda\to\lambda-v^i_gn_i\f$ where \f$v^i_g\f$ is the mesh velocity.
 *
 * \note For the scalar wave system \f$\lambda_0 = \lambda_\psi\f$
 *
 * \warning The boundary conditions are implemented assuming the outer boundary
 * is spherical. It might be possible to generalize the condition to
 * non-spherical boundaries by using \f$x^i/r\f$ instead of \f$n^i\f$, but this
 * hasn't been tested.
 */
template <size_t Dim>
class ConstraintPreservingSphericalRadiation final
    : public BoundaryCondition<Dim> {
 public:
  struct TypeOptionTag {
    using type = detail::ConstraintPreservingSphericalRadiationType;
    static std::string name() { return "Type"; }
    static constexpr Options::String help{
        "Whether to impose Sommerfeld, first-order Bayliss-Turkel, or "
        "second-order Bayliss-Turkel spherical radiation boundary conditions."};
  };

  using options = tmpl::list<TypeOptionTag>;
  static constexpr Options::String help{
      "Constraint-preserving spherical radiation boundary conditions setting "
      "the time derivatives of Psi, Phi, and Pi to avoid incoming constraint "
      "violations, and imposing radiation boundary conditions."};

  ConstraintPreservingSphericalRadiation(
      detail::ConstraintPreservingSphericalRadiationType type);

  ConstraintPreservingSphericalRadiation() = default;
  /// \cond
  ConstraintPreservingSphericalRadiation(
      ConstraintPreservingSphericalRadiation&&) = default;
  ConstraintPreservingSphericalRadiation& operator=(
      ConstraintPreservingSphericalRadiation&&) = default;
  ConstraintPreservingSphericalRadiation(
      const ConstraintPreservingSphericalRadiation&) = default;
  ConstraintPreservingSphericalRadiation& operator=(
      const ConstraintPreservingSphericalRadiation&) = default;
  /// \endcond
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
      tmpl::list<ScalarWave::Pi, ScalarWave::Phi<Dim>, ScalarWave::Psi>;
  using dg_interior_temporary_tags =
      tmpl::list<domain::Tags::Coordinates<Dim, Frame::Inertial>,
                 Tags::ConstraintGamma2>;
  using dg_interior_dt_vars_tags =
      tmpl::list<::Tags::dt<ScalarWave::Pi>, ::Tags::dt<ScalarWave::Phi<Dim>>,
                 ::Tags::dt<ScalarWave::Psi>>;
  using dg_interior_deriv_vars_tags = tmpl::list<
      ::Tags::deriv<ScalarWave::Pi, tmpl::size_t<Dim>, Frame::Inertial>,
      ::Tags::deriv<ScalarWave::Psi, tmpl::size_t<Dim>, Frame::Inertial>,
      ::Tags::deriv<ScalarWave::Phi<Dim>, tmpl::size_t<Dim>, Frame::Inertial>>;
  using dg_gridless_tags = tmpl::list<>;

  std::optional<std::string> dg_time_derivative(
      gsl::not_null<Scalar<DataVector>*> dt_pi_correction,
      gsl::not_null<tnsr::i<DataVector, Dim, Frame::Inertial>*>
          dt_phi_correction,
      gsl::not_null<Scalar<DataVector>*> dt_psi_correction,
      const std::optional<tnsr::I<DataVector, Dim, Frame::Inertial>>&
          face_mesh_velocity,
      const tnsr::i<DataVector, Dim, Frame::Inertial>& normal_covector,
      const Scalar<DataVector>& pi,
      const tnsr::i<DataVector, Dim, Frame::Inertial>& phi,
      const Scalar<DataVector>& psi,
      const tnsr::I<DataVector, Dim, Frame::Inertial>& coords,
      const Scalar<DataVector>& gamma2, const Scalar<DataVector>& dt_pi,
      const tnsr::i<DataVector, Dim, Frame::Inertial>& dt_phi,
      const Scalar<DataVector>& dt_psi,
      const tnsr::i<DataVector, Dim, Frame::Inertial>& d_pi,
      const tnsr::i<DataVector, Dim, Frame::Inertial>& d_psi,
      const tnsr::ij<DataVector, Dim, Frame::Inertial>& d_phi) const;

 private:
  detail::ConstraintPreservingSphericalRadiationType type_{
      detail::ConstraintPreservingSphericalRadiationType::Sommerfeld};
};
}  // namespace ScalarWave::BoundaryConditions

template <>
struct Options::create_from_yaml<
    ScalarWave::BoundaryConditions::detail::
        ConstraintPreservingSphericalRadiationType> {
  template <typename Metavariables>
  static typename ScalarWave::BoundaryConditions::detail::
      ConstraintPreservingSphericalRadiationType
      create(const Options::Option& options) {
    return ScalarWave::BoundaryConditions::detail::
        convert_constraint_preserving_spherical_radiation_type_from_yaml(
            options);
  }
};
