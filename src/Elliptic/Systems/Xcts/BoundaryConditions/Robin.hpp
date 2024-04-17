// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>
#include <memory>
#include <pup.h>
#include <string>
#include <vector>

#include "DataStructures/Tensor/TypeAliases.hpp"
#include "Domain/Tags.hpp"
#include "Domain/Tags/FaceNormal.hpp"
#include "Elliptic/BoundaryConditions/BoundaryCondition.hpp"
#include "Elliptic/BoundaryConditions/BoundaryConditionType.hpp"
#include "Elliptic/Systems/Xcts/FluxesAndSources.hpp"
#include "Options/String.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/Serialization/CharmPupable.hpp"
#include "Utilities/TMPL.hpp"

/// \cond
class DataVector;
/// \endcond

namespace Xcts::BoundaryConditions {

/*!
 * \brief Impose Robin boundary conditions at the outer boundary
 *
 * Impose $\partial_r(r\psi)=0$, $\partial_r(\alpha\psi)=0$, and
 * $\partial_r(r\beta_\mathrm{excess}^j)=0$ on the boundary. These Robin
 * boundary conditions incur an error of order $1/R^2$, where $R$ is the outer
 * radius of the domain. This allows to place the outer boundary much closer
 * to the center than for Dirichlet boundary conditions
 * (Xcts::BoundaryConditions::Flatness), which incur an error of order $1/R$.
 *
 * The Robin boundary conditions are imposed as Neumann-type as follows:
 *
 * \begin{align*}
 * (n^i \partial_i \psi)_N &= -\frac{\psi}{r}, \\
 * (n^i \partial_i (\alpha\psi))_N &= -\frac{\alpha\psi}{r}, \\
 * (n^i (L\beta)^{ij})_N = n^i (L\beta)^{ij} - \left[
 *   \frac{\beta_\mathrm{excess}^j}{r}
 *   + \frac{1}{3} n^j \frac{n^i \beta_\mathrm{excess}^i}{r}
 *   + n^i \partial_i \beta_\mathrm{excess}^j
 *   + \frac{1}{3} n^j n^i n_k \partial_i \beta_\mathrm{excess}^k
 * \right]
 * \end{align*}
 *
 * Here, the condition on the longitudinal shift is derived by imposing the
 * Robin boundary condition
 * $n^i\partial_i\beta_\mathrm{excess}^j=-\beta_\mathrm{excess}^j/r$ only on the
 * normal component of the shift gradient. To do this we can use the projection
 * operator $P_{ij}=\delta_{ij}-n_i n_j$ to set the shift gradient to
 * $\partial_i\beta_\mathrm{excess}^j=P_{ik}\partial_k\beta_\mathrm{excess}^j
 * -n_i \beta_\mathrm{excess}^j/r$ and then apply the longitudinal operator.
 *
 * \tparam EnabledEquations The subset of XCTS equations that are being solved
 */
template <Xcts::Equations EnabledEquations>
class Robin : public elliptic::BoundaryConditions::BoundaryCondition<3> {
 private:
  using Base = elliptic::BoundaryConditions::BoundaryCondition<3>;

 public:
  using options = tmpl::list<>;
  static constexpr Options::String help =
      "Impose Robin boundary conditions at the outer boundary. "
      "They incur an error of order 1/R^2, where R is the outer radius.";

  Robin() = default;
  Robin(const Robin&) = default;
  Robin& operator=(const Robin&) = default;
  Robin(Robin&&) = default;
  Robin& operator=(Robin&&) = default;
  ~Robin() override = default;

  /// \cond
  explicit Robin(CkMigrateMessage* m) : Base(m) {}
  using PUP::able::register_constructor;
  WRAPPED_PUPable_decl_template(Robin);
  /// \endcond

  std::unique_ptr<domain::BoundaryConditions::BoundaryCondition> get_clone()
      const override;

  std::vector<elliptic::BoundaryConditionType> boundary_condition_types()
      const override;

  using argument_tags =
      tmpl::list<domain::Tags::Coordinates<3, Frame::Inertial>,
                 domain::Tags::FaceNormal<3, Frame::Inertial>>;
  using volume_tags = tmpl::list<>;

  void apply(gsl::not_null<Scalar<DataVector>*> conformal_factor_minus_one,
             gsl::not_null<Scalar<DataVector>*> n_dot_conformal_factor_gradient,
             const tnsr::i<DataVector, 3>& deriv_conformal_factor,
             const tnsr::I<DataVector, 3>& x,
             const tnsr::i<DataVector, 3>& face_normal) const;

  void apply(
      gsl::not_null<Scalar<DataVector>*> conformal_factor_minus_one,
      gsl::not_null<Scalar<DataVector>*> lapse_times_conformal_factor_minus_one,
      gsl::not_null<Scalar<DataVector>*> n_dot_conformal_factor_gradient,
      gsl::not_null<Scalar<DataVector>*>
          n_dot_lapse_times_conformal_factor_gradient,
      const tnsr::i<DataVector, 3>& deriv_conformal_factor,
      const tnsr::i<DataVector, 3>& deriv_lapse_times_conformal_factor,
      const tnsr::I<DataVector, 3>& x,
      const tnsr::i<DataVector, 3>& face_normal) const;

  void apply(
      gsl::not_null<Scalar<DataVector>*> conformal_factor_minus_one,
      gsl::not_null<Scalar<DataVector>*> lapse_times_conformal_factor_minus_one,
      gsl::not_null<tnsr::I<DataVector, 3>*> shift_excess,
      gsl::not_null<Scalar<DataVector>*> n_dot_conformal_factor_gradient,
      gsl::not_null<Scalar<DataVector>*>
          n_dot_lapse_times_conformal_factor_gradient,
      gsl::not_null<tnsr::I<DataVector, 3>*> n_dot_longitudinal_shift_excess,
      const tnsr::i<DataVector, 3>& deriv_conformal_factor,
      const tnsr::i<DataVector, 3>& deriv_lapse_times_conformal_factor,
      const tnsr::iJ<DataVector, 3>& deriv_shift_excess,
      const tnsr::I<DataVector, 3>& x,
      const tnsr::i<DataVector, 3>& face_normal) const;

  using argument_tags_linearized =
      tmpl::list<domain::Tags::Coordinates<3, Frame::Inertial>,
                 domain::Tags::FaceNormal<3, Frame::Inertial>>;
  using volume_tags_linearized = tmpl::list<>;

  void apply_linearized(
      gsl::not_null<Scalar<DataVector>*> conformal_factor_correction,
      gsl::not_null<Scalar<DataVector>*>
          n_dot_conformal_factor_gradient_correction,
      const tnsr::i<DataVector, 3>& deriv_conformal_factor_correction,
      const tnsr::I<DataVector, 3>& x,
      const tnsr::i<DataVector, 3>& face_normal) const;

  void apply_linearized(
      gsl::not_null<Scalar<DataVector>*> conformal_factor_correction,
      gsl::not_null<Scalar<DataVector>*>
          lapse_times_conformal_factor_correction,
      gsl::not_null<Scalar<DataVector>*>
          n_dot_conformal_factor_gradient_correction,
      gsl::not_null<Scalar<DataVector>*>
          n_dot_lapse_times_conformal_factor_gradient_correction,
      const tnsr::i<DataVector, 3>& deriv_conformal_factor_correction,
      const tnsr::i<DataVector, 3>&
          deriv_lapse_times_conformal_factor_correction,
      const tnsr::I<DataVector, 3>& x,
      const tnsr::i<DataVector, 3>& face_normal) const;

  void apply_linearized(
      gsl::not_null<Scalar<DataVector>*> conformal_factor_correction,
      gsl::not_null<Scalar<DataVector>*>
          lapse_times_conformal_factor_correction,
      gsl::not_null<tnsr::I<DataVector, 3>*> shift_excess_correction,
      gsl::not_null<Scalar<DataVector>*>
          n_dot_conformal_factor_gradient_correction,
      gsl::not_null<Scalar<DataVector>*>
          n_dot_lapse_times_conformal_factor_gradient_correction,
      gsl::not_null<tnsr::I<DataVector, 3>*>
          n_dot_longitudinal_shift_excess_correction,
      const tnsr::i<DataVector, 3>& deriv_conformal_factor_correction,
      const tnsr::i<DataVector, 3>&
          deriv_lapse_times_conformal_factor_correction,
      const tnsr::iJ<DataVector, 3>& deriv_shift_excess_correction,
      const tnsr::I<DataVector, 3>& x,
      const tnsr::i<DataVector, 3>& face_normal) const;
};

template <Xcts::Equations LocalEnabledEquations>
bool operator==(const Robin<LocalEnabledEquations>& lhs,
                const Robin<LocalEnabledEquations>& rhs);

template <Xcts::Equations EnabledEquations>
bool operator!=(const Robin<EnabledEquations>& lhs,
                const Robin<EnabledEquations>& rhs);

}  // namespace Xcts::BoundaryConditions
