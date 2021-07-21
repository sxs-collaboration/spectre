// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <array>
#include <cstddef>
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
#include "Evolution/Systems/GeneralizedHarmonic/BoundaryConditions/BoundaryCondition.hpp"
#include "Evolution/Systems/GeneralizedHarmonic/ConstraintDamping/Tags.hpp"
#include "Evolution/Systems/GeneralizedHarmonic/Tags.hpp"
#include "Options/Options.hpp"
#include "Parallel/CharmPupable.hpp"
#include "PointwiseFunctions/GeneralRelativity/Tags.hpp"
#include "Time/Tags.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/TMPL.hpp"

namespace GeneralizedHarmonic::BoundaryConditions {
/*!
 * \brief Sets freezing boundary conditions using the Bjorhus method.
 *
 * \details Bjorhus \cite Bjorhus1995 prescribes imposing boundary conditions,
 * on external boundaries, as corrections to the characteristic projections of
 * the right-hand-sides of the evolution equations. In this class we freeze the
 * characteristic projections by setting their time derivatives to zero,
 * whenever these projections have incoming speeds on the external boundary.
 * These corrections to time-derivatives are subsequently projected back to get
 * corrections to the time-derivatives of the evolved fields \f$\Psi_{ab},
 * \Pi_{ab}, \Phi_{iab}\f$.
 */
template <size_t Dim>
class FreezingBjorhus final : public BoundaryCondition<Dim> {
 public:
  using options = tmpl::list<>;
  static constexpr Options::String help{
      "Freezing boundary conditions setting the value of the"
      "time derivatives of the spacetime metric, Phi and Pi to expressions that"
      "freeze any incoming characteristic modes."};
  static std::string name() noexcept { return "FreezingBjorhus"; }

  FreezingBjorhus() = default;
  /// \cond
  FreezingBjorhus(FreezingBjorhus&&) noexcept = default;
  FreezingBjorhus& operator=(FreezingBjorhus&&) noexcept = default;
  FreezingBjorhus(const FreezingBjorhus&) = default;
  FreezingBjorhus& operator=(const FreezingBjorhus&) = default;
  /// \endcond
  ~FreezingBjorhus() override = default;

  explicit FreezingBjorhus(CkMigrateMessage* msg) noexcept;

  WRAPPED_PUPable_decl_base_template(
      domain::BoundaryConditions::BoundaryCondition, FreezingBjorhus);

  auto get_clone() const noexcept -> std::unique_ptr<
      domain::BoundaryConditions::BoundaryCondition> override;

  static constexpr evolution::BoundaryConditions::Type bc_type =
      evolution::BoundaryConditions::Type::TimeDerivative;

  void pup(PUP::er& p) override;

  using dg_interior_evolved_variables_tags = tmpl::list<>;
  using dg_interior_temporary_tags = tmpl::list<
      ConstraintDamping::Tags::ConstraintGamma1,
      ConstraintDamping::Tags::ConstraintGamma2, gr::Tags::Lapse<DataVector>,
      gr::Tags::Shift<Dim, Frame::Inertial, DataVector>,
      gr::Tags::InverseSpacetimeMetric<Dim, Frame::Inertial, DataVector>>;
  using dg_interior_dt_vars_tags = tmpl::list<
      ::Tags::dt<gr::Tags::SpacetimeMetric<Dim, Frame::Inertial, DataVector>>,
      ::Tags::dt<Tags::Pi<Dim, Frame::Inertial>>,
      ::Tags::dt<Tags::Phi<Dim, Frame::Inertial>>>;
  using dg_interior_deriv_vars_tags = tmpl::list<>;
  using dg_gridless_tags = tmpl::list<>;

  std::optional<std::string> dg_time_derivative(
      gsl::not_null<tnsr::aa<DataVector, Dim, Frame::Inertial>*>
          dt_spacetime_metric_correction,
      gsl::not_null<tnsr::aa<DataVector, Dim, Frame::Inertial>*>
          dt_pi_correction,
      gsl::not_null<tnsr::iaa<DataVector, Dim, Frame::Inertial>*>
          dt_phi_correction,

      const std::optional<tnsr::I<DataVector, Dim, Frame::Inertial>>&
          face_mesh_velocity,
      const tnsr::i<DataVector, Dim, Frame::Inertial>& normal_covector,
      const tnsr::I<DataVector, Dim, Frame::Inertial>& /*normal_vector*/,
      // c.f. dg_interior_evolved_variables_tags
      // c.f. dg_interior_temporary_tags
      const Scalar<DataVector>& gamma1, const Scalar<DataVector>& gamma2,
      const Scalar<DataVector>& lapse,
      const tnsr::I<DataVector, Dim, Frame::Inertial>& shift,
      const tnsr::AA<DataVector, Dim, Frame::Inertial>&
          inverse_spacetime_metric,
      // c.f. dg_interior_dt_vars_tags
      const tnsr::aa<DataVector, Dim, Frame::Inertial>& dt_spacetime_metric,
      const tnsr::aa<DataVector, Dim, Frame::Inertial>& dt_pi,
      const tnsr::iaa<DataVector, Dim, Frame::Inertial>& dt_phi
      // c.f. dg_interior_deriv_vars_tags
  ) const noexcept;
};
}  // namespace GeneralizedHarmonic::BoundaryConditions
