// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>
#include <memory>
#include <pup.h>
#include <string>
#include <vector>

#include "DataStructures/Tensor/TypeAliases.hpp"
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
 * \brief Impose flat spacetime at this boundary
 *
 * Impose \f$\psi=1\f$, \f$\alpha\psi=1\f$, \f$\beta_\mathrm{excess}^i=0\f$ on
 * this boundary, where \f$\psi\f$ is the conformal factor, \f$\alpha\f$ is the
 * lapse and \f$\beta_\mathrm{excess}^i=\beta^i-\beta_\mathrm{background}^i\f$
 * is the shift excess (see `Xcts::Tags::ShiftExcess` for details on the split
 * of the shift in background and excess). Note that this choice only truly
 * represents flatness if the conformal background metric is flat.
 *
 * \tparam EnabledEquations The subset of XCTS equations that are being solved
 */
template <Xcts::Equations EnabledEquations>
class Flatness : public elliptic::BoundaryConditions::BoundaryCondition<3> {
 private:
  using Base = elliptic::BoundaryConditions::BoundaryCondition<3>;

 public:
  using options = tmpl::list<>;
  static constexpr Options::String help =
      "Impose flat spacetime at this boundary.";

  Flatness() = default;
  Flatness(const Flatness&) = default;
  Flatness& operator=(const Flatness&) = default;
  Flatness(Flatness&&) = default;
  Flatness& operator=(Flatness&&) = default;
  ~Flatness() = default;

  /// \cond
  explicit Flatness(CkMigrateMessage* m) : Base(m) {}
  using PUP::able::register_constructor;
  WRAPPED_PUPable_decl_template(Flatness);
  /// \endcond

  std::unique_ptr<domain::BoundaryConditions::BoundaryCondition> get_clone()
      const override {
    return std::make_unique<Flatness>(*this);
  }

  std::vector<elliptic::BoundaryConditionType> boundary_condition_types()
      const override {
    return {[]() {
              if constexpr (EnabledEquations == Xcts::Equations::Hamiltonian) {
                return 1;
              } else if constexpr (EnabledEquations ==
                                   Xcts::Equations::HamiltonianAndLapse) {
                return 2;
              } else {
                return 5;
              }
            }(),
            elliptic::BoundaryConditionType::Dirichlet};
  }

  using argument_tags = tmpl::list<>;
  using volume_tags = tmpl::list<>;

  static void apply(
      gsl::not_null<Scalar<DataVector>*> conformal_factor,
      gsl::not_null<Scalar<DataVector>*> n_dot_conformal_factor_gradient);

  static void apply(
      gsl::not_null<Scalar<DataVector>*> conformal_factor,
      gsl::not_null<Scalar<DataVector>*> lapse_times_conformal_factor,
      gsl::not_null<Scalar<DataVector>*> n_dot_conformal_factor_gradient,
      gsl::not_null<Scalar<DataVector>*>
          n_dot_lapse_times_conformal_factor_gradient);

  static void apply(
      gsl::not_null<Scalar<DataVector>*> conformal_factor,
      gsl::not_null<Scalar<DataVector>*> lapse_times_conformal_factor,
      gsl::not_null<tnsr::I<DataVector, 3>*> shift_excess,
      gsl::not_null<Scalar<DataVector>*> n_dot_conformal_factor_gradient,
      gsl::not_null<Scalar<DataVector>*>
          n_dot_lapse_times_conformal_factor_gradient,
      gsl::not_null<tnsr::I<DataVector, 3>*> n_dot_longitudinal_shift_excess);

  using argument_tags_linearized = tmpl::list<>;
  using volume_tags_linearized = tmpl::list<>;

  static void apply_linearized(
      gsl::not_null<Scalar<DataVector>*> conformal_factor_correction,
      gsl::not_null<Scalar<DataVector>*>
          n_dot_conformal_factor_gradient_correction);

  static void apply_linearized(
      gsl::not_null<Scalar<DataVector>*> conformal_factor_correction,
      gsl::not_null<Scalar<DataVector>*>
          lapse_times_conformal_factor_correction,
      gsl::not_null<Scalar<DataVector>*>
          n_dot_conformal_factor_gradient_correction,
      gsl::not_null<Scalar<DataVector>*>
          n_dot_lapse_times_conformal_factor_gradient_correction);

  static void apply_linearized(
      gsl::not_null<Scalar<DataVector>*> conformal_factor_correction,
      gsl::not_null<Scalar<DataVector>*>
          lapse_times_conformal_factor_correction,
      gsl::not_null<tnsr::I<DataVector, 3>*> shift_excess_correction,
      gsl::not_null<Scalar<DataVector>*>
          n_dot_conformal_factor_gradient_correction,
      gsl::not_null<Scalar<DataVector>*>
          n_dot_lapse_times_conformal_factor_gradient_correction,
      gsl::not_null<tnsr::I<DataVector, 3>*>
          n_dot_longitudinal_shift_excess_correction);
};

template <Xcts::Equations EnabledEquations>
bool operator==(const Flatness<EnabledEquations>& lhs,
                const Flatness<EnabledEquations>& rhs);

template <Xcts::Equations EnabledEquations>
bool operator!=(const Flatness<EnabledEquations>& lhs,
                const Flatness<EnabledEquations>& rhs);

}  // namespace Xcts::BoundaryConditions
