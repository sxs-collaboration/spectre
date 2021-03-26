// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>
#include <memory>
#include <pup.h>
#include <string>

#include "DataStructures/Tensor/TypeAliases.hpp"
#include "Elliptic/BoundaryConditions/BoundaryCondition.hpp"
#include "Options/Options.hpp"
#include "Parallel/CharmPupable.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/TMPL.hpp"

/// \cond
class DataVector;
/// \endcond

namespace Xcts::BoundaryConditions {
namespace detail {

struct FlatnessImpl {
  using options = tmpl::list<>;
  static constexpr Options::String help =
      "Impose flat spacetime at this boundary.";

  using argument_tags = tmpl::list<>;
  using volume_tags = tmpl::list<>;

  static void apply(gsl::not_null<Scalar<DataVector>*> conformal_factor,
                    gsl::not_null<Scalar<DataVector>*>
                        n_dot_conformal_factor_gradient) noexcept;

  static void apply(
      gsl::not_null<Scalar<DataVector>*> conformal_factor,
      gsl::not_null<Scalar<DataVector>*> lapse_times_conformal_factor,
      gsl::not_null<Scalar<DataVector>*> n_dot_conformal_factor_gradient,
      gsl::not_null<Scalar<DataVector>*>
          n_dot_lapse_times_conformal_factor_gradient) noexcept;

  static void apply(
      gsl::not_null<Scalar<DataVector>*> conformal_factor,
      gsl::not_null<Scalar<DataVector>*> lapse_times_conformal_factor,
      gsl::not_null<tnsr::I<DataVector, 3>*> shift_excess,
      gsl::not_null<Scalar<DataVector>*> n_dot_conformal_factor_gradient,
      gsl::not_null<Scalar<DataVector>*>
          n_dot_lapse_times_conformal_factor_gradient,
      gsl::not_null<tnsr::I<DataVector, 3>*>
          n_dot_longitudinal_shift_excess) noexcept;

  using argument_tags_linearized = tmpl::list<>;
  using volume_tags_linearized = tmpl::list<>;

  static void apply_linearized(
      gsl::not_null<Scalar<DataVector>*> conformal_factor_correction,
      gsl::not_null<Scalar<DataVector>*>
          n_dot_conformal_factor_gradient_correction) noexcept;

  static void apply_linearized(
      gsl::not_null<Scalar<DataVector>*> conformal_factor_correction,
      gsl::not_null<Scalar<DataVector>*>
          lapse_times_conformal_factor_correction,
      gsl::not_null<Scalar<DataVector>*>
          n_dot_conformal_factor_gradient_correction,
      gsl::not_null<Scalar<DataVector>*>
          n_dot_lapse_times_conformal_factor_gradient_correction) noexcept;

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
          n_dot_longitudinal_shift_excess_correction) noexcept;
};

bool operator==(const FlatnessImpl& lhs, const FlatnessImpl& rhs) noexcept;

bool operator!=(const FlatnessImpl& lhs, const FlatnessImpl& rhs) noexcept;

}  // namespace detail

// The following implements the registration and factory-creation mechanism

/// \cond
template <typename Registrars>
struct Flatness;

namespace Registrars {
struct Flatness {
  template <typename Registrars>
  using f = BoundaryConditions::Flatness<Registrars>;
};
}  // namespace Registrars
/// \endcond

/*!
 * \brief Impose flat spacetime at this boundary
 *
 * Impose \f$\psi=1\f$, \f$\alpha\psi=1\f$, \f$\beta_\mathrm{excess}^i=0\f$ on
 * this boundary, where \f$\psi\f$ is the conformal factor, \f$\alpha\f$ is the
 * lapse and \f$\beta_\mathrm{excess}^i=\beta^i-\beta_\mathrm{background}^i\f$
 * is the shift excess (see `Xcts::Tags::ShiftExcess` for details on the split
 * of the shift in background and excess). Note that this choice only truly
 * represents flatness if the conformal background metric is flat.
 */
template <typename Registrars = tmpl::list<Registrars::Flatness>>
class Flatness
    : public elliptic::BoundaryConditions::BoundaryCondition<3, Registrars>,
      public detail::FlatnessImpl {
 private:
  using Base = elliptic::BoundaryConditions::BoundaryCondition<3, Registrars>;

 public:
  Flatness() = default;
  Flatness(const Flatness&) noexcept = default;
  Flatness& operator=(const Flatness&) noexcept = default;
  Flatness(Flatness&&) noexcept = default;
  Flatness& operator=(Flatness&&) noexcept = default;
  ~Flatness() noexcept = default;

  /// \cond
  explicit Flatness(CkMigrateMessage* m) noexcept : Base(m) {}
  using PUP::able::register_constructor;
  WRAPPED_PUPable_decl_template(Flatness);
  /// \endcond

  std::unique_ptr<domain::BoundaryConditions::BoundaryCondition> get_clone()
      const noexcept override {
    return std::make_unique<Flatness>(*this);
  }
};

/// \cond
template <typename Registrars>
PUP::able::PUP_ID Flatness<Registrars>::my_PUP_ID = 0;  // NOLINT
/// \endcond

}  // namespace Xcts::BoundaryConditions
