// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>
#include <limits>
#include <pup.h>
#include <utility>

#include "DataStructures/DataBox/DataBox.hpp"
#include "Options/Options.hpp"
#include "Parallel/CharmPupable.hpp"
#include "Time/StepChoosers/StepChooser.hpp"  // IWYU pragma: keep
#include "Time/Tags.hpp"
#include "Utilities/TMPL.hpp"

/// \cond
namespace Parallel {
template <typename Metavariables>
class GlobalCache;
}  // namespace Parallel
namespace domain {
namespace Tags {
template <size_t Dim, typename Frame>
struct MinimumGridSpacing;
}  // namespace Tags
}  // namespace domain
// IWYU pragma: no_forward_declare db::DataBox
/// \endcond

namespace StepChoosers {
template <typename Frame, typename System, typename StepChooserRegistrars>
class Cfl;

namespace Registrars {
template <typename Frame, typename System>
using Cfl = Registration::Registrar<StepChoosers::Cfl, Frame, System>;
}  // namespace Registrars

/// Suggests a step size based on the CFL stability criterion.
template <typename Frame, typename System,
          typename StepChooserRegistrars =
              tmpl::list<Registrars::Cfl<Frame, System>>>
class Cfl : public StepChooser<StepChooserRegistrars> {
 public:
  /// \cond
  Cfl() = default;
  explicit Cfl(CkMigrateMessage* /*unused*/) noexcept {}
  using PUP::able::register_constructor;
  WRAPPED_PUPable_decl_template(Cfl);  // NOLINT
  /// \endcond

  struct SafetyFactor {
    using type = double;
    static constexpr Options::String help{"Multiplier for computed step"};
    static type lower_bound() noexcept { return 0.0; }
  };

  static constexpr Options::String help{
      "Suggests a step size based on the CFL stability criterion."};
  using options = tmpl::list<SafetyFactor>;

  explicit Cfl(const double safety_factor) noexcept
      : safety_factor_(safety_factor) {}

  static constexpr UsableFor usable_for = UsableFor::AnyStepChoice;

  using argument_tags =
      tmpl::list<domain::Tags::MinimumGridSpacing<System::volume_dim, Frame>,
                 ::Tags::TimeStepper<>,
                 typename System::compute_largest_characteristic_speed>;
  using return_tags = tmpl::list<>;

  using compute_tags = tmpl::list<
      domain::Tags::MinimumGridSpacingCompute<System::volume_dim, Frame>,
      typename System::compute_largest_characteristic_speed>;

  template <typename Metavariables>
  std::pair<double, bool> operator()(
      const double minimum_grid_spacing,
      const typename Metavariables::time_stepper_tag::type::element_type&
          time_stepper,
      const double speed, const double last_step_magnitude,
      const Parallel::GlobalCache<Metavariables>& /*cache*/) const noexcept {
    const double time_stepper_stability_factor = time_stepper.stable_step();
    const double step_size = safety_factor_ * time_stepper_stability_factor *
                             minimum_grid_spacing /
                             (speed * System::volume_dim);
    // Reject the step if the CFL condition is violated.
    return std::make_pair(step_size, last_step_magnitude <= step_size);
  }

  // NOLINTNEXTLINE(google-runtime-references)
  void pup(PUP::er& p) noexcept override { p | safety_factor_; }

 private:
  double safety_factor_ = std::numeric_limits<double>::signaling_NaN();
};

/// \cond
template <typename Frame, typename System, typename StepChooserRegistrars>
PUP::able::PUP_ID Cfl<Frame, System, StepChooserRegistrars>::my_PUP_ID =
    0;  // NOLINT
/// \endcond
}  // namespace StepChoosers
