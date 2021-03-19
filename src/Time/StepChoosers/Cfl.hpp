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
template <size_t Dim, typename Frame, typename StepChooserRegistrars>
class Cfl;

namespace Registrars {
template <size_t Dim, typename Frame>
struct Cfl {
  template <typename StepChooserRegistrars>
  using f = StepChoosers::Cfl<Dim, Frame, StepChooserRegistrars>;
};
}  // namespace Registrars

/// Suggests a step size based on the CFL stability criterion.
template <size_t Dim, typename Frame,
          typename StepChooserRegistrars =
              tmpl::list<Registrars::Cfl<Dim, Frame>>>
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

  using argument_tags = tmpl::list<domain::Tags::MinimumGridSpacing<Dim, Frame>,
                                   Tags::DataBox, Tags::TimeStepper<>>;

  template <typename Metavariables, typename DbTags>
  std::pair<double, bool> operator()(
      const double minimum_grid_spacing, const db::DataBox<DbTags>& box,
      const typename Metavariables::time_stepper_tag::type::element_type&
          time_stepper,
      const double last_step_magnitude,
      const Parallel::GlobalCache<Metavariables>& /*cache*/) const noexcept {
    using compute_largest_characteristic_speed =
        typename Metavariables::system::compute_largest_characteristic_speed;
    const double speed = db::apply<compute_largest_characteristic_speed>(box);
    const double time_stepper_stability_factor = time_stepper.stable_step();

    const double step_size = safety_factor_ * time_stepper_stability_factor *
                             minimum_grid_spacing / (speed * Dim);
    // Reject the step if the CFL condition is violated.
    return std::make_pair(step_size, last_step_magnitude <= step_size);
  }

  // NOLINTNEXTLINE(google-runtime-references)
  void pup(PUP::er& p) noexcept override { p | safety_factor_; }

 private:
  double safety_factor_ = std::numeric_limits<double>::signaling_NaN();
};

/// \cond
template <size_t Dim, typename Frame, typename StepChooserRegistrars>
PUP::able::PUP_ID Cfl<Dim, Frame, StepChooserRegistrars>::my_PUP_ID =
    0;  // NOLINT
/// \endcond
}  // namespace StepChoosers
