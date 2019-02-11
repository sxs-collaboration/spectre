// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <limits>
#include <pup.h>

#include "Domain/MinimumGridSpacing.hpp"
#include "Options/Options.hpp"
#include "Parallel/CharmPupable.hpp"
#include "Parallel/ConstGlobalCache.hpp"
#include "Time/StepChoosers/StepChooser.hpp"  // IWYU pragma: keep
#include "Time/Tags.hpp"
#include "Utilities/Registration.hpp"
#include "Utilities/TMPL.hpp"

/// \cond
namespace Parallel {
template <typename Metavariables>
class ConstGlobalCache;
}  // namespace Parallel
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
    static constexpr OptionString help{"Multiplier for computed step"};
    static type default_value() noexcept { return 1.0; }
    static type lower_bound() noexcept { return 0.0; }
  };

  static constexpr OptionString help{
      "Suggests a step size based on the CFL stability criterion."};
  using options = tmpl::list<SafetyFactor>;

  explicit Cfl(const double safety_factor) noexcept
      : safety_factor_(safety_factor) {}

  using argument_tags =
      tmpl::list<Tags::MinimumGridSpacing<Dim, Frame>, Tags::DataBox>;

  template <typename Metavariables, typename DbTags>
  double operator()(
      const double minimum_grid_spacing, const db::DataBox<DbTags>& box,
      const Parallel::ConstGlobalCache<Metavariables>& cache) const noexcept {
    using compute_largest_characteristic_speed =
        typename Metavariables::system::compute_largest_characteristic_speed;
    const double speed =
        db::apply<typename compute_largest_characteristic_speed::argument_tags>(
            compute_largest_characteristic_speed{}, box);
    const double time_stepper_stability_factor =
        Parallel::get<OptionTags::TimeStepper>(cache).stable_step();

    return safety_factor_ * time_stepper_stability_factor *
           minimum_grid_spacing / speed;
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
