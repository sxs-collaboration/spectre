// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>
#include <limits>
#include <pup.h>
#include <utility>

#include "Options/Context.hpp"
#include "Options/String.hpp"
#include "Time/StepChoosers/StepChooser.hpp"
#include "Time/TimeStepRequest.hpp"
#include "Utilities/Serialization/CharmPupable.hpp"
#include "Utilities/TMPL.hpp"

/// \cond
template <size_t VolumeDim>
class Element;
class TimeStepId;
namespace Tags {
struct TimeStepId;
}  // namespace Tags
namespace domain::Tags {
template <size_t VolumeDim>
struct Element;
}  // namespace domain::Tags
/// \endcond

namespace StepChoosers {
/// Changes the step size pseudo-randomly.  Values are distributed
/// uniformly in $\log(dt)$.  The current step is always accepted.
template <typename StepChooserUse, size_t VolumeDim>
class Random : public StepChooser<StepChooserUse> {
 public:
  /// \cond
  Random();
  explicit Random(CkMigrateMessage* /*unused*/);
  using PUP::able::register_constructor;
  WRAPPED_PUPable_decl_template(Random);  // NOLINT
  /// \endcond

  struct Minimum {
    using type = double;
    static constexpr Options::String help{"Minimum value to suggest"};
    static type lower_bound() { return 0.0; }
  };

  struct Maximum {
    using type = double;
    static constexpr Options::String help{"Maximum value to suggest"};
    static type lower_bound() { return 0.0; }
  };

  struct Seed {
    using type = size_t;
    static constexpr Options::String help{"RNG seed"};
  };

  static constexpr Options::String help =
      "Changes the step size pseudo-randomly.";
  using options = tmpl::list<Minimum, Maximum, Seed>;

  Random(double minimum, double maximum, size_t seed,
         const Options::Context& context = {});

  using argument_tags =
      tmpl::list<domain::Tags::Element<VolumeDim>, Tags::TimeStepId>;

  std::pair<TimeStepRequest, bool> operator()(const Element<VolumeDim>& element,
                                              const TimeStepId& time_step_id,
                                              double last_step) const;

  bool uses_local_data() const override;
  bool can_be_delayed() const override;

  // NOLINTNEXTLINE(google-runtime-references)
  void pup(PUP::er& p) override;

 private:
  double minimum_ = std::numeric_limits<double>::signaling_NaN();
  double maximum_ = std::numeric_limits<double>::signaling_NaN();
  size_t seed_ = 0;
};
}  // namespace StepChoosers
