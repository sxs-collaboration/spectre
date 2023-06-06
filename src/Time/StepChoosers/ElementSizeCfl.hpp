// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>
#include <limits>
#include <pup.h>
#include <utility>

#include "DataStructures/DataBox/DataBox.hpp"
#include "Domain/SizeOfElement.hpp"
#include "Options/String.hpp"
#include "Time/StepChoosers/StepChooser.hpp"
#include "Time/Tags.hpp"
#include "Time/TimeSteppers/TimeStepper.hpp"
#include "Utilities/Serialization/CharmPupable.hpp"
#include "Utilities/TMPL.hpp"

/// \cond
namespace domain {
namespace Tags {
template <size_t Dim>
struct SizeOfElement;
}  // namespace Tags
}  // namespace domain
// IWYU pragma: no_forward_declare db::DataBox
/// \endcond

namespace StepChoosers {
/// Suggests a step size based on the CFL stability criterion, but uses the full
/// size of the element as the length scale in question.
///
/// This is useful as a coarse estimate for slabs, or to place a ceiling on
/// another dynamically-adjusted step chooser.
template <typename StepChooserUse, size_t Dim, typename System>
class ElementSizeCfl : public StepChooser<StepChooserUse> {
 public:
  /// \cond
  ElementSizeCfl() = default;
  explicit ElementSizeCfl(CkMigrateMessage* /*unused*/) {}
  using PUP::able::register_constructor;
  WRAPPED_PUPable_decl_template(ElementSizeCfl);  // NOLINT
  /// \endcond

  struct SafetyFactor {
    using type = double;
    static constexpr Options::String help{"Multiplier for computed step"};
    static type lower_bound() { return 0.0; }
  };

  static constexpr Options::String help{
      "Suggests a step size based on the CFL stability criterion, but in which "
      "the entire size of the element is used as the spacing in the "
      "computation. This is useful primarily for placing a ceiling on another "
      "dynamically-adjusted step chooser"};
  using options = tmpl::list<SafetyFactor>;

  explicit ElementSizeCfl(const double safety_factor)
      : safety_factor_(safety_factor) {}

  using argument_tags =
      tmpl::list<::Tags::TimeStepper<>, domain::Tags::SizeOfElement<Dim>,
                 typename System::compute_largest_characteristic_speed>;
  using compute_tags =
      tmpl::list<domain::Tags::SizeOfElementCompute<Dim>,
                 typename System::compute_largest_characteristic_speed>;

  std::pair<double, bool> operator()(
      const TimeStepper& time_stepper,
      const std::array<double, Dim>& element_size, const double speed,
      const double last_step_magnitude) const {
    double min_size_of_element = std::numeric_limits<double>::infinity();
    for (auto face_to_face_dimension : element_size) {
      if (face_to_face_dimension < min_size_of_element) {
        min_size_of_element = face_to_face_dimension;
      }
    }
    const double time_stepper_stability_factor = time_stepper.stable_step();
    const double step_size = safety_factor_ * time_stepper_stability_factor *
                             min_size_of_element / (speed * Dim);
    // Reject the step if the CFL condition is violated.
    return std::make_pair(step_size, last_step_magnitude <= step_size);
  }

  bool uses_local_data() const override { return true; }

  // NOLINTNEXTLINE(google-runtime-references)
  void pup(PUP::er& p) override { p | safety_factor_; }

 private:
  double safety_factor_ = std::numeric_limits<double>::signaling_NaN();
};

/// \cond
template <typename StepChooserUse, size_t Dim, typename System>
PUP::able::PUP_ID ElementSizeCfl<StepChooserUse, Dim, System>::my_PUP_ID =
    0;  // NOLINT
/// \endcond
}  // namespace StepChoosers
