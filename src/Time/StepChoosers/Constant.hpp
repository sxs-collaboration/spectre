// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <limits>
#include <utility>

#include "Options/Options.hpp"
#include "Options/String.hpp"
#include "Time/StepChoosers/StepChooser.hpp"
#include "Time/TimeStepRequest.hpp"
#include "Utilities/Serialization/CharmPupable.hpp"
#include "Utilities/TMPL.hpp"

/// \cond
namespace PUP {
class er;
}  // namespace PUP
/// \endcond

namespace StepChoosers {

/// Sets a constant goal.
class Constant : public StepChooser<StepChooserUse::Slab>,
                 public StepChooser<StepChooserUse::LtsStep> {
 public:
  /// \cond
  Constant() = default;
  explicit Constant(CkMigrateMessage* /*unused*/) {}
  using PUP::able::register_constructor;
  WRAPPED_PUPable_decl_template(Constant);  // NOLINT
  /// \endcond

  static constexpr Options::String help{"Sets a constant goal."};

  explicit Constant(double value);

  using argument_tags = tmpl::list<>;

  std::pair<TimeStepRequest, bool> operator()(double last_step) const;

  bool uses_local_data() const override;
  bool can_be_delayed() const override;

  // NOLINTNEXTLINE(google-runtime-references)
  void pup(PUP::er& p) override;

 private:
  double value_ = std::numeric_limits<double>::signaling_NaN();
};
}  // namespace StepChoosers

template <>
struct Options::create_from_yaml<StepChoosers::Constant> {
  template <typename Metavariables>
  static StepChoosers::Constant create(const Options::Option& options) {
    return create<void>(options);
  }
};

template <>
StepChoosers::Constant
Options::create_from_yaml<StepChoosers::Constant>::create<void>(
    const Options::Option& options);
