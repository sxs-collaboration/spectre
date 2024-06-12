// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <memory>
#include <pup.h>
#include <string>

#include "Domain/FunctionsOfTime/FunctionOfTime.hpp"
#include "Options/String.hpp"
#include "ParallelAlgorithms/EventsAndTriggers/Trigger.hpp"
#include "Utilities/Serialization/CharmPupable.hpp"
#include "Utilities/TMPL.hpp"

namespace domain::Tags {
template <size_t VolumeDim>
struct Domain;
struct FunctionsOfTime;
}  // namespace domain::Tags
namespace Tags {
struct Time;
}  // namespace Tags

namespace Triggers {
class FractionOfOrbit : public Trigger {
 public:
  /// \cond
  FractionOfOrbit() = default;
  explicit FractionOfOrbit(CkMigrateMessage* /*unused*/) {}
  using PUP::able::register_constructor;
  WRAPPED_PUPable_decl_template(FractionOfOrbit);  // NOLINT
  /// \endcond

  struct Value {
    using type = double;
    static constexpr Options::String help = {
        "Fraction of an orbit completed between triggers."};
  };

  using options = tmpl::list<Value>;
  static constexpr Options::String help{
      "Trigger when the evolution has reached a fraction of an orbit since the "
      "trigger was last triggered."};

  explicit FractionOfOrbit(double fraction_of_orbit);

  using argument_tags = tmpl::list<Tags::Time, domain::Tags::FunctionsOfTime>;

  bool operator()(const double time,
                  const std::unordered_map<
                      std::string,
                      std::unique_ptr<domain::FunctionsOfTime::FunctionOfTime>>&
                      functions_of_time);

  // NOLINENEXTLINE(google-runtime-references)
  void pup(PUP::er& p) override;

 private:
  double fraction_of_orbit_{};
  double last_trigger_time_ = 0.0;
};
}  // namespace Triggers
