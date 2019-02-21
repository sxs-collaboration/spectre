// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <limits>
#include <pup.h>

#include "ErrorHandling/Assert.hpp"
#include "Options/Options.hpp"
#include "Parallel/CharmPupable.hpp"
#include "Time/StepChoosers/StepChooser.hpp"  // IWYU pragma: keep
#include "Utilities/Registration.hpp"
#include "Utilities/TMPL.hpp"

/// \cond
namespace Parallel {
template <typename Metavariables>
class ConstGlobalCache;
}  // namespace Parallel
/// \endcond

namespace StepChoosers {
template <typename StepChooserRegistrars>
class Constant;

namespace Registrars {
using Constant = Registration::Registrar<StepChoosers::Constant>;
}  // namespace Registrars

/// Suggests a constant step size.
template <typename StepChooserRegistrars = tmpl::list<Registrars::Constant>>
class Constant : public StepChooser<StepChooserRegistrars> {
 public:
  /// \cond
  Constant() = default;
  explicit Constant(CkMigrateMessage* /*unused*/) noexcept {}
  using PUP::able::register_constructor;
  WRAPPED_PUPable_decl_template(Constant);  // NOLINT
  /// \endcond

  static constexpr OptionString help{"Suggests a constant step size."};

  explicit Constant(const double value) noexcept : value_(value) {
    ASSERT(value_ > 0., "Requested step magnitude should be positive.");
  }

  using argument_tags = tmpl::list<>;

  template <typename Metavariables>
  double operator()(const Parallel::ConstGlobalCache<Metavariables>& /*cache*/)
      const noexcept {
    return value_;
  }

  // NOLINTNEXTLINE(google-runtime-references)
  void pup(PUP::er& p) noexcept override { p | value_; }

 private:
  double value_ = std::numeric_limits<double>::signaling_NaN();
};

/// \cond
template <typename StepChooserRegistrars>
PUP::able::PUP_ID Constant<StepChooserRegistrars>::my_PUP_ID = 0;  // NOLINT
/// \endcond

namespace Constant_detail {
double parse_options(const Option& options);
}  // namespace Constant_detail
}  // namespace StepChoosers

template <typename StepChooserRegistrars>
struct create_from_yaml<StepChoosers::Constant<StepChooserRegistrars>> {
  template <typename Metavariables>
  static StepChoosers::Constant<StepChooserRegistrars> create(
      const Option& options) {
    return StepChoosers::Constant<StepChooserRegistrars>(
        StepChoosers::Constant_detail::parse_options(options));
  }
};
