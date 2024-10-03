// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>
#include <utility>
#include <vector>

#include "Options/String.hpp"
#include "Time/StepChoosers/StepChooser.hpp"
#include "Time/TimeStepRequest.hpp"
#include "Utilities/Serialization/CharmPupable.hpp"
#include "Utilities/TMPL.hpp"

/// \cond
template <size_t VolumeDim>
class Element;
namespace domain {
namespace Tags {
template <size_t VolumeDim>
struct Element;
}  // namespace Tags
}  // namespace domain
namespace PUP {
class er;
}  // namespace PUP
/// \endcond

namespace StepChoosers {
/// Sets a goal specified per-block.
///
/// \note This debugging StepChooser is not included in the
/// `standard_step_choosers` list, but can be added to the
/// `factory_creation` struct in the metavariables.
template <size_t Dim>
class ByBlock : public StepChooser<StepChooserUse::Slab>,
                public StepChooser<StepChooserUse::LtsStep> {
 public:
  /// \cond
  ByBlock() = default;
  explicit ByBlock(CkMigrateMessage* /*unused*/) {}
  using PUP::able::register_constructor;
  WRAPPED_PUPable_decl_template(ByBlock);  // NOLINT
  /// \endcond

  struct Sizes {
    using type = std::vector<double>;
    static constexpr Options::String help{
        "Step sizes, indexed by block number"};
  };

  static constexpr Options::String help{"Sets a goal specified per-block."};
  using options = tmpl::list<Sizes>;

  explicit ByBlock(std::vector<double> sizes);

  using argument_tags = tmpl::list<domain::Tags::Element<Dim>>;

  std::pair<TimeStepRequest, bool> operator()(const Element<Dim>& element,
                                              double last_step) const;

  bool uses_local_data() const override;
  bool can_be_delayed() const override;

  // NOLINTNEXTLINE(google-runtime-references)
  void pup(PUP::er& p) override;

 private:
  std::vector<double> sizes_;
};
}  // namespace StepChoosers
