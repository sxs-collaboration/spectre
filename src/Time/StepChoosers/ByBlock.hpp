// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>
#include <pup.h>
#include <pup_stl.h>  // IWYU pragma: keep
#include <utility>
#include <vector>

#include "ErrorHandling/Error.hpp"
#include "Options/Options.hpp"
#include "ParallelBackend/CharmPupable.hpp"
#include "Time/StepChoosers/StepChooser.hpp"  // IWYU pragma: keep
#include "Utilities/TMPL.hpp"

/// \cond
template <size_t VolumeDim>
class Element;
namespace Parallel {
template <typename Metavariables>
class ConstGlobalCache;
}  // namespace Parallel
namespace Tags {
template <size_t VolumeDim>
struct Element;
}  // namespace Tags
/// \endcond

namespace StepChoosers {
template <size_t Dim, typename StepChooserRegistrars>
class ByBlock;

namespace Registrars {
template <size_t Dim>
struct ByBlock {
  template <typename StepChooserRegistrars>
  using f = StepChoosers::ByBlock<Dim, StepChooserRegistrars>;
};
}  // namespace Registrars

/// Suggests specified step sizes in each block
template <size_t Dim,
          typename StepChooserRegistrars = tmpl::list<Registrars::ByBlock<Dim>>>
class ByBlock : public StepChooser<StepChooserRegistrars> {
 public:
  /// \cond
  ByBlock() = default;
  explicit ByBlock(CkMigrateMessage* /*unused*/) noexcept {}
  using PUP::able::register_constructor;
  WRAPPED_PUPable_decl_template(ByBlock);  // NOLINT
  /// \endcond

  struct Sizes {
    using type = std::vector<double>;
    static constexpr OptionString help{"Step sizes, indexed by block number"};
  };

  static constexpr OptionString help{
      "Suggests specified step sizes in each block"};
  using options = tmpl::list<Sizes>;

  explicit ByBlock(std::vector<double> sizes) noexcept
      : sizes_(std::move(sizes)) {}

  using argument_tags = tmpl::list<Tags::Element<Dim>>;

  template <typename Metavariables>
  double operator()(const Element<Dim>& element,
                    const Parallel::ConstGlobalCache<Metavariables>& /*cache*/)
      const noexcept {
    const size_t block = element.id().block_id();
    if (block >= sizes_.size()) {
      ERROR("Step size not specified for block " << block);
    }
    return sizes_[block];
  }

  // NOLINTNEXTLINE(google-runtime-references)
  void pup(PUP::er& p) noexcept override { p | sizes_; }

 private:
  std::vector<double> sizes_;
};

/// \cond
template <size_t Dim, typename StepChooserRegistrars>
PUP::able::PUP_ID ByBlock<Dim, StepChooserRegistrars>::my_PUP_ID = 0;  // NOLINT
/// \endcond
}  // namespace StepChoosers
