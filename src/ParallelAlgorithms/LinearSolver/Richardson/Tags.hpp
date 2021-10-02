// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <string>

#include "DataStructures/DataBox/Tag.hpp"
#include "Options/Options.hpp"
#include "Utilities/TMPL.hpp"

namespace LinearSolver::Richardson {

namespace OptionTags {

template <typename OptionsGroup>
struct RelaxationParameter {
  using type = double;
  using group = OptionsGroup;
  static constexpr Options::String help =
      "The weight for the residual in the scheme";
};

}  // namespace OptionTags

namespace Tags {

/// The Richardson relaxation parameter \f$\omega\f$
///
/// \see `LinearSolver::Richardson::Richardson`
template <typename OptionsGroup>
struct RelaxationParameter : db::SimpleTag {
  static std::string name() {
    return "RelaxationParameter(" + Options::name<OptionsGroup>() + ")";
  }
  using type = double;

  static constexpr bool pass_metavariables = false;
  using option_tags = tmpl::list<OptionTags::RelaxationParameter<OptionsGroup>>;
  static double create_from_options(const double value) { return value; }
};

}  // namespace Tags

}  // namespace LinearSolver::Richardson
