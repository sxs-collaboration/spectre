// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <memory>

#include "DataStructures/DataBox/Tag.hpp"
#include "Evolution/DgSubcell/Tags/SubcellSolver.hpp"
#include "Evolution/Systems/Burgers/FiniteDifference/Reconstructor.hpp"
#include "Options/Options.hpp"

namespace Burgers::fd {
/// Option tags for reconstruction
namespace OptionTags {
/// \brief Option tag for the reconstructor
struct Reconstructor {
  using type = std::unique_ptr<fd::Reconstructor>;

  static constexpr Options::String help = {"The reconstruction scheme to use."};
  using group = evolution::dg::subcell::OptionTags::SubcellSolverGroup;
};
}  // namespace OptionTags

/// %Tags for reconstruction
namespace Tags {
/// \brief Tag for the reconstructor
struct Reconstructor : db::SimpleTag {
  using type = std::unique_ptr<fd::Reconstructor>;
  using option_tags = tmpl::list<OptionTags::Reconstructor>;

  static constexpr bool pass_metavariables = false;
  static type create_from_options(const type& reconstructor) noexcept {
    return reconstructor->get_clone();
  }
};
}  // namespace Tags
}  // namespace Burgers::fd
