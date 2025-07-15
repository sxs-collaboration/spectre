// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>
#include <memory>

#include "DataStructures/DataBox/Tag.hpp"
#include "Evolution/DgSubcell/Tags/SubcellSolver.hpp"
#include "Evolution/Systems/ForceFree/FiniteDifference/Reconstructor.hpp"
#include "Evolution/Systems/ForceFree/Tags.hpp"
#include "Options/String.hpp"
#include "Utilities/TMPL.hpp"

namespace ForceFree::fd {

// A tag list of variables used for FD reconstruction process. For the ForceFree
// evolution system, we use the whole set of evolved variables and the
// generalized current density TildeJ.
using tags_list_for_reconstruction =
    tmpl::list<ForceFree::Tags::TildeJ, ForceFree::Tags::TildeE,
               ForceFree::Tags::TildeB, ForceFree::Tags::TildePsi,
               ForceFree::Tags::TildePhi, ForceFree::Tags::TildeQ>;

namespace OptionTags {
/*!
 * \brief Holds the subcell reconstructor in the input file
 */
struct Reconstructor {
  using type = std::unique_ptr<fd::Reconstructor>;

  static constexpr Options::String help = {"The reconstruction scheme to use."};
  using group = evolution::dg::subcell::OptionTags::SubcellSolverGroup;
};
}  // namespace OptionTags

namespace Tags {
/*!
 * \brief Tag for the reconstructor
 */
struct Reconstructor : db::SimpleTag {
  using type = std::unique_ptr<fd::Reconstructor>;
  using option_tags = tmpl::list<OptionTags::Reconstructor>;

  static constexpr bool pass_metavariables = false;
  static type create_from_options(const type& reconstructor) {
    return reconstructor->get_clone();
  }
};
}  // namespace Tags
}  // namespace ForceFree::fd
