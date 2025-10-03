// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>
#include <string>

#include "DataStructures/DataBox/Tag.hpp"
#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/TypeAliases.hpp"
#include "Evolution/DgSubcell/Tags/SubcellOptions.hpp"
#include "Evolution/DgSubcell/Tags/SubcellSolver.hpp"
#include "Evolution/Systems/Ccz4/FiniteDifference/Reconstructor.hpp"
#include "Evolution/Systems/Ccz4/FiniteDifference/System.hpp"
#include "Evolution/Systems/Ccz4/Tags.hpp"
#include "Evolution/Systems/Ccz4/TagsDeclarations.hpp"
#include "Evolution/Tags.hpp"
#include "Options/String.hpp"
#include "PointwiseFunctions/GeneralRelativity/Tags.hpp"
#include "PointwiseFunctions/GeneralRelativity/Tags/Conformal.hpp"

namespace Ccz4::fd {
/// \brief Option tags for evolving SoCcz4 with finite difference
namespace OptionTags {
/// \brief Option tag for the reconstructor
struct Reconstructor {
  using type = std::unique_ptr<fd::Reconstructor>;

  static constexpr Options::String help = {"The reconstruction scheme to use."};
  using group = evolution::dg::subcell::OptionTags::SubcellSolverGroup;
};

/// \brief Option tag for whether to evolve the lapse and shift
struct EvolveLapseAndShift {
  using type = bool;

  static constexpr Options::String help = {
      "The option to use time-independent laspe and shift."};
  using group = ::Ccz4::OptionTags::Ccz4Group;
};

/// \brief Option tag for whether to use constrained evolution
///
/// When true, the determint of the conformal spatial metric is rescaled
/// to one and the trace of ATilde is removed using the rescaled metric
/// after every complete time step.
struct ConstrainedEvolution {
  using type = bool;

  static constexpr Options::String help = {
      "Whether to use constrained evolution."};
  using group = ::Ccz4::OptionTags::Ccz4Group;
};

/// \brief Option tag for the epsilon parameter of the Kreiss-Oliger dissipation
struct KreissOligerEpsilon {
  using type = double;

  static constexpr Options::String help = {
      "The epsilon parameter for Kreiss-Oliger dissipation."};
  using group = ::Ccz4::OptionTags::Ccz4Group;
};
}  // namespace OptionTags

/// \brief Tags for evolving SoCcz4 with finite difference
namespace Tags {
/// \brief Tag for the reconstructor
struct Reconstructor : db::SimpleTag {
  using type = std::unique_ptr<fd::Reconstructor>;
  using option_tags = tmpl::list<OptionTags::Reconstructor>;

  static constexpr bool pass_metavariables = false;
  static type create_from_options(const type& reconstructor) {
    return reconstructor->get_clone();
  }
};

/*!
 * \brief Tag for whether to evolve the lapse and shif
 */
struct EvolveLapseAndShift : db::SimpleTag {
  using type = bool;
  using option_tags = tmpl::list<OptionTags::EvolveLapseAndShift>;

  static constexpr bool pass_metavariables = false;
  static type create_from_options(const bool evolve_lapse_and_shift) {
    return evolve_lapse_and_shift;
  }
};

/*!
 * \brief Tag for whether to evolve the lapse and shift
 */
struct ConstrainedEvolution : db::SimpleTag {
  using type = bool;
  using option_tags = tmpl::list<OptionTags::ConstrainedEvolution>;

  static constexpr bool pass_metavariables = false;
  static type create_from_options(const bool constrained_evolution) {
    return constrained_evolution;
  }
};

/*!
 * \brief Tag for the epsilon parameter of the Kreiss-Oliger dissipation
 */
struct KreissOligerEpsilon : db::SimpleTag {
  using type = double;
  using option_tags = tmpl::list<OptionTags::KreissOligerEpsilon>;

  static constexpr bool pass_metavariables = false;
  static type create_from_options(const double kreiss_oliger_epsilon) {
    return kreiss_oliger_epsilon;
  }
};

/// \brief Tags sent for second-order Ccz4 evolution.
using spacetime_reconstruction_tags = System::variables_tag_list;
}  // namespace Tags
}  // namespace Ccz4::fd
