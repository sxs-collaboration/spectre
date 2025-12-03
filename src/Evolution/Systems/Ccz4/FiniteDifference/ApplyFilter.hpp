// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include "DataStructures/DataBox/Protocols/Mutator.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Domain/Structure/DirectionalIdMap.hpp"
#include "Evolution/DgSubcell/GhostData.hpp"
#include "Evolution/DgSubcell/Mesh.hpp"
#include "Evolution/DgSubcell/Tags/GhostDataForReconstruction.hpp"
#include "Evolution/DgSubcell/Tags/Mesh.hpp"
#include "Evolution/Systems/Ccz4/FiniteDifference/System.hpp"
#include "Evolution/Systems/Ccz4/FiniteDifference/Tags.hpp"
#include "Evolution/Systems/Ccz4/Tags.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/ProtocolHelpers.hpp"
#include "Utilities/TMPL.hpp"

namespace Ccz4::fd {
/*!
 * \brief Apply the Kreiss-Oliger filter to the evolved variables
 *
 */
struct ApplyFilter : tt::ConformsTo<db::protocols::Mutator> {
  using return_tags = tmpl::list<System::variables_tag>;
  using argument_tags =
      tmpl::list<evolution::dg::subcell::Tags::Mesh<3>,
                 Tags::EvolveLapseAndShift, Tags::KreissOligerEpsilon,
                 evolution::dg::subcell::Tags::GhostDataForReconstruction<3>>;

  static void apply(
      gsl::not_null<typename System::variables_tag::type*>
          evolved_vars_ptr,
      const Mesh<3>& subcell_mesh, bool evolve_lapse_and_shift,
      double kreiss_oliger_epsilon,
      const DirectionalIdMap<3, evolution::dg::subcell::GhostData>& ghost_data);
};
}  // namespace Ccz4::fd
