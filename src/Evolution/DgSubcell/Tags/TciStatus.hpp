// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <boost/functional/hash.hpp>
#include <cstddef>

#include "DataStructures/DataBox/Tag.hpp"
#include "DataStructures/FixedHashMap.hpp"
#include "DataStructures/Tensor/TypeAliases.hpp"
#include "Domain/Structure/Direction.hpp"
#include "Domain/Structure/ElementId.hpp"
#include "Domain/Structure/MaxNumberOfNeighbors.hpp"
#include "Domain/Tags.hpp"
#include "Evolution/DgSubcell/ActiveGrid.hpp"
#include "Evolution/DgSubcell/Tags/ActiveGrid.hpp"
#include "Evolution/DgSubcell/Tags/Mesh.hpp"
#include "Utilities/Gsl.hpp"

/// \cond
class DataVector;
template <size_t Dim>
class Mesh;
/// \endcond

namespace evolution::dg::subcell::Tags {
/// Stores the status of the troubled cell indicator in the element as an `int`.
///
/// A non-zero value indicates the TCI decided the element is troubled.
struct TciDecision : db::SimpleTag {
  using type = int;
};

/// The TCI decision of neighboring elements.
template <size_t Dim>
struct NeighborTciDecisions : db::SimpleTag {
  using type =
      FixedHashMap<maximum_number_of_neighbors(Dim),
                   std::pair<Direction<Dim>, ElementId<Dim>>, int,
                   boost::hash<std::pair<Direction<Dim>, ElementId<Dim>>>>;
};

/// Stores the status of the troubled cell indicator in the element
/// (TciDecision) as a `Scalar<DataVector>` so it can be observed.
struct TciStatus : db::SimpleTag {
  using type = Scalar<DataVector>;
};

/// Compute tag to get a `TciStatus` from a `TciDecision`.
template <size_t Dim>
struct TciStatusCompute : db::ComputeTag, TciStatus {
  using base = TciStatus;
  using return_type = typename base::type;
  using argument_tags = tmpl::list<Tags::TciDecision, Tags::ActiveGrid,
                                   Tags::Mesh<Dim>, ::domain::Tags::Mesh<Dim>>;
  static void function(gsl::not_null<return_type*> result, int tci_decision,
                       subcell::ActiveGrid active_grid,
                       const ::Mesh<Dim>& subcell_mesh,
                       const ::Mesh<Dim>& dg_mesh);
};
}  // namespace evolution::dg::subcell::Tags
