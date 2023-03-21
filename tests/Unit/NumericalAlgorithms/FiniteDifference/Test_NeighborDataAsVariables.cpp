// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <boost/functional/hash.hpp>
#include <cstddef>
#include <utility>

#include "DataStructures/DataBox/Tag.hpp"
#include "DataStructures/DataVector.hpp"
#include "DataStructures/FixedHashMap.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "DataStructures/Variables.hpp"
#include "Domain/Structure/Direction.hpp"
#include "Domain/Structure/ElementId.hpp"
#include "Domain/Structure/MaxNumberOfNeighbors.hpp"
#include "Evolution/DgSubcell/GhostData.hpp"
#include "NumericalAlgorithms/FiniteDifference/NeighborDataAsVariables.hpp"
#include "NumericalAlgorithms/Spectral/Mesh.hpp"
#include "NumericalAlgorithms/Spectral/Spectral.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/MakeArray.hpp"
#include "Utilities/TMPL.hpp"

namespace {
namespace Tags {
struct Scalar : db::SimpleTag {
  using type = ::Scalar<DataVector>;
};

template <size_t Dim>
struct Vector : db::SimpleTag {
  using type = tnsr::I<DataVector, Dim, Frame::Inertial>;
};
}  // namespace Tags

template <size_t Dim>
void test() {
  using Vars = Variables<tmpl::list<Tags::Scalar, Tags::Vector<Dim>>>;
  const size_t ghost_zone_size = 2;
  const Mesh<Dim> subcell_mesh{5, Spectral::Basis::FiniteDifference,
                               Spectral::Quadrature::CellCentered};
  const size_t neighbor_mesh_size =
      ghost_zone_size * subcell_mesh.extents().slice_away(0).product();
  FixedHashMap<maximum_number_of_neighbors(Dim),
               std::pair<Direction<Dim>, ElementId<Dim>>,
               evolution::dg::subcell::GhostData,
               boost::hash<std::pair<Direction<Dim>, ElementId<Dim>>>>
      neighbor_data{};
  for (size_t i = 0; i < Direction<Dim>::all_directions().size(); ++i) {
    neighbor_data[std::pair{gsl::at(Direction<Dim>::all_directions(), i),
                            ElementId<Dim>{i}}]
        .neighbor_ghost_data_for_reconstruction() =
        DataVector{Vars::number_of_independent_components * neighbor_mesh_size,
                   square(i + 1.0)};
  }
  FixedHashMap<maximum_number_of_neighbors(Dim),
               std::pair<Direction<Dim>, ElementId<Dim>>, Vars,
               boost::hash<std::pair<Direction<Dim>, ElementId<Dim>>>>
      neighbor_data_as_vars{};
  fd::neighbor_data_as_variables(make_not_null(&neighbor_data_as_vars),
                                 neighbor_data, ghost_zone_size, subcell_mesh);
  REQUIRE(neighbor_data_as_vars.size() == neighbor_data.size());
  for (const auto& [neighbor_id, vars] : neighbor_data_as_vars) {
    REQUIRE(neighbor_data.find(neighbor_id) != neighbor_data.end());
    CHECK(neighbor_data.at(neighbor_id)
              .neighbor_ghost_data_for_reconstruction()
              .data() == vars.data());
    CHECK(vars.number_of_grid_points() == neighbor_mesh_size);
    CHECK(vars.size() ==
          Vars::number_of_independent_components * neighbor_mesh_size);
  }
#ifdef SPECTRE_DEBUG
  if constexpr (Dim > 1) {
    auto extents = make_array<Dim>(subcell_mesh.extents(0));
    ++extents[0];
    Mesh<Dim> non_istropic_mesh{extents, subcell_mesh.basis(),
                                subcell_mesh.quadrature()};
    CHECK_THROWS_WITH(
        fd::neighbor_data_as_variables(make_not_null(&neighbor_data_as_vars),
                                       neighbor_data, ghost_zone_size,
                                       non_istropic_mesh),
        Catch::Matchers::Contains("subcell_mesh must be isotropic but got"));
  }
#endif  // SPECTRE_DEBUG
}
}  // namespace

SPECTRE_TEST_CASE("Unit.FiniteDifference.NeighborDataAsVariables",
                  "[Unit][NumericalAlgorithms]") {
  test<1>();
  test<2>();
  test<3>();
}
