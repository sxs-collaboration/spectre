// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <array>
#include <cstddef>
#include <memory>
#include <optional>
#include <unordered_set>
#include <utility>
#include <vector>

#include "DataStructures/DataBox/DataBox.hpp"
#include "Domain/Block.hpp"
#include "Domain/CoordinateMaps/CoordinateMap.hpp"
#include "Domain/CoordinateMaps/CoordinateMap.tpp"
#include "Domain/CoordinateMaps/Identity.hpp"
#include "Domain/CreateInitialElement.hpp"
#include "Domain/Creators/Tags/FunctionsOfTime.hpp"
#include "Domain/Structure/Element.hpp"
#include "Domain/Tags.hpp"
#include "Domain/TagsTimeDependent.hpp"
#include "Evolution/DgSubcell/Mesh.hpp"
#include "Evolution/DgSubcell/Tags/Coordinates.hpp"
#include "Evolution/DgSubcell/Tags/Jacobians.hpp"
#include "Evolution/DgSubcell/Tags/Mesh.hpp"
#include "NumericalAlgorithms/Spectral/LogicalCoordinates.hpp"
#include "NumericalAlgorithms/Spectral/Mesh.hpp"
#include "Time/Tags.hpp"
#include "Time/Time.hpp"
#include "Utilities/CloneUniquePtrs.hpp"

namespace grmhd::GhValenciaDivClean {
namespace {

void test() {
  const ElementId<3> element_id{
      0, {SegmentId{3, 4}, SegmentId{3, 4}, SegmentId{3, 7}}};

  Block<3> block{
      domain::make_coordinate_map_base<Frame::BlockLogical, Frame::Inertial>(
          domain::CoordinateMaps::Identity<3>{}),
      0,
      {}};

  std::unordered_map<std::string,
                     std::unique_ptr<domain::FunctionsOfTime::FunctionOfTime>>
      functions_of_time{};

  ElementMap<3, Frame::Grid> element_map{
      element_id, block.is_time_dependent()
                      ? block.moving_mesh_logical_to_grid_map().get_clone()
                      : block.stationary_map().get_to_grid_frame()};
  const double time = 0.0;
  const Mesh<3> subcell_mesh{2, Spectral::Basis::FiniteDifference,
                             Spectral::Quadrature::CellCentered};

  auto box = db::create<
      db::AddSimpleTags<evolution::dg::subcell::Tags::Mesh<3>,
                        domain::Tags::ElementMap<3, Frame::Grid>,
                        domain::CoordinateMaps::Tags::CoordinateMap<
                            3, Frame::Grid, Frame::Inertial>,
                        ::Tags::Time, domain::Tags::FunctionsOfTimeInitialize>,
      db::AddComputeTags<
          evolution::dg::subcell::Tags::LogicalCoordinatesCompute<3>,
          ::domain::Tags::MappedCoordinates<
              ::domain::Tags::ElementMap<3, Frame::Grid>,
              evolution::dg::subcell::Tags::Coordinates<3,
                                                        Frame::ElementLogical>,
              evolution::dg::subcell::Tags::Coordinates>,
          evolution::dg::subcell::fd::Tags::InverseJacobianLogicalToGridCompute<
              ::domain::Tags::ElementMap<3, Frame::Grid>, 3>,
          evolution::dg::subcell::fd::Tags::
              InverseJacobianLogicalToInertialCompute<
                  ::domain::CoordinateMaps::Tags::CoordinateMap<
                      3, Frame::Grid, Frame::Inertial>,
                  3>>>(
      subcell_mesh, std::move(element_map),
      domain::make_coordinate_map_base<Frame::Grid, Frame::Inertial>(
          domain::CoordinateMaps::Identity<3>{}),
      time, clone_unique_ptrs(functions_of_time));

  const auto& inv_jac_grid = db::get<
      evolution::dg::subcell::fd::Tags::InverseJacobianLogicalToGrid<3>>(box);
  const auto& inv_jac_inertial = db::get<
      evolution::dg::subcell::fd::Tags::InverseJacobianLogicalToInertial<3>>(
      box);

  // Check that the two jacobians in frames connected by an identity map are
  // identical
  for (size_t storage_index = 0; storage_index < inv_jac_inertial.size();
       ++storage_index) {
    CHECK_ITERABLE_APPROX(inv_jac_inertial[storage_index],
                          inv_jac_grid[storage_index]);
  }
}

SPECTRE_TEST_CASE("Unit.Evolution.DgSubcell.JacobianCompute",
                  "[Unit][Evolution]") {
  test();
}
}  // namespace
}  // namespace grmhd::GhValenciaDivClean
