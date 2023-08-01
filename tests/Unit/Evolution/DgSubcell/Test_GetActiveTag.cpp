// Distributed under the MIT License.
// See LICENSE.txt for details.


#include "Framework/TestingFramework.hpp"

#include "DataStructures/DataBox/DataBox.hpp"
#include "Domain/Tags.hpp"
#include "Evolution/DgSubcell/ActiveGrid.hpp"
#include "Evolution/DgSubcell/GetActiveTag.hpp"
#include "Evolution/DgSubcell/Tags/ActiveGrid.hpp"
#include "Evolution/DgSubcell/Tags/Coordinates.hpp"
#include "Utilities/Literals.hpp"

SPECTRE_TEST_CASE("Unit.Evolution.Subcell.GetActiveTag", "[Evolution][Unit]") {
  for (const auto active_grid : {evolution::dg::subcell::ActiveGrid::Dg,
                                 evolution::dg::subcell::ActiveGrid::Subcell}) {
    const auto box = db::create<db::AddSimpleTags<
        domain::Tags::Coordinates<1, Frame::Inertial>,
        evolution::dg::subcell::Tags::Coordinates<1, Frame::Inertial>,
        evolution::dg::subcell::Tags::ActiveGrid>>(
        tnsr::I<DataVector, 1>{10_st, 1.0}, tnsr::I<DataVector, 1>{20_st, 2.0},
        active_grid);
    if (active_grid == evolution::dg::subcell::ActiveGrid::Dg) {
      CHECK(evolution::dg::subcell::get_active_tag<
                domain::Tags::Coordinates<1, Frame::Inertial>,
                evolution::dg::subcell::Tags::Coordinates<1, Frame::Inertial>>(
                box) ==
            db::get<domain::Tags::Coordinates<1, Frame::Inertial>>(box));
      CHECK(evolution::dg::subcell::get_inactive_tag<
                domain::Tags::Coordinates<1, Frame::Inertial>,
                evolution::dg::subcell::Tags::Coordinates<1, Frame::Inertial>>(
                box) ==
            db::get<
                evolution::dg::subcell::Tags::Coordinates<1, Frame::Inertial>>(
                box));
    } else {
      CHECK(evolution::dg::subcell::get_active_tag<
                domain::Tags::Coordinates<1, Frame::Inertial>,
                evolution::dg::subcell::Tags::Coordinates<1, Frame::Inertial>>(
                box) ==
            db::get<
                evolution::dg::subcell::Tags::Coordinates<1, Frame::Inertial>>(
                box));
      CHECK(evolution::dg::subcell::get_inactive_tag<
                domain::Tags::Coordinates<1, Frame::Inertial>,
                evolution::dg::subcell::Tags::Coordinates<1, Frame::Inertial>>(
                box) ==
            db::get<domain::Tags::Coordinates<1, Frame::Inertial>>(box));
    }
  }
}
