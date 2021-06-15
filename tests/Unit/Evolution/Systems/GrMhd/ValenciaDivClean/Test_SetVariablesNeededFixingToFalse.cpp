// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include "DataStructures/DataBox/DataBox.hpp"
#include "Evolution/Systems/GrMhd/ValenciaDivClean/SetVariablesNeededFixingToFalse.hpp"
#include "Evolution/Systems/GrMhd/ValenciaDivClean/Tags.hpp"

SPECTRE_TEST_CASE(
    "Unit.Evolution.Systems.ValenciaDivClean.SetVariablesNeededFixingToFalse",
    "[Unit][Evolution]") {
  auto box = db::create<
      db::AddSimpleTags<grmhd::ValenciaDivClean::Tags::VariablesNeededFixing>>(
      true);
  db::mutate_apply<grmhd::ValenciaDivClean::SetVariablesNeededFixingToFalse>(
      make_not_null(&box));
  CHECK_FALSE(
      db::get<grmhd::ValenciaDivClean::Tags::VariablesNeededFixing>(box));
}
