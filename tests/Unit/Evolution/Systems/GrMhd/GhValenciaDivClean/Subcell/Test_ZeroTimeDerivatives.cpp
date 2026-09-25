// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <optional>
#include <string>
#include <vector>

#include "DataStructures/DataBox/DataBox.hpp"
#include "DataStructures/DataBox/PrefixHelpers.hpp"
#include "DataStructures/DataBox/Prefixes.hpp"
#include "DataStructures/Variables.hpp"
#include "Domain/Creators/Sphere.hpp"
#include "Domain/Structure/Neighbors.hpp"
#include "Domain/Structure/OrientationMap.hpp"
#include "Domain/Tags.hpp"
#include "Evolution/DiscontinuousGalerkin/OnlyDgBlockIds.hpp"
#include "Evolution/Systems/GeneralizedHarmonic/System.hpp"
#include "Evolution/Systems/GrMhd/GhValenciaDivClean/Subcell/ZeroTimeDerivatives.hpp"
#include "Evolution/Systems/GrMhd/GhValenciaDivClean/System.hpp"
#include "Evolution/Systems/RadiationTransport/NoNeutrinos/System.hpp"

SPECTRE_TEST_CASE(
    "Unit.Evolution.Systems.GhValenciaDivClean.Subcell.ZeroTimeDerivatives",
    "[Unit][Evolution]") {
  using NeutrinoTransportSystem = RadiationTransport::NoNeutrinos::System;
  using System = grmhd::GhValenciaDivClean::System<NeutrinoTransportSystem>;
  using DtVarsTag = ::Tags::Variables<
      db::wrap_tags_in<::Tags::dt, typename System::variables_tag::tags_list>>;
  using DtVars = typename DtVarsTag::type;
  using MhdTags =
      typename grmhd::ValenciaDivClean::System::variables_tag::tags_list;
  const size_t num_points = 10;

  const domain::creators::Sphere sphere(
      10.0, 80.0, domain::creators::Sphere::InnerCube{0.0}, 3_st, 8_st, false);
  REQUIRE(sphere.create_domain().blocks().size() == 7);
  REQUIRE(sphere.create_domain().block_names().at(6) == "InnerCube");

  const std::vector<size_t> only_dg_block_ids =
      evolution::dg::compute_only_dg_block_ids(
          std::optional{std::vector<std::string>{"InnerCube"}}, sphere);

  // This shouldn't alter anything because we don't have a neighbor doing
  // DG-only
  const Element<3> element_in_dg_only{
      ElementId<3>{0, {}},
      {{Direction<3>::lower_xi(),
        Neighbors<3>{ElementId<3>{5, {}},
                     OrientationMap<3>::create_aligned()}}}};
  auto box_in_dg_only =
      db::create<tmpl::list<DtVarsTag, evolution::dg::Tags::OnlyDgBlockIds<3>,
                            domain::Tags::Element<3>>>(
          DtVars{num_points, 1.2345}, only_dg_block_ids, element_in_dg_only);
  db::mutate_apply<
      grmhd::GhValenciaDivClean::subcell::ZeroMhdTimeDerivatives<System>>(
      make_not_null(&box_in_dg_only));
  CHECK(db::get<DtVarsTag>(box_in_dg_only) == DtVars{num_points, 1.2345});

  // Check that we are mutating if our neighbor is locked to DG.
  const Element<3> element_neighboring_dg_only{
      ElementId<3>{0, {}},
      {{Direction<3>::lower_xi(),
        Neighbors<3>{ElementId<3>{6, {}},
                     OrientationMap<3>::create_aligned()}}}};
  auto box_neighboring_dg_only =
      db::create<tmpl::list<DtVarsTag, evolution::dg::Tags::OnlyDgBlockIds<3>,
                            domain::Tags::Element<3>>>(
          DtVars{num_points, 1.2345}, only_dg_block_ids,
          element_neighboring_dg_only);
  db::mutate_apply<
      grmhd::GhValenciaDivClean::subcell::ZeroMhdTimeDerivatives<System>>(
      make_not_null(&box_neighboring_dg_only));
  {
    DtVars expected{num_points, 1.2345};
    tmpl::for_each<MhdTags>([&expected]<class Tag>(tmpl::type_<Tag> /*meta*/) {
      auto& var = get<::Tags::dt<Tag>>(expected);
      for (size_t storage_index = 0; storage_index < var.size();
           ++storage_index) {
        var[storage_index] = 0.0;
      }
    });
    CHECK(db::get<DtVarsTag>(box_neighboring_dg_only) == expected);
  }

  // Check that if we are doing DG-only and have a neighboring doing DG-only
  // that we don't zero out data.
  const Element<3> element_neighboring_and_in_dg_only{
      ElementId<3>{6, {}},
      {{Direction<3>::lower_xi(),
        Neighbors<3>{ElementId<3>{6, {}},
                     OrientationMap<3>::create_aligned()}}}};
  auto box_neighboring_and_in_dg_only =
      db::create<tmpl::list<DtVarsTag, evolution::dg::Tags::OnlyDgBlockIds<3>,
                            domain::Tags::Element<3>>>(
          DtVars{num_points, 1.2345}, only_dg_block_ids,
          element_neighboring_and_in_dg_only);
  db::mutate_apply<
      grmhd::GhValenciaDivClean::subcell::ZeroMhdTimeDerivatives<System>>(
      make_not_null(&box_neighboring_and_in_dg_only));
  CHECK(db::get<DtVarsTag>(box_neighboring_and_in_dg_only) ==
        DtVars{num_points, 1.2345});
}
