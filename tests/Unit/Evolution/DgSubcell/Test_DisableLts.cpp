// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <cstddef>
#include <optional>
#include <string>
#include <utility>

#include "DataStructures/DataBox/DataBox.hpp"
#include "DataStructures/Index.hpp"
#include "Domain/CreateInitialElement.hpp"
#include "Domain/Creators/AlignedLattice.hpp"
#include "Domain/Structure/DirectionalIdMap.hpp"
#include "Domain/Structure/ElementId.hpp"
#include "Domain/Structure/SegmentId.hpp"
#include "Domain/Tags.hpp"
#include "Evolution/DgSubcell/DisableLts.hpp"
#include "Evolution/DgSubcell/ReconstructionMethod.hpp"
#include "Evolution/DgSubcell/SubcellOptions.hpp"
#include "Evolution/DgSubcell/Tags/SubcellOptions.hpp"
#include "Evolution/DiscontinuousGalerkin/MortarInfo.hpp"
#include "Evolution/DiscontinuousGalerkin/MortarTags.hpp"
#include "Evolution/DiscontinuousGalerkin/TimeSteppingPolicy.hpp"
#include "NumericalAlgorithms/FiniteDifference/DerivativeOrder.hpp"
#include "Time/Tags/FixedLtsRatio.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/Literals.hpp"
#include "Utilities/MakeArray.hpp"
#include "Utilities/MakeString.hpp"
#include "Utilities/MakeVector.hpp"

namespace {
template <size_t Dim>
void test(const size_t block, const size_t segment) {
  const bool dg_element = block == 0;

  auto segments = make_array<Dim>(SegmentId(1, 0));
  segments[0] = SegmentId(1, segment);
  const ElementId<Dim> id(block, segments);

  const domain::creators::AlignedLattice<Dim> domain_creator(
      make_array<Dim>(make_vector(0.0, 1.0, 2.0)), make_array<Dim>(1_st),
      make_array<Dim>(6_st), {}, {}, {});
  const auto domain = domain_creator.domain();
  const auto element = domain::create_initial_element(
      id, domain.blocks(), domain_creator.initial_refinement_levels());

  DirectionalIdMap<Dim, evolution::dg::MortarInfo<Dim>> initial_mortar_infos{};
  for (const auto& [direction, neighbors] : element.neighbors()) {
    for (const auto& neighbor : neighbors) {
      initial_mortar_infos[{direction, neighbor}].time_stepping_policy() =
          evolution::dg::TimeSteppingPolicy::Conservative;
    }
  }

  const size_t ratio = 64;

  // NOLINTNEXTLINE(misc-const-correctness)
  evolution::dg::subcell::SubcellOptions subcell_opts(
      evolution::dg::subcell::SubcellOptions(
          4.0, 1, 2.0e-3, 2.0e-4, false, false,
          evolution::dg::subcell::fd::ReconstructionMethod::DimByDim, false,
          make_vector<std::string>(MakeString{} << "Block" << Index<Dim>(0)),
          fd::DerivativeOrder::Two, 1, 1, 1, 1, ratio),
      domain_creator);

  auto box = db::create<db::AddSimpleTags<
      Tags::FixedLtsRatio, evolution::dg::Tags::MortarInfo<Dim>,
      domain::Tags::Element<Dim>,
      evolution::dg::subcell::Tags::SubcellOptions<Dim>>>(
      std::optional<size_t>{}, std::move(initial_mortar_infos), element,
      std::move(subcell_opts));

  db::mutate_apply<evolution::dg::subcell::DisableLts<Dim>>(
      make_not_null(&box));

  CHECK(db::get<Tags::FixedLtsRatio>(box) ==
        (dg_element ? std::nullopt : std::optional{ratio}));
  const auto& mortar_infos = db::get<evolution::dg::Tags::MortarInfo<Dim>>(box);
  CHECK(mortar_infos.size() == element.number_of_neighbors());
  for (const auto& [direction, neighbors] : element.neighbors()) {
    for (const auto& neighbor : neighbors) {
      if (dg_element) {
        CHECK(mortar_infos.at({direction, neighbor}).time_stepping_policy() ==
              evolution::dg::TimeSteppingPolicy::Conservative);
      } else {
        CHECK(mortar_infos.at({direction, neighbor}).time_stepping_policy() ==
              (neighbor.block_id() == 0
                   ? evolution::dg::TimeSteppingPolicy::Conservative
                   : evolution::dg::TimeSteppingPolicy::EqualRate));
      }
    }
  }
}

SPECTRE_TEST_CASE("Unit.Evolution.Subcell.DisableLts", "[Evolution][Unit]") {
  for (size_t block = 0; block < 2; ++block) {
    for (size_t segment = 0; segment < 2; ++segment) {
      test<1>(block, segment);
      test<2>(block, segment);
      test<3>(block, segment);
    }
  }
}
}  // namespace
