// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <array>
#include <cstddef>
#include <memory>
#include <string>
#include <vector>

#include "DataStructures/Index.hpp"
#include "Domain/Creators/AlignedLattice.hpp"
#include "Domain/Creators/DomainCreator.hpp"
#include "Domain/Structure/ElementId.hpp"
#include "Evolution/DgSubcell/ReconstructionMethod.hpp"
#include "Evolution/DgSubcell/SubcellEqualRateRegion.hpp"
#include "Evolution/DgSubcell/SubcellOptions.hpp"
#include "Evolution/DiscontinuousGalerkin/EqualRateLts/EqualRateRegionGenerator.hpp"
#include "NumericalAlgorithms/FiniteDifference/DerivativeOrder.hpp"
#include "Utilities/ConstantExpressions.hpp"
#include "Utilities/Literals.hpp"
#include "Utilities/MakeArray.hpp"
#include "Utilities/MakeString.hpp"
#include "Utilities/MakeVector.hpp"
#include "Utilities/Serialization/Serialize.hpp"

namespace {
static_assert(evolution::dg::equal_rate_region_generator<
              evolution::dg::subcell::SubcellEqualRateRegion<1>, 1>);
static_assert(evolution::dg::equal_rate_region_generator<
              evolution::dg::subcell::SubcellEqualRateRegion<2>, 2>);
static_assert(evolution::dg::equal_rate_region_generator<
              evolution::dg::subcell::SubcellEqualRateRegion<3>, 3>);

template <size_t Dim>
void test() {
  const std::unique_ptr<DomainCreator<Dim>> domain_creator =
      std::make_unique<domain::creators::AlignedLattice<Dim>>(
          make_array<Dim>(make_vector(0.0, 1.0, 2.0, 3.0)),
          std::array<std::vector<domain::CoordinateMaps::Distribution>, Dim>{},
          std::array<std::vector<double>, Dim>{}, make_array<Dim>(1_st),
          make_array<Dim>(6_st),
          std::vector<domain::creators::RefinementRegion<Dim>>{},
          std::vector<domain::creators::RefinementRegion<Dim>>{},
          std::vector<std::array<size_t, Dim>>{});

  const auto only_dg_blocks =
      make_vector<std::string>(MakeString{} << "Block" << Index<Dim>(0),
                               MakeString{} << "Block" << Index<Dim>(1));
  const size_t corner_block = 0;
  const size_t center_block = (pow<Dim>(3) - 1) / 2;

  const evolution::dg::subcell::SubcellOptions subcell_opts(
      4.0, 1, 2.0e-3, 2.0e-4, false, false,
      evolution::dg::subcell::fd::ReconstructionMethod::DimByDim, false,
      only_dg_blocks, fd::DerivativeOrder::Two, 1, 1, 1, 1);

  const auto check_regions =
      [&](const evolution::dg::subcell::SubcellEqualRateRegion<Dim>&
              subcell_regions) {
        {
          const auto regions = subcell_regions.regions();
          CHECK(regions.size() == 1);
          CHECK(regions.begin()->first == "Subcell");
          CHECK(regions.begin()->second == 0);
        }

        CHECK(
            not subcell_regions.is_in_region(0, ElementId<Dim>(corner_block)));
        CHECK(
            not subcell_regions.is_in_region(0, ElementId<Dim>(center_block)));

        for (size_t block = 0; block < pow<Dim>(3); ++block) {
          if (block == corner_block or block == center_block) {
            continue;
          }
          CHECK(subcell_regions.is_in_region(0, ElementId<Dim>(block)));
        }
      };

  const evolution::dg::subcell::SubcellEqualRateRegion<Dim> subcell_regions(
      subcell_opts, domain_creator);
  check_regions(subcell_regions);
  check_regions(serialize_and_deserialize(subcell_regions));
}

SPECTRE_TEST_CASE("Unit.Evolution.Subcell.SubcellEqualRateRegion",
                  "[Evolution][Unit]") {
  test<1>();
  test<2>();
  test<3>();
}
}  // namespace
