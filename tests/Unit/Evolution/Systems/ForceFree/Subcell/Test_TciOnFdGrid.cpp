// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <cstddef>
#include <limits>
#include <memory>

#include "DataStructures/DataBox/DataBox.hpp"
#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/EagerMath/Magnitude.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "DataStructures/Variables.hpp"
#include "DataStructures/VariablesTag.hpp"
#include "Domain/Tags.hpp"
#include "Evolution/DgSubcell/Mesh.hpp"
#include "Evolution/DgSubcell/Reconstruction.hpp"
#include "Evolution/DgSubcell/SubcellOptions.hpp"
#include "Evolution/DgSubcell/Tags/Mesh.hpp"
#include "Evolution/DgSubcell/Tags/SubcellOptions.hpp"
#include "Evolution/Systems/ForceFree/Subcell/TciOnFdGrid.hpp"
#include "Evolution/Systems/ForceFree/System.hpp"
#include "Evolution/Systems/ForceFree/Tags.hpp"
#include "NumericalAlgorithms/Spectral/Mesh.hpp"
#include "NumericalAlgorithms/Spectral/Spectral.hpp"
#include "PointwiseFunctions/GeneralRelativity/Tags.hpp"
#include "Utilities/Gsl.hpp"

namespace {
enum class TestThis {
  AllGood,
  PerssonMagTildeE,
  PerssonMagTildeB,
  PerssonTildeQ,
  RdmpMagTildeE,
  RdmpMagTildeB,
  RdmpTildeQ
};

void test(const TestThis test_this, const int expected_tci_status) {
  CAPTURE(test_this);
  CAPTURE(expected_tci_status);

  const Mesh<3> dg_mesh{5, Spectral::Basis::Legendre,
                        Spectral::Quadrature::GaussLobatto};
  const Mesh<3> subcell_mesh = evolution::dg::subcell::fd::mesh(dg_mesh);

  using TildeE = ForceFree::Tags::TildeE;
  using TildeB = ForceFree::Tags::TildeB;
  using TildeQ = ForceFree::Tags::TildeQ;

  using VarsForTciTest = Variables<tmpl::list<TildeE, TildeB, TildeQ>>;
  VarsForTciTest subcell_vars{subcell_mesh.number_of_grid_points(), 0.0};

  // set variables on the dg mesh for the test
  get(get<TildeQ>(subcell_vars)) = 1.0;
  for (size_t i = 0; i < 3; ++i) {
    get<TildeE>(subcell_vars).get(i) = 2.0;
    get<TildeB>(subcell_vars).get(i) = 3.0;
  }

  const double persson_exponent = 5.0;

  const evolution::dg::subcell::SubcellOptions subcell_options{
      1.0e-10,
      1.0e-10,
      1.0e-10,
      1.0e-10,
      persson_exponent,
      persson_exponent,
      false,
      evolution::dg::subcell::fd::ReconstructionMethod::DimByDim,
      false,
      std::nullopt,
      fd::DerivativeOrder::Two};

  auto box = db::create<db::AddSimpleTags<
      ::Tags::Variables<VarsForTciTest::tags_list>, ::domain::Tags::Mesh<3>,
      ::evolution::dg::subcell::Tags::Mesh<3>,
      evolution::dg::subcell::Tags::SubcellOptions<3>,
      evolution::dg::subcell::Tags::DataForRdmpTci>>(
      subcell_vars, dg_mesh, subcell_mesh, subcell_options,
      evolution::dg::subcell::RdmpTciData{});

  const size_t point_to_change = subcell_mesh.number_of_grid_points() / 2;

  if (test_this == TestThis::PerssonMagTildeE) {
    db::mutate<TildeE>(
        [point_to_change](const auto tilde_e_ptr) {
          for (size_t i = 0; i < 3; ++i) {
            tilde_e_ptr->get(i)[point_to_change] *= 10.0;
          }
        },
        make_not_null(&box));
  } else if (test_this == TestThis::PerssonMagTildeB) {
    db::mutate<TildeB>(
        [point_to_change](const auto tilde_b_ptr) {
          for (size_t i = 0; i < 3; ++i) {
            tilde_b_ptr->get(i)[point_to_change] *= 10.0;
          }
        },
        make_not_null(&box));
  } else if (test_this == TestThis::PerssonTildeQ) {
    db::mutate<TildeQ>(
        [point_to_change](const auto tilde_q_ptr) {
          get(*tilde_q_ptr)[point_to_change] *= 10.0;
        },
        make_not_null(&box));
  }

  // Set the RDMP TCI past data.
  using std::max;
  using std::min;
  const auto DimByDim =
      evolution::dg::subcell::fd::ReconstructionMethod::DimByDim;

  const auto subcell_mag_tilde_e = get(magnitude(db::get<TildeE>(box)));
  const auto subcell_mag_tilde_b = get(magnitude(db::get<TildeB>(box)));
  const auto subcell_tilde_q = get(db::get<TildeQ>(box));
  const auto dg_mag_tilde_e = evolution::dg::subcell::fd::reconstruct(
      subcell_mag_tilde_e, dg_mesh, subcell_mesh.extents(), DimByDim);
  const auto dg_mag_tilde_b = evolution::dg::subcell::fd::reconstruct(
      subcell_mag_tilde_b, dg_mesh, subcell_mesh.extents(), DimByDim);
  const auto dg_tilde_q = evolution::dg::subcell::fd::reconstruct(
      subcell_tilde_q, dg_mesh, subcell_mesh.extents(), DimByDim);

  evolution::dg::subcell::RdmpTciData past_rdmp_tci_data{};
  past_rdmp_tci_data.max_variables_values =
      DataVector{max(max(dg_mag_tilde_e), max(subcell_mag_tilde_e)),
                 max(max(dg_mag_tilde_b), max(subcell_mag_tilde_b)),
                 max(max(dg_tilde_q), max(subcell_tilde_q))};
  past_rdmp_tci_data.min_variables_values =
      DataVector{min(min(dg_mag_tilde_e), min(subcell_mag_tilde_e)),
                 min(min(dg_mag_tilde_b), min(subcell_mag_tilde_b)),
                 min(min(dg_tilde_q), min(subcell_tilde_q))};

  // Note : RDMP TCI data consists of max and min of subcell variables only when
  // using FD grid
  evolution::dg::subcell::RdmpTciData expected_rdmp_tci_data{};
  expected_rdmp_tci_data.max_variables_values = DataVector{
      max(subcell_mag_tilde_e), max(subcell_mag_tilde_b), max(subcell_tilde_q)};
  expected_rdmp_tci_data.min_variables_values = DataVector{
      min(subcell_mag_tilde_e), min(subcell_mag_tilde_b), min(subcell_tilde_q)};

  // Modify past data if we are expecting an RDMP TCI failure.
  db::mutate<evolution::dg::subcell::Tags::DataForRdmpTci>(
      [&past_rdmp_tci_data, &test_this](const auto rdmp_tci_data_ptr) {
        *rdmp_tci_data_ptr = past_rdmp_tci_data;
        // Assumes min is positive, increase it so we fail the TCI
        if (test_this == TestThis::RdmpMagTildeE) {
          rdmp_tci_data_ptr->min_variables_values[0] *= 1.01;
        } else if (test_this == TestThis::RdmpMagTildeB) {
          rdmp_tci_data_ptr->min_variables_values[1] *= 1.01;
        } else if (test_this == TestThis::RdmpTildeQ) {
          rdmp_tci_data_ptr->min_variables_values[2] *= 1.01;
        }
      },
      make_not_null(&box));

  const std::tuple<int, evolution::dg::subcell::RdmpTciData> result =
      db::mutate_apply<ForceFree::subcell::TciOnFdGrid>(
          make_not_null(&box), persson_exponent, false);

  CHECK(get<1>(result) == expected_rdmp_tci_data);

  if (test_this == TestThis::AllGood) {
    CHECK_FALSE(get<0>(result));
  } else {
    CHECK(get<0>(result) == expected_tci_status);
  }
}
}  // namespace

SPECTRE_TEST_CASE("Unit.Evolution.ForceFree.Subcell.TciOnFdGrid",
                  "[Unit][Evolution]") {
  test(TestThis::AllGood, 0);
  test(TestThis::PerssonMagTildeE, 1);
  test(TestThis::PerssonMagTildeB, 2);
  test(TestThis::PerssonTildeQ, 3);
  test(TestThis::RdmpMagTildeE, 4);
  test(TestThis::RdmpMagTildeB, 5);
  test(TestThis::RdmpTildeQ, 6);
}
