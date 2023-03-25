// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <algorithm>
#include <cstddef>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Evolution/DgSubcell/Mesh.hpp"
#include "Evolution/DgSubcell/Reconstruction.hpp"
#include "Evolution/Systems/Burgers/Subcell/TciOnFdGrid.hpp"
#include "NumericalAlgorithms/Spectral/Mesh.hpp"
#include "NumericalAlgorithms/Spectral/Spectral.hpp"

SPECTRE_TEST_CASE("Unit.Evolution.Systems.Burgers.Subcell.TciOnFdGrid",
                  "[Unit][Evolution]") {
  enum class TestThis { AllGood, PerssonU, RdmpU };

  for (const auto test_this :
       {TestThis::AllGood, TestThis::PerssonU, TestThis::RdmpU}) {
    const Mesh<1> dg_mesh{5, Spectral::Basis::Legendre,
                          Spectral::Quadrature::GaussLobatto};
    const Mesh<1> subcell_mesh = evolution::dg::subcell::fd::mesh(dg_mesh);
    const size_t number_of_points{subcell_mesh.number_of_grid_points()};

    Scalar<DataVector> u{number_of_points, 1.0};

    if (test_this == TestThis::PerssonU) {
      // make a troubled cell
      get(u)[number_of_points / 2] += 1.0;
    }

    // Set the RDMP TCI past data.
    using std::max;
    using std::min;
    evolution::dg::subcell::RdmpTciData past_rdmp_tci_data{
        {max(max(get(u)),
             max(evolution::dg::subcell::fd::reconstruct(
                 get(u), dg_mesh, subcell_mesh.extents(),
                 evolution::dg::subcell::fd::ReconstructionMethod::DimByDim)))},
        {min(
            min(get(u)),
            min(evolution::dg::subcell::fd::reconstruct(
                get(u), dg_mesh, subcell_mesh.extents(),
                evolution::dg::subcell::fd::ReconstructionMethod::DimByDim)))}};

    const evolution::dg::subcell::RdmpTciData expected_rdmp_tci_data{
        {max(get(u))}, {min(get(u))}};

    if (test_this == TestThis::RdmpU) {
      // Assumes min is positive, increase it so we fail the TCI
      past_rdmp_tci_data.min_variables_values[0] *= 1.01;
    }

    // check the result
    const double persson_exponent{4.0};
    const evolution::dg::subcell::SubcellOptions subcell_options{
        1.0e-16,
        1.0e-4,
        1.0e-16,
        1.0e-4,
        persson_exponent,
        persson_exponent,
        false,
        evolution::dg::subcell::fd::ReconstructionMethod::DimByDim,
        false,
        std::nullopt,
        fd::DerivativeOrder::Two};
    const std::tuple<bool, evolution::dg::subcell::RdmpTciData> result =
        Burgers::subcell::TciOnFdGrid::apply(
            u, dg_mesh, subcell_mesh, past_rdmp_tci_data, subcell_options,
            persson_exponent, false);
    CHECK(get<1>(result) == expected_rdmp_tci_data);
    if (test_this == TestThis::AllGood) {
      CHECK_FALSE(std::get<0>(result));
    } else {
      CHECK(std::get<0>(result));
    }
  }
}
