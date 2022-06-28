// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <cstddef>
#include <memory>

#include "DataStructures/DataBox/DataBox.hpp"
#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/EagerMath/Magnitude.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Domain/Tags.hpp"
#include "Evolution/DgSubcell/Mesh.hpp"
#include "Evolution/DgSubcell/Reconstruction.hpp"
#include "Evolution/DgSubcell/Tags/Mesh.hpp"
#include "Evolution/Systems/GrMhd/ValenciaDivClean/Subcell/TciOnFdGrid.hpp"
#include "Evolution/Systems/GrMhd/ValenciaDivClean/Subcell/TciOptions.hpp"
#include "Evolution/Systems/GrMhd/ValenciaDivClean/Tags.hpp"
#include "NumericalAlgorithms/Spectral/Mesh.hpp"
#include "NumericalAlgorithms/Spectral/Spectral.hpp"
#include "Utilities/Gsl.hpp"

namespace {
enum class TestThis {
  AllGood,
  NeededFixing,
  PerssonTildeD,
  PerssonTildeTau,
  PerssonTildeB,
  NegativeTildeD,
  NegativeTildeTau,
  RdmpTildeD,
  RdmpTildeTau,
  RdmpMagnitudeTildeB
};

void test(const TestThis test_this) {
  const Mesh<3> mesh{6, Spectral::Basis::Legendre,
                     Spectral::Quadrature::GaussLobatto};
  const Mesh<3> subcell_mesh = evolution::dg::subcell::fd::mesh(mesh);
  const double persson_exponent = 5.0;
  const grmhd::ValenciaDivClean::subcell::TciOptions tci_options{
      1.0e-12, 1.0e-40, 1.0e-11, 1.0e-12,
      test_this == TestThis::PerssonTildeB ? std::optional<double>{1.0e-2}
                                           : std::nullopt};

  const evolution::dg::subcell::SubcellOptions subcell_options{
      1.0e-60,  // Tiny value because the magnetic field is so small
      1.0e-4,
      1.0e-60,  // Tiny value because the magnetic field is so small
      1.0e-4,
      persson_exponent,
      persson_exponent,
      false,
      evolution::dg::subcell::fd::ReconstructionMethod::DimByDim};

  auto box = db::create<db::AddSimpleTags<
      grmhd::ValenciaDivClean::Tags::TildeD,
      grmhd::ValenciaDivClean::Tags::TildeTau,
      grmhd::ValenciaDivClean::Tags::TildeB<>,
      grmhd::ValenciaDivClean::Tags::VariablesNeededFixing,
      domain::Tags::Mesh<3>, ::evolution::dg::subcell::Tags::Mesh<3>,
      grmhd::ValenciaDivClean::subcell::Tags::TciOptions,
      evolution::dg::subcell::Tags::SubcellOptions,
      evolution::dg::subcell::Tags::DataForRdmpTci>>(
      Scalar<DataVector>(subcell_mesh.number_of_grid_points(), 1.0),
      Scalar<DataVector>(subcell_mesh.number_of_grid_points(), 1.0),
      tnsr::I<DataVector, 3, Frame::Inertial>(
          subcell_mesh.number_of_grid_points(), 1.0),
      test_this == TestThis::NeededFixing, mesh, subcell_mesh, tci_options,
      subcell_options, evolution::dg::subcell::RdmpTciData{});

  const size_t point_to_change = mesh.number_of_grid_points() / 2;
  if (test_this == TestThis::PerssonTildeTau) {
    db::mutate<grmhd::ValenciaDivClean::Tags::TildeTau>(
        make_not_null(&box), [point_to_change](const auto tilde_tau_ptr) {
          get(*tilde_tau_ptr)[point_to_change] *= 2.0;
        });
  } else if (test_this == TestThis::PerssonTildeD) {
    db::mutate<grmhd::ValenciaDivClean::Tags::TildeD>(
        make_not_null(&box), [point_to_change](const auto tilde_d_ptr) {
          get(*tilde_d_ptr)[point_to_change] *= 2.0;
        });
  } else if (test_this == TestThis::PerssonTildeB) {
    db::mutate<grmhd::ValenciaDivClean::Tags::TildeB<>>(
        make_not_null(&box), [point_to_change](const auto tilde_b_ptr) {
          for (size_t i = 0; i < 3; ++i) {
            tilde_b_ptr->get(i)[point_to_change] *= 2.0;
          }
        });
  } else if (test_this == TestThis::NegativeTildeD) {
    db::mutate<grmhd::ValenciaDivClean::Tags::TildeD>(
        make_not_null(&box), [point_to_change](const auto tilde_d_ptr) {
          get(*tilde_d_ptr)[point_to_change] = -1.0e-20;
        });
  } else if (test_this == TestThis::NegativeTildeTau) {
    db::mutate<grmhd::ValenciaDivClean::Tags::TildeTau>(
        make_not_null(&box), [point_to_change](const auto tilde_tau_ptr) {
          get(*tilde_tau_ptr)[point_to_change] = -1.0e-20;
        });
  }

  // Set the RDMP TCI past data.
  using std::max;
  using std::min;
  evolution::dg::subcell::RdmpTciData past_rdmp_tci_data{};
  const auto magnitude_tilde_b =
      magnitude(db::get<grmhd::ValenciaDivClean::Tags::TildeB<>>(box));

  past_rdmp_tci_data.max_variables_values = std::vector<double>{
      max(max(get(db::get<grmhd::ValenciaDivClean::Tags::TildeD>(box))),
          max(evolution::dg::subcell::fd::reconstruct(
              get(db::get<grmhd::ValenciaDivClean::Tags::TildeD>(box)), mesh,
              subcell_mesh.extents(),
              evolution::dg::subcell::fd::ReconstructionMethod::DimByDim))),
      max(max(get(db::get<grmhd::ValenciaDivClean::Tags::TildeTau>(box))),
          max(evolution::dg::subcell::fd::reconstruct(
              get(db::get<grmhd::ValenciaDivClean::Tags::TildeTau>(box)), mesh,
              subcell_mesh.extents(),
              evolution::dg::subcell::fd::ReconstructionMethod::DimByDim))),
      max(max(get(magnitude_tilde_b)),
          max(evolution::dg::subcell::fd::reconstruct(
              get(magnitude_tilde_b), mesh, subcell_mesh.extents(),
              evolution::dg::subcell::fd::ReconstructionMethod::DimByDim)))};
  past_rdmp_tci_data.min_variables_values = std::vector<double>{
      min(min(get(db::get<grmhd::ValenciaDivClean::Tags::TildeD>(box))),
          min(evolution::dg::subcell::fd::reconstruct(
              get(db::get<grmhd::ValenciaDivClean::Tags::TildeD>(box)), mesh,
              subcell_mesh.extents(),
              evolution::dg::subcell::fd::ReconstructionMethod::DimByDim))),
      min(min(get(db::get<grmhd::ValenciaDivClean::Tags::TildeTau>(box))),
          min(evolution::dg::subcell::fd::reconstruct(
              get(db::get<grmhd::ValenciaDivClean::Tags::TildeTau>(box)), mesh,
              subcell_mesh.extents(),
              evolution::dg::subcell::fd::ReconstructionMethod::DimByDim))),
      min(min(get(magnitude_tilde_b)),
          min(evolution::dg::subcell::fd::reconstruct(
              get(magnitude_tilde_b), mesh, subcell_mesh.extents(),
              evolution::dg::subcell::fd::ReconstructionMethod::DimByDim)))};

  evolution::dg::subcell::RdmpTciData expected_rdmp_tci_data{};
  expected_rdmp_tci_data.max_variables_values = std::vector<double>{
      max(get(db::get<grmhd::ValenciaDivClean::Tags::TildeD>(box))),
      max(get(db::get<grmhd::ValenciaDivClean::Tags::TildeTau>(box))),
      max(get(magnitude_tilde_b))};
  expected_rdmp_tci_data.min_variables_values = std::vector<double>{
      min(get(db::get<grmhd::ValenciaDivClean::Tags::TildeD>(box))),
      min(get(db::get<grmhd::ValenciaDivClean::Tags::TildeTau>(box))),
      min(get(magnitude_tilde_b))};

  // Modify past data if we are expected an RDMP TCI failure.
  db::mutate<evolution::dg::subcell::Tags::DataForRdmpTci>(
      make_not_null(&box),
      [&past_rdmp_tci_data, &test_this](const auto rdmp_tci_data_ptr) {
        *rdmp_tci_data_ptr = past_rdmp_tci_data;
        if (test_this == TestThis::RdmpTildeD) {
          // Assumes min is positive, increase it so we fail the TCI
          rdmp_tci_data_ptr->min_variables_values[0] *= 1.01;
        } else if (test_this == TestThis::RdmpTildeTau) {
          // Assumes min is positive, increase it so we fail the TCI
          rdmp_tci_data_ptr->min_variables_values[1] *= 1.01;
        } else if (test_this == TestThis::RdmpMagnitudeTildeB) {
          // Assumes min is positive, increase it so we fail the TCI
          rdmp_tci_data_ptr->min_variables_values[2] *= 1.01;
        }
      });

  const std::tuple<bool, evolution::dg::subcell::RdmpTciData> result =
      db::mutate_apply<grmhd::ValenciaDivClean::subcell::TciOnFdGrid>(
          make_not_null(&box), persson_exponent);
  CHECK(get<1>(result) == expected_rdmp_tci_data);

  if (test_this == TestThis::AllGood) {
    CHECK_FALSE(get<0>(result));
  } else {
    CHECK(get<0>(result));
  }
}
}  // namespace

SPECTRE_TEST_CASE("Unit.Evolution.Systems.ValenciaDivClean.Subcell.TciOnFdGrid",
                  "[Unit][Evolution]") {
  for (const TestThis& test_this :
       {TestThis::AllGood, TestThis::NeededFixing, TestThis::PerssonTildeD,
        TestThis::PerssonTildeTau, TestThis::PerssonTildeB,
        TestThis::NegativeTildeD, TestThis::NegativeTildeTau,
        TestThis::RdmpTildeD, TestThis::RdmpTildeTau,
        TestThis::RdmpMagnitudeTildeB}) {
    test(test_this);
  }
}
