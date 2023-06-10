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
#include "PointwiseFunctions/GeneralRelativity/Tags.hpp"
#include "PointwiseFunctions/Hydro/Tags.hpp"
#include "Utilities/Gsl.hpp"

namespace {
enum class TestThis {
  AllGood,
  Atmosphere,
  NeededFixing,
  PerssonTildeD,
  PerssonTildeYe,
  PerssonPressure,
  PerssonTildeB,
  NegativeTildeD,
  NegativeTildeYe,
  NegativeTildeTau,
  RdmpTildeD,
  RdmpTildeYe,
  RdmpTildeTau,
  RdmpMagnitudeTildeB
};

void test(const TestThis test_this, const int expected_tci_status) {
  const Mesh<3> mesh{6, Spectral::Basis::Legendre,
                     Spectral::Quadrature::GaussLobatto};
  const Mesh<3> subcell_mesh = evolution::dg::subcell::fd::mesh(mesh);
  const double persson_exponent = 5.0;
  const grmhd::ValenciaDivClean::subcell::TciOptions tci_options{
      1.0e-12,
      1.e-3,
      1.0e-40,
      1.0e-11,
      1.0e-12,
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
      evolution::dg::subcell::fd::ReconstructionMethod::DimByDim,
      false,
      std::nullopt,
      fd::DerivativeOrder::Two};

  auto box = db::create<db::AddSimpleTags<
      grmhd::ValenciaDivClean::Tags::TildeD,
      grmhd::ValenciaDivClean::Tags::TildeYe,
      grmhd::ValenciaDivClean::Tags::TildeTau,
      grmhd::ValenciaDivClean::Tags::TildeB<>,
      hydro::Tags::RestMassDensity<DataVector>,
      hydro::Tags::Pressure<DataVector>,
      grmhd::ValenciaDivClean::Tags::VariablesNeededFixing,
      domain::Tags::Mesh<3>, ::evolution::dg::subcell::Tags::Mesh<3>,
      grmhd::ValenciaDivClean::subcell::Tags::TciOptions,
      evolution::dg::subcell::Tags::SubcellOptions<3>,
      evolution::dg::subcell::Tags::DataForRdmpTci>>(
      Scalar<DataVector>(subcell_mesh.number_of_grid_points(), 1.0),
      Scalar<DataVector>(subcell_mesh.number_of_grid_points(), 0.1),
      Scalar<DataVector>(subcell_mesh.number_of_grid_points(), 1.0),
      tnsr::I<DataVector, 3, Frame::Inertial>(
          subcell_mesh.number_of_grid_points(), 1.0),
      Scalar<DataVector>(subcell_mesh.number_of_grid_points(), 1.0),
      Scalar<DataVector>(subcell_mesh.number_of_grid_points(), 1.0),
      test_this == TestThis::NeededFixing, mesh, subcell_mesh, tci_options,
      subcell_options, evolution::dg::subcell::RdmpTciData{});

  const size_t point_to_change = mesh.number_of_grid_points() / 2;
  if (test_this == TestThis::PerssonPressure) {
    db::mutate<hydro::Tags::Pressure<DataVector>>(
        [point_to_change](const auto pressure_ptr) {
          get(*pressure_ptr)[point_to_change] *= 2.0;
        },
        make_not_null(&box));
  } else if (test_this == TestThis::PerssonTildeD) {
    db::mutate<grmhd::ValenciaDivClean::Tags::TildeD>(
        [point_to_change](const auto tilde_d_ptr) {
          get(*tilde_d_ptr)[point_to_change] *= 2.0;
        },
        make_not_null(&box));
  } else if (test_this == TestThis::PerssonTildeYe) {
    db::mutate<grmhd::ValenciaDivClean::Tags::TildeYe>(
        [point_to_change](const auto tilde_d_ptr) {
          get(*tilde_d_ptr)[point_to_change] *= 2.0;
        },
        make_not_null(&box));
  } else if (test_this == TestThis::PerssonTildeB) {
    db::mutate<grmhd::ValenciaDivClean::Tags::TildeB<>>(
        [point_to_change](const auto tilde_b_ptr) {
          for (size_t i = 0; i < 3; ++i) {
            tilde_b_ptr->get(i)[point_to_change] *= 2.0;
          }
        },
        make_not_null(&box));
  } else if (test_this == TestThis::NegativeTildeD) {
    db::mutate<grmhd::ValenciaDivClean::Tags::TildeD>(
        [point_to_change](const auto tilde_d_ptr) {
          get(*tilde_d_ptr)[point_to_change] = -1.0e-20;
        },
        make_not_null(&box));
  } else if (test_this == TestThis::NegativeTildeYe) {
    db::mutate<grmhd::ValenciaDivClean::Tags::TildeYe>(
        [point_to_change](const auto tilde_d_ptr) {
          get(*tilde_d_ptr)[point_to_change] = -1.0e-20;
        },
        make_not_null(&box));
  } else if (test_this == TestThis::NegativeTildeTau) {
    db::mutate<grmhd::ValenciaDivClean::Tags::TildeTau>(
        [point_to_change](const auto tilde_tau_ptr) {
          get(*tilde_tau_ptr)[point_to_change] = -1.0e-20;
        },
        make_not_null(&box));
  } else if (test_this == TestThis::Atmosphere) {
    db::mutate<hydro::Tags::RestMassDensity<DataVector>,
               grmhd::ValenciaDivClean::Tags::VariablesNeededFixing>(
        [](const auto rest_mass_density_ptr,
           const auto variables_needed_fixing_ptr) {
          *variables_needed_fixing_ptr = true;
          get(*rest_mass_density_ptr) =
              5.0e-12;  // smaller than atmosphere density but
                        // bigger than the Min(TildeD) TCI option
        },
        make_not_null(&box));
  }

  // Set the RDMP TCI past data.
  using std::max;
  using std::min;
  evolution::dg::subcell::RdmpTciData past_rdmp_tci_data{};
  const auto magnitude_tilde_b =
      magnitude(db::get<grmhd::ValenciaDivClean::Tags::TildeB<>>(box));

  past_rdmp_tci_data.max_variables_values = DataVector{
      max(max(get(db::get<grmhd::ValenciaDivClean::Tags::TildeD>(box))),
          max(evolution::dg::subcell::fd::reconstruct(
              get(db::get<grmhd::ValenciaDivClean::Tags::TildeD>(box)), mesh,
              subcell_mesh.extents(),
              evolution::dg::subcell::fd::ReconstructionMethod::DimByDim))),
      max(max(get(db::get<grmhd::ValenciaDivClean::Tags::TildeYe>(box))),
          max(evolution::dg::subcell::fd::reconstruct(
              get(db::get<grmhd::ValenciaDivClean::Tags::TildeYe>(box)), mesh,
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
  past_rdmp_tci_data.min_variables_values = DataVector{
      min(min(get(db::get<grmhd::ValenciaDivClean::Tags::TildeD>(box))),
          min(evolution::dg::subcell::fd::reconstruct(
              get(db::get<grmhd::ValenciaDivClean::Tags::TildeD>(box)), mesh,
              subcell_mesh.extents(),
              evolution::dg::subcell::fd::ReconstructionMethod::DimByDim))),
      min(min(get(db::get<grmhd::ValenciaDivClean::Tags::TildeYe>(box))),
          min(evolution::dg::subcell::fd::reconstruct(
              get(db::get<grmhd::ValenciaDivClean::Tags::TildeYe>(box)), mesh,
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
  expected_rdmp_tci_data.max_variables_values = DataVector{
      max(get(db::get<grmhd::ValenciaDivClean::Tags::TildeD>(box))),
      max(get(db::get<grmhd::ValenciaDivClean::Tags::TildeYe>(box))),
      max(get(db::get<grmhd::ValenciaDivClean::Tags::TildeTau>(box))),
      max(get(magnitude_tilde_b))};
  expected_rdmp_tci_data.min_variables_values = DataVector{
      min(get(db::get<grmhd::ValenciaDivClean::Tags::TildeD>(box))),
      min(get(db::get<grmhd::ValenciaDivClean::Tags::TildeYe>(box))),
      min(get(db::get<grmhd::ValenciaDivClean::Tags::TildeTau>(box))),
      min(get(magnitude_tilde_b))};

  // Modify past data if we are expected an RDMP TCI failure.
  db::mutate<evolution::dg::subcell::Tags::DataForRdmpTci>(
      [&past_rdmp_tci_data, &test_this](const auto rdmp_tci_data_ptr) {
        *rdmp_tci_data_ptr = past_rdmp_tci_data;
        if (test_this == TestThis::RdmpTildeD) {
          // Assumes min is positive, increase it so we fail the TCI
          rdmp_tci_data_ptr->min_variables_values[0] *= 1.01;
        } else if (test_this == TestThis::RdmpTildeYe) {
          // Assumes min is positive, increase it so we fail the TCI
          rdmp_tci_data_ptr->min_variables_values[1] *= 1.01;
        } else if (test_this == TestThis::RdmpTildeTau) {
          // Assumes min is positive, increase it so we fail the TCI
          rdmp_tci_data_ptr->min_variables_values[2] *= 1.01;
        } else if (test_this == TestThis::RdmpMagnitudeTildeB) {
          // Assumes min is positive, increase it so we fail the TCI
          rdmp_tci_data_ptr->min_variables_values[3] *= 1.01;
        }
      },
      make_not_null(&box));

  const std::tuple<int, evolution::dg::subcell::RdmpTciData> result =
      db::mutate_apply<grmhd::ValenciaDivClean::subcell::TciOnFdGrid>(
          make_not_null(&box), persson_exponent, false);
  CHECK(get<1>(result) == expected_rdmp_tci_data);

  if (test_this == TestThis::AllGood) {
    CHECK_FALSE(get<0>(result));
  } else {
    CHECK(get<0>(result) == expected_tci_status);
  }
}
}  // namespace

SPECTRE_TEST_CASE("Unit.Evolution.Systems.ValenciaDivClean.Subcell.TciOnFdGrid",
                  "[Unit][Evolution]") {
  test(TestThis::AllGood, 0);
  test(TestThis::Atmosphere, 0);
  test(TestThis::NegativeTildeD, 1);
  test(TestThis::NegativeTildeYe, 1);
  test(TestThis::NegativeTildeTau, 1);
  test(TestThis::NeededFixing, 2);
  test(TestThis::PerssonTildeD, 3);
  test(TestThis::PerssonTildeYe, 4);
  test(TestThis::PerssonPressure, 5);
  test(TestThis::RdmpTildeD, 6);
  test(TestThis::RdmpTildeYe, 7);
  test(TestThis::RdmpTildeTau, 8);
  test(TestThis::RdmpMagnitudeTildeB, 9);
  test(TestThis::PerssonTildeB, 10);
}
