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
#include "Evolution/DgSubcell/Projection.hpp"
#include "Evolution/DgSubcell/SubcellOptions.hpp"
#include "Evolution/DgSubcell/Tags/Mesh.hpp"
#include "Evolution/DgSubcell/Tags/SubcellOptions.hpp"
#include "Evolution/Systems/GrMhd/ValenciaDivClean/ConservativeFromPrimitive.hpp"
#include "Evolution/Systems/GrMhd/ValenciaDivClean/NewmanHamlin.hpp"
#include "Evolution/Systems/GrMhd/ValenciaDivClean/Subcell/TciOnDgGrid.hpp"
#include "Evolution/Systems/GrMhd/ValenciaDivClean/Subcell/TciOptions.hpp"
#include "Evolution/Systems/GrMhd/ValenciaDivClean/System.hpp"
#include "NumericalAlgorithms/Spectral/Mesh.hpp"
#include "NumericalAlgorithms/Spectral/Spectral.hpp"
#include "PointwiseFunctions/GeneralRelativity/Tags.hpp"
#include "PointwiseFunctions/Hydro/EquationsOfState/EquationOfState.hpp"
#include "PointwiseFunctions/Hydro/EquationsOfState/PolytropicFluid.hpp"
#include "PointwiseFunctions/Hydro/SpecificEnthalpy.hpp"
#include "PointwiseFunctions/Hydro/Tags.hpp"
#include "Utilities/Gsl.hpp"
namespace {
enum class TestThis {
  AllGood,
  SmallTildeD,
  InAtmosphere,
  TildeB2TooBig,
  PrimRecoveryFailed,
  PerssonTildeD,
  PerssonPressure,
  PerssonTildeB,
  NegativeTildeDSubcell,
  NegativeTildeTauSubcell,
  NegativeTildeTau,
  RdmpTildeD,
  RdmpTildeTau,
  RdmpMagnitudeTildeB
};

void test(const TestThis test_this, const int expected_tci_status,
          const bool element_stays_on_dg) {
  CAPTURE(test_this);
  CAPTURE(expected_tci_status);
  CAPTURE(element_stays_on_dg);
  const EquationsOfState::PolytropicFluid<true> eos{100.0, 2.0};
  const Mesh<3> mesh{6, Spectral::Basis::Legendre,
                     Spectral::Quadrature::GaussLobatto};
  const Mesh<3> subcell_mesh = evolution::dg::subcell::fd::mesh(mesh);
  using ConsVars =
      typename grmhd::ValenciaDivClean::System::variables_tag::type;
  using PrimVars = Variables<hydro::grmhd_tags<DataVector>>;

  const double persson_exponent = 4.0;
  PrimVars prim_vars{mesh.number_of_grid_points(), 0.0};
  get(get<hydro::Tags::RestMassDensity<DataVector>>(prim_vars)) = 1.0;
  get(get<hydro::Tags::ElectronFraction<DataVector>>(prim_vars)) = 0.1;
  if (test_this == TestThis::InAtmosphere) {
    get(get<hydro::Tags::RestMassDensity<DataVector>>(prim_vars)) = 1.0e-12;
  }
  get<hydro::Tags::SpecificInternalEnergy<DataVector>>(prim_vars) =
      eos.specific_internal_energy_from_density(
          get<hydro::Tags::RestMassDensity<DataVector>>(prim_vars));
  get(get<hydro::Tags::LorentzFactor<DataVector>>(prim_vars)) = 1.0;
  get<hydro::Tags::Pressure<DataVector>>(prim_vars) = eos.pressure_from_density(
      get<hydro::Tags::RestMassDensity<DataVector>>(prim_vars));
  get<hydro::Tags::SpecificEnthalpy<DataVector>>(prim_vars) =
      hydro::relativistic_specific_enthalpy(
          get<hydro::Tags::RestMassDensity<DataVector>>(prim_vars),
          get<hydro::Tags::SpecificInternalEnergy<DataVector>>(prim_vars),
          get<hydro::Tags::Pressure<DataVector>>(prim_vars));
  // set magnetic field to tiny but non-zero value
  for (size_t i = 0; i < 3; ++i) {
    get<hydro::Tags::MagneticField<DataVector, 3, Frame::Inertial>>(prim_vars)
        .get(i) = 1.0e-50;
  }
  // Tracking max field for check later
  double max_B_field = 1.0e-50;
  // Just use flat space since none of the TCI checks really depend on the
  // spacetime variables.
  tnsr::ii<DataVector, 3, Frame::Inertial> spatial_metric{
      mesh.number_of_grid_points(), 0.0};
  tnsr::II<DataVector, 3, Frame::Inertial> inv_spatial_metric{
      mesh.number_of_grid_points(), 0.0};
  for (size_t i = 0; i < 3; ++i) {
    spatial_metric.get(i, i) = inv_spatial_metric.get(i, i) = 1.0;
  }
  const Scalar<DataVector> sqrt_det_spatial_metric{mesh.number_of_grid_points(),
                                                   1.0};

  const grmhd::ValenciaDivClean::subcell::TciOptions tci_options{
      1.0e-20,
      1.e-3,
      1.0e-40,
      1.1e-12,
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
      ::Tags::Variables<typename ConsVars::tags_list>,
      ::Tags::Variables<typename PrimVars::tags_list>, ::domain::Tags::Mesh<3>,
      ::evolution::dg::subcell::Tags::Mesh<3>,
      hydro::Tags::EquationOfState<
          std::unique_ptr<EquationsOfState::EquationOfState<true, 1>>>,
      gr::Tags::SqrtDetSpatialMetric<DataVector>,
      gr::Tags::SpatialMetric<DataVector, 3>,
      gr::Tags::InverseSpatialMetric<DataVector, 3>,
      grmhd::ValenciaDivClean::subcell::Tags::TciOptions,
      evolution::dg::subcell::Tags::SubcellOptions<3>,
      evolution::dg::subcell::Tags::DataForRdmpTci>>(
      ConsVars{mesh.number_of_grid_points()}, prim_vars, mesh, subcell_mesh,
      std::unique_ptr<EquationsOfState::EquationOfState<true, 1>>{
          std::make_unique<EquationsOfState::PolytropicFluid<true>>(eos)},
      sqrt_det_spatial_metric, spatial_metric, inv_spatial_metric, tci_options,
      subcell_options, evolution::dg::subcell::RdmpTciData{});

  db::mutate_apply<grmhd::ValenciaDivClean::ConservativeFromPrimitive>(
      make_not_null(&box));

  const auto original_rest_mass_density =
      get<hydro::Tags::RestMassDensity<DataVector>>(prim_vars);

  // set B and Phi to 0 since they should be set by recovery
  db::mutate<hydro::Tags::MagneticField<DataVector, 3, Frame::Inertial>,
             hydro::Tags::DivergenceCleaningField<DataVector>>(
      [](const auto mag_field_ptr, const auto phi_ptr) {
        for (size_t i = 0; i < 3; ++i) {
          mag_field_ptr->get(i) = 0;
        }
        get(*phi_ptr) = 0;
      },
      make_not_null(&box));

  const size_t point_to_change = mesh.number_of_grid_points() / 2;

  if (test_this == TestThis::AllGood) {
    // slightly modify D to ensure that after the con2prim, the original density
    // does not match the new density
    db::mutate<grmhd::ValenciaDivClean::Tags::TildeD>(
        [point_to_change](const auto tilde_d_ptr) {
          get(*tilde_d_ptr)[point_to_change] *= 1.001;
        },
        make_not_null(&box));
  }
  if (test_this == TestThis::SmallTildeD) {
    db::mutate<grmhd::ValenciaDivClean::Tags::TildeD>(
        [point_to_change](const auto tilde_d_ptr) {
          get(*tilde_d_ptr)[point_to_change] = 1.0e-30;
        },
        make_not_null(&box));
  } else if (test_this == TestThis::InAtmosphere) {
    // Make sure the PerssonTCI would trigger in the atmosphere to verify that
    // the reason we didn't mark the cell as troubled is because we're in
    // atmosphere.
    db::mutate<hydro::Tags::Pressure<DataVector>>(
        [point_to_change](const auto pressure_ptr) {
          get(*pressure_ptr)[point_to_change] *= 1.5;
        },
        make_not_null(&box));
  } else if (test_this == TestThis::TildeB2TooBig) {
    const double large_B_field_magnitude = 1.0e4;
    db::mutate<grmhd::ValenciaDivClean::Tags::TildeB<Frame::Inertial>>(

        [&large_B_field_magnitude, point_to_change](const auto tilde_b_ptr) {
          get<0>(*tilde_b_ptr)[point_to_change] = large_B_field_magnitude;
          get<1>(*tilde_b_ptr)[point_to_change] = large_B_field_magnitude;
          get<2>(*tilde_b_ptr)[point_to_change] = large_B_field_magnitude;
        },
        make_not_null(&box));
    max_B_field = std::max(max_B_field, large_B_field_magnitude);

  } else if (test_this == TestThis::PrimRecoveryFailed) {
    db::mutate<grmhd::ValenciaDivClean::Tags::TildeS<Frame::Inertial>>(
        [point_to_change](const auto tilde_s_ptr) {
          // Manipulate one of the conserved variable in an (very) inconsistent
          // way so that primitive recovery fails.
          get<0>(*tilde_s_ptr)[point_to_change] = 1e3;
        },
        make_not_null(&box));
  } else if (test_this == TestThis::PerssonPressure) {
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
  } else if (test_this == TestThis::PerssonTildeB) {
    db::mutate<grmhd::ValenciaDivClean::Tags::TildeB<>>(

        [&max_B_field, point_to_change](const auto tilde_b_ptr) {
          for (size_t i = 0; i < 3; ++i) {
            tilde_b_ptr->get(i)[point_to_change] = 6.0;
            max_B_field =
                std::max(max_B_field, tilde_b_ptr->get(i)[point_to_change]);
          }
        },
        make_not_null(&box));
  } else if (test_this == TestThis::NegativeTildeDSubcell) {
    db::mutate<grmhd::ValenciaDivClean::Tags::TildeD>(
        [point_to_change](const auto tilde_d_ptr) {
          get(*tilde_d_ptr)[point_to_change] = 1.0e-200;
        },
        make_not_null(&box));
  } else if (test_this == TestThis::NegativeTildeTauSubcell) {
    db::mutate<grmhd::ValenciaDivClean::Tags::TildeTau>(
        [point_to_change](const auto tilde_d_ptr) {
          get(*tilde_d_ptr)[point_to_change] = 1.0e-200;
        },
        make_not_null(&box));
  } else if (test_this == TestThis::NegativeTildeTau) {
    db::mutate<grmhd::ValenciaDivClean::Tags::TildeTau>(
        [point_to_change](const auto tilde_d_ptr) {
          get(*tilde_d_ptr)[point_to_change] = -1.0e-20;
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
          max(evolution::dg::subcell::fd::project(
              get(db::get<grmhd::ValenciaDivClean::Tags::TildeD>(box)), mesh,
              subcell_mesh.extents()))),
      max(max(get(db::get<grmhd::ValenciaDivClean::Tags::TildeYe>(box))),
          max(evolution::dg::subcell::fd::project(
              get(db::get<grmhd::ValenciaDivClean::Tags::TildeYe>(box)), mesh,
              subcell_mesh.extents()))),
      max(max(get(db::get<grmhd::ValenciaDivClean::Tags::TildeTau>(box))),
          max(evolution::dg::subcell::fd::project(
              get(db::get<grmhd::ValenciaDivClean::Tags::TildeTau>(box)), mesh,
              subcell_mesh.extents()))),
      max(max(get(magnitude_tilde_b)),
          max(evolution::dg::subcell::fd::project(get(magnitude_tilde_b), mesh,
                                                  subcell_mesh.extents())))};
  past_rdmp_tci_data.min_variables_values = DataVector{
      min(min(get(db::get<grmhd::ValenciaDivClean::Tags::TildeD>(box))),
          min(evolution::dg::subcell::fd::project(
              get(db::get<grmhd::ValenciaDivClean::Tags::TildeD>(box)), mesh,
              subcell_mesh.extents()))),
      min(min(get(db::get<grmhd::ValenciaDivClean::Tags::TildeYe>(box))),
          min(evolution::dg::subcell::fd::project(
              get(db::get<grmhd::ValenciaDivClean::Tags::TildeYe>(box)), mesh,
              subcell_mesh.extents()))),
      min(min(get(db::get<grmhd::ValenciaDivClean::Tags::TildeTau>(box))),
          min(evolution::dg::subcell::fd::project(
              get(db::get<grmhd::ValenciaDivClean::Tags::TildeTau>(box)), mesh,
              subcell_mesh.extents()))),
      min(min(get(magnitude_tilde_b)),
          min(evolution::dg::subcell::fd::project(get(magnitude_tilde_b), mesh,
                                                  subcell_mesh.extents())))};

  const evolution::dg::subcell::RdmpTciData expected_rdmp_tci_data =
      past_rdmp_tci_data;

  // Modify past data if we are expecting an RDMP TCI failure.
  db::mutate<evolution::dg::subcell::Tags::DataForRdmpTci>(
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
      },
      make_not_null(&box));

  const std::tuple<int, evolution::dg::subcell::RdmpTciData> result =
      db::mutate_apply<grmhd::ValenciaDivClean::subcell::TciOnDgGrid<
          grmhd::ValenciaDivClean::PrimitiveRecoverySchemes::NewmanHamlin>>(
          make_not_null(&box), persson_exponent, element_stays_on_dg);

  CHECK(get<1>(result) == expected_rdmp_tci_data);

  if (test_this == TestThis::AllGood or test_this == TestThis::InAtmosphere) {
    CHECK_FALSE(get<0>(result));
    CHECK(db::get<hydro::Tags::MagneticField<DataVector, 3, Frame::Inertial>>(
              box) ==
          get<hydro::Tags::MagneticField<DataVector, 3, Frame::Inertial>>(
              prim_vars));
    CHECK(db::get<hydro::Tags::DivergenceCleaningField<DataVector>>(box) ==
          get<hydro::Tags::DivergenceCleaningField<DataVector>>(prim_vars));
    if (test_this == TestThis::AllGood) {
      // Density should not be the same for an updated Dtilde
      CHECK_FALSE(db::get<hydro::Tags::RestMassDensity<DataVector>>(box) ==
                  original_rest_mass_density);
    }
  } else {
    CHECK(get<0>(result) == expected_tci_status);
    // Ensure data before does not equal data after, if TCI is in trouble
    // box is output from tciondggrid, prim_vars are initial value
    if (element_stays_on_dg) {
      // The updated box primitive variables should have a maximum that match
      // the original primvars magnetic field x component: 1e-50 for non
      // B field TCI test cases, 6 for PerssonTildeB, or 1.0e4 for TildeB2TooBig
      CHECK(max(get<0>(
                db::get<
                    hydro::Tags::MagneticField<DataVector, 3, Frame::Inertial>>(
                    box))) == max_B_field);
    } else {
      // If element_stays_on_dg is false, then
      // the primitives will not update, causing a mismatch between
      // the updated box values and the original primvars.
      CHECK(db::get<hydro::Tags::MagneticField<DataVector, 3, Frame::Inertial>>(
                box) !=
            get<hydro::Tags::MagneticField<DataVector, 3, Frame::Inertial>>(
                prim_vars));
    }
  }
}
}  // namespace

SPECTRE_TEST_CASE("Unit.Evolution.Systems.ValenciaDivClean.Subcell.TciOnDgGrid",
                  "[Unit][Evolution]") {
  for (const auto& element_stays_on_dg : make_array(false, true)) {
    test(TestThis::AllGood, 0, element_stays_on_dg);
    test(TestThis::InAtmosphere, 0, element_stays_on_dg);
    test(TestThis::SmallTildeD, -1, element_stays_on_dg);
    test(TestThis::NegativeTildeDSubcell, -1, element_stays_on_dg);
    test(TestThis::NegativeTildeTau, -2, element_stays_on_dg);
    test(TestThis::NegativeTildeTauSubcell, -2, element_stays_on_dg);
    test(TestThis::TildeB2TooBig, -3, element_stays_on_dg);
    test(TestThis::PrimRecoveryFailed, -4, element_stays_on_dg);
    test(TestThis::PerssonTildeD, -5, element_stays_on_dg);
    test(TestThis::PerssonPressure, -7, element_stays_on_dg);
    test(TestThis::PerssonTildeB, -8, element_stays_on_dg);
    test(TestThis::RdmpTildeD, -9, element_stays_on_dg);
    test(TestThis::RdmpTildeTau, -10, element_stays_on_dg);
    test(TestThis::RdmpMagnitudeTildeB, -11, element_stays_on_dg);
  }  // end for loop over element_stays_on_dg
}
