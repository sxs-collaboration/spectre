// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <array>
#include <cstddef>
#include <string>

#include "Evolution/Systems/GrMhd/ValenciaDivClean/BoundaryCorrections/Factory.hpp"
#include "Evolution/Systems/GrMhd/ValenciaDivClean/BoundaryCorrections/Hll.hpp"
#include "Evolution/Systems/GrMhd/ValenciaDivClean/System.hpp"
#include "Framework/SetupLocalPythonEnvironment.hpp"
#include "Framework/TestCreation.hpp"
#include "Helpers/Evolution/DiscontinuousGalerkin/BoundaryCorrections.hpp"
#include "NumericalAlgorithms/Spectral/Basis.hpp"
#include "NumericalAlgorithms/Spectral/Mesh.hpp"
#include "NumericalAlgorithms/Spectral/Quadrature.hpp"
#include "PointwiseFunctions/Hydro/EquationsOfState/EquationOfState.hpp"
#include "PointwiseFunctions/Hydro/EquationsOfState/PolytropicFluid.hpp"
#include "PointwiseFunctions/Hydro/Tags.hpp"
#include "Utilities/TaggedTuple.hpp"

namespace {
namespace helpers = TestHelpers::evolution::dg;

struct ConvertPolytropic {
  using unpacked_container = bool;
  using packed_container = EquationsOfState::EquationOfState<true, 3>;
  using packed_type = bool;

  static inline unpacked_container unpack(const packed_container& /*packed*/,
                                          const size_t /*grid_point_index*/) {
    return true;
  }

  [[noreturn]] static inline void pack(
      const gsl::not_null<packed_container*> /*packed*/,
      const unpacked_container& /*unpacked*/,
      const size_t /*grid_point_index*/) {
    ERROR("Should not be converting an EOS from an unpacked to a packed type");
  }

  static inline size_t get_size(const packed_container& /*packed*/) {
    return 1;
  }
};

SPECTRE_TEST_CASE("Unit.GrMhd.ValenciaDivClean.BoundaryCorrections.Hll",
                  "[Unit][GrMhd]") {
  PUPable_reg(grmhd::ValenciaDivClean::BoundaryCorrections::Hll);
  pypp::SetupLocalPythonEnvironment local_python_env{
      "Evolution/Systems/GrMhd/ValenciaDivClean/BoundaryCorrections"};
  MAKE_GENERATOR(gen);

  using system = grmhd::ValenciaDivClean::System;

  const tuples::TaggedTuple<
      helpers::Tags::Range<gr::Tags::Lapse<DataVector>>,
      helpers::Tags::Range<gr::Tags::Shift<DataVector, 3>>>
      ranges{std::array{0.3, 1.0}, std::array{0.01, 0.02}};
  const tuples::TaggedTuple<hydro::Tags::GrmhdEquationOfState> volume_data{
      EquationsOfState::PolytropicFluid<true>{100.0, 2.0}.promote_to_3d_eos()};

  TestHelpers::evolution::dg::test_boundary_correction_conservation<system>(
      make_not_null(&gen),
      grmhd::ValenciaDivClean::BoundaryCorrections::Hll{1.0e-30, 1.0e-8},
      Mesh<2>{5, Spectral::Basis::Legendre, Spectral::Quadrature::Gauss},
      volume_data, ranges);

  TestHelpers::evolution::dg::test_boundary_correction_with_python<
      system, tmpl::list<ConvertPolytropic>>(
      make_not_null(&gen), "Hll", "dg_package_data", "dg_boundary_terms",
      grmhd::ValenciaDivClean::BoundaryCorrections::Hll{1.0e-30, 1.0e-8},
      Mesh<2>{5, Spectral::Basis::Legendre, Spectral::Quadrature::Gauss},
      volume_data, ranges);

  // Test hydro
  const tuples::TaggedTuple<
      helpers::Tags::Range<gr::Tags::Lapse<DataVector>>,
      helpers::Tags::Range<gr::Tags::Shift<DataVector, 3>>,
      helpers::Tags::Range<grmhd::ValenciaDivClean::Tags::TildeB<>>>
      ranges_hydro{std::array{0.3, 1.0}, std::array{0.01, 0.02},
                   std::array{1.0e-20, 1.0e-25}};
  TestHelpers::evolution::dg::test_boundary_correction_with_python<
      system, tmpl::list<ConvertPolytropic>>(
      make_not_null(&gen), "Hll", "dg_package_data", "dg_boundary_terms",
      grmhd::ValenciaDivClean::BoundaryCorrections::Hll{1.0e-30, 1.0e-8},
      Mesh<2>{5, Spectral::Basis::Legendre, Spectral::Quadrature::Gauss},
      volume_data, ranges_hydro);

  // Test atmosphere density cutoff
  const tuples::TaggedTuple<
      helpers::Tags::Range<hydro::Tags::RestMassDensity<DataVector>>,
      helpers::Tags::Range<gr::Tags::Lapse<DataVector>>,
      helpers::Tags::Range<gr::Tags::Shift<DataVector, 3>>>
      ranges_atmo{std::array{1.0e-10, 1.0e-9}, std::array{0.3, 1.0},
                  std::array{0.01, 0.02}};
  TestHelpers::evolution::dg::test_boundary_correction_with_python<
      system, tmpl::list<ConvertPolytropic>>(
      make_not_null(&gen), "Hll", "dg_package_data", "dg_boundary_terms",
      grmhd::ValenciaDivClean::BoundaryCorrections::Hll{1.0e-30, 1.0e-8},
      Mesh<2>{5, Spectral::Basis::Legendre, Spectral::Quadrature::Gauss},
      volume_data, ranges_atmo);

  const auto hll = TestHelpers::test_creation<std::unique_ptr<
      grmhd::ValenciaDivClean::BoundaryCorrections::BoundaryCorrection>>(
      "Hll:\n"
      "  MagneticFieldMagnitudeForHydro: 1.0e-30\n"
      "  AtmosphereDensityCutoff: 1.0e-8\n");

  TestHelpers::evolution::dg::test_boundary_correction_with_python<
      system, tmpl::list<ConvertPolytropic>>(
      make_not_null(&gen), "Hll", "dg_package_data", "dg_boundary_terms",
      dynamic_cast<const grmhd::ValenciaDivClean::BoundaryCorrections::Hll&>(
          *hll),
      Mesh<2>{5, Spectral::Basis::Legendre, Spectral::Quadrature::Gauss},
      volume_data, ranges);

  CHECK_FALSE(
      grmhd::ValenciaDivClean::BoundaryCorrections::Hll{1.0e-30, 1.0e-8} !=
      grmhd::ValenciaDivClean::BoundaryCorrections::Hll{1.0e-30, 1.0e-8});
  CHECK(grmhd::ValenciaDivClean::BoundaryCorrections::Hll{1.0e-30, 1.0e-8} !=
        grmhd::ValenciaDivClean::BoundaryCorrections::Hll{2.0e-30, 1.0e-8});
  CHECK(grmhd::ValenciaDivClean::BoundaryCorrections::Hll{1.0e-30, 1.0e-8} !=
        grmhd::ValenciaDivClean::BoundaryCorrections::Hll{1.0e-30, 2.0e-8});
}
}  // namespace
