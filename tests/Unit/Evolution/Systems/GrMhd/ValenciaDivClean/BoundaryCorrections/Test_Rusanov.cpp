// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <array>
#include <cstddef>
#include <string>

#include "Evolution/Systems/GrMhd/ValenciaDivClean/BoundaryCorrections/Factory.hpp"
#include "Evolution/Systems/GrMhd/ValenciaDivClean/BoundaryCorrections/Rusanov.hpp"
#include "Evolution/Systems/GrMhd/ValenciaDivClean/System.hpp"
#include "Framework/SetupLocalPythonEnvironment.hpp"
#include "Framework/TestCreation.hpp"
#include "Helpers/Evolution/DiscontinuousGalerkin/BoundaryCorrections.hpp"
#include "NumericalAlgorithms/Spectral/Basis.hpp"
#include "NumericalAlgorithms/Spectral/Mesh.hpp"
#include "NumericalAlgorithms/Spectral/Quadrature.hpp"
#include "PointwiseFunctions/Hydro/EquationsOfState/EquationOfState.hpp"
#include "PointwiseFunctions/Hydro/EquationsOfState/IdealFluid.hpp"
#include "PointwiseFunctions/Hydro/EquationsOfState/PolytropicFluid.hpp"
#include "PointwiseFunctions/Hydro/Tags.hpp"
#include "Utilities/TMPL.hpp"
#include "Utilities/TaggedTuple.hpp"

namespace {
// NOLINTNEXTLINE(misc-unused-alias-decls)
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
}  // namespace

SPECTRE_TEST_CASE("Unit.GrMhd.ValenciaDivClean.BoundaryCorrections.Rusanov",
                  "[Unit][GrMhd]") {
  PUPable_reg(grmhd::ValenciaDivClean::BoundaryCorrections::Rusanov);
  pypp::SetupLocalPythonEnvironment local_python_env{
      "Evolution/Systems/GrMhd/ValenciaDivClean/BoundaryCorrections"};
  MAKE_GENERATOR(gen);

  using system = grmhd::ValenciaDivClean::System;

  const tuples::TaggedTuple<hydro::Tags::GrmhdEquationOfState> volume_data{
      EquationsOfState::PolytropicFluid<true>{100.0, 2.0}.promote_to_3d_eos()};

  TestHelpers::evolution::dg::test_boundary_correction_conservation<system>(
      make_not_null(&gen),
      grmhd::ValenciaDivClean::BoundaryCorrections::Rusanov{},
      Mesh<2>{5, Spectral::Basis::Legendre, Spectral::Quadrature::Gauss},
      volume_data, {});

  TestHelpers::evolution::dg::test_boundary_correction_with_python<
      system, tmpl::list<ConvertPolytropic>>(
      make_not_null(&gen), "Rusanov", "dg_package_data", "dg_boundary_terms",
      grmhd::ValenciaDivClean::BoundaryCorrections::Rusanov{},
      Mesh<2>{5, Spectral::Basis::Legendre, Spectral::Quadrature::Gauss},
      volume_data, {});

  const auto rusanov = TestHelpers::test_creation<std::unique_ptr<
      grmhd::ValenciaDivClean::BoundaryCorrections::BoundaryCorrection>>(
      "Rusanov:");

  TestHelpers::evolution::dg::test_boundary_correction_with_python<
      system, tmpl::list<ConvertPolytropic>>(
      make_not_null(&gen), "Rusanov", "dg_package_data", "dg_boundary_terms",
      dynamic_cast<
          const grmhd::ValenciaDivClean::BoundaryCorrections::Rusanov&>(
          *rusanov),
      Mesh<2>{5, Spectral::Basis::Legendre, Spectral::Quadrature::Gauss},
      volume_data, {});

  CHECK_FALSE(grmhd::ValenciaDivClean::BoundaryCorrections::Rusanov{} !=
              grmhd::ValenciaDivClean::BoundaryCorrections::Rusanov{});
}
