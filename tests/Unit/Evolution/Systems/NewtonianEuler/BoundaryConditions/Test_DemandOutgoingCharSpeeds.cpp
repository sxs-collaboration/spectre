// Distributed under the MIT License
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <array>
#include <cstddef>

#include "DataStructures/DataBox/DataBox.hpp"
#include "Evolution/Systems/NewtonianEuler/BoundaryConditions/BoundaryCondition.hpp"
#include "Evolution/Systems/NewtonianEuler/BoundaryConditions/DemandOutgoingCharSpeeds.hpp"
#include "Evolution/Systems/NewtonianEuler/BoundaryCorrections/Rusanov.hpp"
#include "Evolution/Systems/NewtonianEuler/System.hpp"
#include "Evolution/Systems/NewtonianEuler/Tags.hpp"
#include "Framework/SetupLocalPythonEnvironment.hpp"
#include "Helpers/Evolution/DiscontinuousGalerkin/BoundaryConditions.hpp"
#include "Helpers/Evolution/DiscontinuousGalerkin/Range.hpp"
#include "PointwiseFunctions/Hydro/EquationsOfState/Barotropic2D.hpp"
#include "PointwiseFunctions/Hydro/EquationsOfState/EquationOfState.hpp"
#include "PointwiseFunctions/Hydro/EquationsOfState/IdealFluid.hpp"
#include "PointwiseFunctions/Hydro/EquationsOfState/PolytropicFluid.hpp"
#include "PointwiseFunctions/Hydro/Tags.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/TMPL.hpp"
#include "Utilities/TaggedTuple.hpp"

namespace helpers = TestHelpers::evolution::dg;

namespace {

struct ConvertPolytropic {
  using unpacked_container = bool;
  using packed_container = EquationsOfState::EquationOfState<false, 2>;
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

struct ConvertIdeal {
  using unpacked_container = bool;
  using packed_container = EquationsOfState::EquationOfState<false, 2>;
  using packed_type = bool;

  static inline unpacked_container unpack(const packed_container& /*packed*/,
                                          const size_t /*grid_point_index*/) {
    return false;
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

struct DummyInitialData {
  using argument_tags = tmpl::list<>;
  struct source_term_type {
    using sourced_variables = tmpl::list<>;
    using argument_tags = tmpl::list<>;
  };
};

template <size_t Dim, typename EosType>
void test(EosType& eos) {
  MAKE_GENERATOR(gen);

  auto box = db::create<db::AddSimpleTags<
      hydro::Tags::EquationOfState<false, EosType::thermodynamic_dim>>>(
      eos.get_clone());

  const tuples::TaggedTuple<
      helpers::Tags::Range<hydro::Tags::RestMassDensity<DataVector>>,
      helpers::Tags::Range<hydro::Tags::SpecificInternalEnergy<DataVector>>>
      ranges{std::array{1.0e-30, 1.0}, std::array{1.0e-30, 1.0}};

  helpers::test_boundary_condition_with_python<
      NewtonianEuler::BoundaryConditions::DemandOutgoingCharSpeeds<Dim>,
      NewtonianEuler::BoundaryConditions::BoundaryCondition<Dim>,
      NewtonianEuler::System<Dim>,
      tmpl::list<NewtonianEuler::BoundaryCorrections::Rusanov<Dim>>,
      tmpl::list<tmpl::conditional_t<
          std::is_same_v<EosType, EquationsOfState::IdealFluid<false>>,
          ConvertIdeal, ConvertPolytropic>>>(
      make_not_null(&gen),
      "Evolution.Systems.NewtonianEuler.BoundaryConditions."
      "DemandOutgoingCharSpeeds",
      tuples::TaggedTuple<helpers::Tags::PythonFunctionForErrorMessage<>>{
          "error"},
      "DemandOutgoingCharSpeeds:\n", Index<Dim - 1>{Dim == 1 ? 1 : 5}, box,
      ranges);
}  // namespace
}  // namespace

SPECTRE_TEST_CASE(
    "Unit.NewtonianEuler.BoundaryConditions.DemandOutgoingCharSpeeds",
    "[Unit][Evolution]") {
  pypp::SetupLocalPythonEnvironment local_python_env{""};

  EquationsOfState::Barotropic2D eos_polytrope{
      EquationsOfState::PolytropicFluid<false>{1.4, 5.0 / 3.0}};
  test<1>(eos_polytrope);
  test<2>(eos_polytrope);
  test<3>(eos_polytrope);

  EquationsOfState::IdealFluid<false> eos_ideal{1.3};
  test<1>(eos_ideal);
  test<2>(eos_ideal);
  test<3>(eos_ideal);
}
