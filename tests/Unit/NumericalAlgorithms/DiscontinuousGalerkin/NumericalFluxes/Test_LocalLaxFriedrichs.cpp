// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "tests/Unit/TestingFramework.hpp"

#include <array>
#include <cstddef>
#include <limits>

#include "DataStructures/DataBox/Prefixes.hpp"  // IWYU pragma: keep
#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "DataStructures/Variables.hpp"  // IWYU pragma: keep
#include "NumericalAlgorithms/DiscontinuousGalerkin/NumericalFluxes/LocalLaxFriedrichs.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/TMPL.hpp"
#include "tests/Unit/NumericalAlgorithms/DiscontinuousGalerkin/NumericalFluxes/TestHelpers.hpp"
#include "tests/Unit/Pypp/CheckWithRandomValues.hpp"
#include "tests/Unit/Pypp/SetupLocalPythonEnvironment.hpp"

// IWYU pragma: no_forward_declare Tensor

namespace {

template <size_t Dim>
void test_llf_flux_tags() noexcept {
  using system = TestHelpers::NumericalFluxes::System<Dim>;
  using llf_flux = dg::NumericalFluxes::LocalLaxFriedrichs<system>;

  using expected_type_argument_tags = tmpl::list<
      Tags::NormalDotFlux<TestHelpers::NumericalFluxes::Tags::Variable1>,
      Tags::NormalDotFlux<TestHelpers::NumericalFluxes::Tags::Variable2<Dim>>,
      Tags::NormalDotFlux<TestHelpers::NumericalFluxes::Tags::Variable3<Dim>>,
      Tags::NormalDotFlux<TestHelpers::NumericalFluxes::Tags::Variable4<Dim>>,
      TestHelpers::NumericalFluxes::Tags::Variable1,
      TestHelpers::NumericalFluxes::Tags::Variable2<Dim>,
      TestHelpers::NumericalFluxes::Tags::Variable3<Dim>,
      TestHelpers::NumericalFluxes::Tags::Variable4<Dim>,
      typename TestHelpers::NumericalFluxes::System<Dim>::char_speeds_tag>;

  static_assert(
      cpp17::is_same_v<typename llf_flux::argument_tags,
                       expected_type_argument_tags>,
      "Failed testing dg::NumericalFluxes::LocalLaxFriedrichs::argument_tags");

  using expected_type_package_tags = tmpl::list<
      Tags::NormalDotFlux<TestHelpers::NumericalFluxes::Tags::Variable1>,
      Tags::NormalDotFlux<TestHelpers::NumericalFluxes::Tags::Variable2<Dim>>,
      Tags::NormalDotFlux<TestHelpers::NumericalFluxes::Tags::Variable3<Dim>>,
      Tags::NormalDotFlux<TestHelpers::NumericalFluxes::Tags::Variable4<Dim>>,
      TestHelpers::NumericalFluxes::Tags::Variable1,
      TestHelpers::NumericalFluxes::Tags::Variable2<Dim>,
      TestHelpers::NumericalFluxes::Tags::Variable3<Dim>,
      TestHelpers::NumericalFluxes::Tags::Variable4<Dim>,
      typename dg::NumericalFluxes::LocalLaxFriedrichs<
          TestHelpers::NumericalFluxes::System<Dim>>::MaxAbsCharSpeed>;

  static_assert(
      cpp17::is_same_v<typename llf_flux::package_tags,
                       expected_type_package_tags>,
      "Failed testing dg::NumericalFluxes::LocalLaxFriedrichs::package_tags");
}

template <size_t Dim>
void apply_llf_flux(
    const gsl::not_null<Scalar<DataVector>*> n_dot_num_f_1,
    const gsl::not_null<tnsr::I<DataVector, Dim>*> n_dot_num_f_2,
    const gsl::not_null<tnsr::i<DataVector, Dim>*> n_dot_num_f_3,
    const gsl::not_null<tnsr::Ij<DataVector, Dim>*> n_dot_num_f_4,
    const Scalar<DataVector>& n_dot_f_1_int,
    const tnsr::I<DataVector, Dim>& n_dot_f_2_int,
    const tnsr::i<DataVector, Dim>& n_dot_f_3_int,
    const tnsr::Ij<DataVector, Dim>& n_dot_f_4_int,
    const Scalar<DataVector>& var_1_int,
    const tnsr::I<DataVector, Dim>& var_2_int,
    const tnsr::i<DataVector, Dim>& var_3_int,
    const tnsr::Ij<DataVector, Dim>& var_4_int,
    const Scalar<DataVector>& minus_n_dot_f_1_ext,
    const tnsr::I<DataVector, Dim>& minus_n_dot_f_2_ext,
    const tnsr::i<DataVector, Dim>& minus_n_dot_f_3_ext,
    const tnsr::Ij<DataVector, Dim>& minus_n_dot_f_4_ext,
    const Scalar<DataVector>& var_1_ext,
    const tnsr::I<DataVector, Dim>& var_2_ext,
    const tnsr::i<DataVector, Dim>& var_3_ext,
    const tnsr::Ij<DataVector, Dim>& var_4_ext) noexcept {
  using llf_flux = dg::NumericalFluxes::LocalLaxFriedrichs<
      TestHelpers::NumericalFluxes::System<Dim>>;

  llf_flux flux_computer{};

  Variables<typename llf_flux::package_tags> packaged_data_interior(
      n_dot_f_1_int.begin()->size(),
      std::numeric_limits<double>::signaling_NaN());
  flux_computer.package_data(
      make_not_null(&packaged_data_interior), n_dot_f_1_int, n_dot_f_2_int,
      n_dot_f_3_int, n_dot_f_4_int, var_1_int, var_2_int, var_3_int, var_4_int,
      TestHelpers::NumericalFluxes::characteristic_speeds(var_1_int, var_2_int,
                                                          var_3_int));

  Variables<typename llf_flux::package_tags> packaged_data_exterior(
      n_dot_f_1_int.begin()->size(),
      std::numeric_limits<double>::signaling_NaN());
  flux_computer.package_data(
      make_not_null(&packaged_data_exterior), minus_n_dot_f_1_ext,
      minus_n_dot_f_2_ext, minus_n_dot_f_3_ext, minus_n_dot_f_4_ext, var_1_ext,
      var_2_ext, var_3_ext, var_4_ext,
      TestHelpers::NumericalFluxes::characteristic_speeds(var_1_ext, var_2_ext,
                                                          var_3_ext));

  TestHelpers::NumericalFluxes::apply_numerical_flux(
      flux_computer, packaged_data_interior, packaged_data_exterior,
      n_dot_num_f_1, n_dot_num_f_2, n_dot_num_f_3, n_dot_num_f_4);
}

template <size_t Dim>
void test_llf_flux(const DataVector& used_for_size) noexcept {
  pypp::check_with_random_values<16>(
      &apply_llf_flux<Dim>, "TestFunctions",
      {"apply_var_1_llf_flux", "apply_var_2_llf_flux", "apply_var_3_llf_flux",
       "apply_var_4_llf_flux"},
      {{{-1.0, 1.0},
        {-1.0, 1.0},
        {-1.0, 1.0},
        {-1.0, 1.0},
        {0.0, 1.0},
        {-1.0, 1.0},
        {-1.0, 1.0},
        {-1.0, 1.0},
        {-1.0, 1.0},
        {-1.0, 1.0},
        {-1.0, 1.0},
        {-1.0, 1.0},
        {0.0, 1.0},
        {-1.0, 1.0},
        {-1.0, 1.0},
        {-1.0, 1.0}}},
      used_for_size);
}

}  // namespace

SPECTRE_TEST_CASE("Unit.DiscontinuousGalerkin.Fluxes.LocalLaxFriedrichs",
                  "[Unit][NumericalAlgorithms][Fluxes]") {
  pypp::SetupLocalPythonEnvironment local_python_env{
      "NumericalAlgorithms/DiscontinuousGalerkin/NumericalFluxes"};

  INVOKE_TEST_FUNCTION(test_llf_flux_tags, (), (1, 2, 3));

  GENERATE_UNINITIALIZED_DATAVECTOR;
  CHECK_FOR_DATAVECTORS(test_llf_flux, (1, 2, 3))
}
