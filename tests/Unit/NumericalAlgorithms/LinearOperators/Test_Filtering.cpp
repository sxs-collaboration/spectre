// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "tests/Unit/TestingFramework.hpp"

#include <array>
#include <cmath>
#include <cstddef>
#include <string>

#include "DataStructures/DataBox/DataBox.hpp"
#include "DataStructures/DataBox/Tag.hpp"
#include "DataStructures/DataVector.hpp"  // IWYU pragma: keep
#include "DataStructures/Matrix.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "DataStructures/Variables.hpp"
#include "Domain/Mesh.hpp"
#include "Domain/Tags.hpp"  // IWYU pragma: keep
#include "NumericalAlgorithms/LinearOperators/ApplyMatrices.hpp"
#include "NumericalAlgorithms/LinearOperators/ExponentialFilter.hpp"
#include "NumericalAlgorithms/LinearOperators/FilterAction.hpp"
#include "NumericalAlgorithms/Spectral/Filtering.hpp"
#include "NumericalAlgorithms/Spectral/Spectral.hpp"
#include "Parallel/ParallelComponentHelpers.hpp"
#include "Parallel/PhaseDependentActionList.hpp"  // IWYU pragma: keep
#include "Utilities/Gsl.hpp"
#include "Utilities/Requires.hpp"
#include "Utilities/TMPL.hpp"
#include "tests/Unit/ActionTesting.hpp"
#include "tests/Unit/TestCreation.hpp"

// IWYU pragma: no_forward_declare ActionTesting::InitializeDataBox
// IWYU pragma: no_forward_declare dg::Actions::ExponentialFilter

namespace {
namespace Tags {
struct ScalarVar : db::SimpleTag {
  using type = Scalar<DataVector>;
};

template <size_t Dim>
struct VectorVar : db::SimpleTag {
  using type = tnsr::I<DataVector, Dim>;
};
}  // namespace Tags

template <size_t Dim>
struct System {
  static constexpr size_t volume_dim = Dim;
  using variables_tag =
      ::Tags::Variables<tmpl::list<Tags::ScalarVar, Tags::VectorVar<Dim>>>;
};

template <typename Metavariables>
struct Component {
  using metavariables = Metavariables;
  static constexpr size_t dim = metavariables::system::volume_dim;

  using chare_type = ActionTesting::MockArrayChare;
  using array_index = int;
  using simple_tags =
      db::AddSimpleTags<domain::Tags::Mesh<dim>,
                        typename metavariables::system::variables_tag>;

  using phase_dependent_action_list = tmpl::list<
      Parallel::PhaseActions<
          typename Metavariables::Phase, Metavariables::Phase::Initialization,
          tmpl::list<ActionTesting::InitializeDataBox<simple_tags>>>,
      /// [action_list_example]
      Parallel::PhaseActions<
          typename Metavariables::Phase, Metavariables::Phase::Testing,
          tmpl::conditional_t<
              metavariables::filter_individually,
              tmpl::list<dg::Actions::Filter<Filters::Exponential<0>,
                                             tmpl::list<Tags::ScalarVar>>,
                         dg::Actions::Filter<Filters::Exponential<1>,
                                             tmpl::list<Tags::VectorVar<dim>>>>,
              tmpl::list<dg::Actions::Filter<
                  Filters::Exponential<0>,
                  tmpl::list<Tags::VectorVar<dim>, Tags::ScalarVar>>>>>>;
  /// [action_list_example]
};

template <size_t Dim, bool FilterIndividually>
struct Metavariables {
  static constexpr bool filter_individually = FilterIndividually;

  using system = System<Dim>;
  static constexpr bool local_time_stepping = true;
  using component_list = tmpl::list<Component<Metavariables>>;
  enum class Phase { Initialization, Testing, Exit };
};

template <typename Metavariables,
          Requires<Metavariables::filter_individually> = nullptr>
typename ActionTesting::MockRuntimeSystem<Metavariables>::CacheTuple
create_cache_tuple(const double alpha, const unsigned half_power,
                   const bool disable_for_debugging) noexcept {
  return {Filters::Exponential<0>{alpha, half_power, disable_for_debugging},
          Filters::Exponential<1>{2.0 * alpha, 2 * half_power,
                                  disable_for_debugging}};
}

template <typename Metavariables,
          Requires<not Metavariables::filter_individually> = nullptr>
typename ActionTesting::MockRuntimeSystem<Metavariables>::CacheTuple
create_cache_tuple(const double alpha, const unsigned half_power,
                   const bool disable_for_debugging) noexcept {
  return {Filters::Exponential<0>{alpha, half_power, disable_for_debugging}};
}

template <size_t Dim, Spectral::Basis BasisType,
          Spectral::Quadrature QuadratureType, bool FilterIndividually>
void test_exponential_filter_action(const double alpha,
                                    const unsigned half_power,
                                    const bool disable_for_debugging) noexcept {
  CAPTURE(BasisType);
  CAPTURE(QuadratureType);
  CAPTURE(disable_for_debugging);

  // Need to increase approx slightly on some hardware
  Approx custom_approx = Approx::custom().epsilon(5.0e-13);

  using metavariables = Metavariables<Dim, FilterIndividually>;
  using component = Component<metavariables>;

  // Division by Dim to reduce time of test
  for (size_t num_pts =
           Spectral::minimum_number_of_points<BasisType, QuadratureType>;
       num_pts < Spectral::maximum_number_of_points<BasisType> / Dim;
       ++num_pts) {
    CAPTURE(num_pts);
    const Mesh<Dim> mesh(num_pts, BasisType, QuadratureType);

    Variables<tmpl::list<Tags::ScalarVar, Tags::VectorVar<Dim>>> initial_vars(
        mesh.number_of_grid_points());
    for (size_t i = 0; i < mesh.number_of_grid_points(); ++i) {
      get(get<Tags::ScalarVar>(initial_vars))[i] = pow(i, num_pts) * 0.5;
      for (size_t d = 0; d < Dim; ++d) {
        get<Tags::VectorVar<Dim>>(initial_vars).get(d)[i] =
            d + pow(i, num_pts) * 0.75;
      }
    }

    ActionTesting::MockRuntimeSystem<metavariables> runner(
        create_cache_tuple<metavariables>(alpha, half_power,
                                          disable_for_debugging));
    ActionTesting::emplace_component_and_initialize<component>(
        &runner, 0, {mesh, initial_vars});
    ActionTesting::set_phase(make_not_null(&runner),
                             metavariables::Phase::Testing);

    ActionTesting::next_action<component>(make_not_null(&runner), 0);
    if (FilterIndividually) {
      ActionTesting::next_action<component>(make_not_null(&runner), 0);
    }

    std::array<Matrix, Dim> filter_scalar{};
    std::array<Matrix, Dim> filter_vector{};
    for (size_t d = 0; d < Dim; d++) {
      if (disable_for_debugging) {
        gsl::at(filter_scalar, d) = Matrix{};
        gsl::at(filter_vector, d) = Matrix{};
      } else {
        gsl::at(filter_scalar, d) = Spectral::filtering::exponential_filter(
            mesh.slice_through(d), alpha, half_power);
        if (FilterIndividually) {
          gsl::at(filter_vector, d) = Spectral::filtering::exponential_filter(
              mesh.slice_through(d), 2.0 * alpha, 2 * half_power);
        } else {
          gsl::at(filter_vector, d) = gsl::at(filter_scalar, d);
        }
      }
    }

    Scalar<DataVector> expected_scalar(mesh.number_of_grid_points(), 0.0);
    tnsr::I<DataVector, Dim> expected_vector(mesh.number_of_grid_points(), 0.0);
    apply_matrices(make_not_null(&get(expected_scalar)), filter_scalar,
                   get(get<Tags::ScalarVar>(initial_vars)), mesh.extents());
    for (size_t d = 0; d < Dim; d++) {
      apply_matrices(make_not_null(&expected_vector.get(d)), filter_vector,
                     get<Tags::VectorVar<Dim>>(initial_vars).get(d),
                     mesh.extents());
    }
    CHECK_ITERABLE_CUSTOM_APPROX(
        expected_scalar,
        (ActionTesting::get_databox_tag<component, Tags::ScalarVar>(runner, 0)),
        custom_approx);
    CHECK_ITERABLE_CUSTOM_APPROX(
        expected_vector,
        (ActionTesting::get_databox_tag<component, Tags::VectorVar<Dim>>(runner,
                                                                         0)),
        custom_approx);
  }
}

template <size_t Dim, bool FilterIndividually>
void invoke_test_exponential_filter_action(
    const double alpha, const unsigned half_power,
    const bool disable_for_debugging) noexcept {
  test_exponential_filter_action<Dim, Spectral::Basis::Legendre,
                                 Spectral::Quadrature::GaussLobatto,
                                 FilterIndividually>(alpha, half_power,
                                                     disable_for_debugging);
  test_exponential_filter_action<Dim, Spectral::Basis::Legendre,
                                 Spectral::Quadrature::Gauss,
                                 FilterIndividually>(alpha, half_power,
                                                     disable_for_debugging);
  test_exponential_filter_action<Dim, Spectral::Basis::Chebyshev,
                                 Spectral::Quadrature::GaussLobatto,
                                 FilterIndividually>(alpha, half_power,
                                                     disable_for_debugging);
  test_exponential_filter_action<Dim, Spectral::Basis::Chebyshev,
                                 Spectral::Quadrature::Gauss,
                                 FilterIndividually>(alpha, half_power,
                                                     disable_for_debugging);
}

template <size_t Dim>
void test_exponential_filter_creation() noexcept {
  using Filter = Filters::Exponential<0>;

  const Filter filter = TestHelpers::test_creation<Filter>(
      "Alpha: 36\n"
      "HalfPower: 32\n");

  CHECK(filter == Filter{36.0, 32, false});
  CHECK_FALSE(filter == Filter{35.0, 32, false});
  CHECK_FALSE(filter == Filter{36.0, 33, false});
  CHECK_FALSE(filter == Filter{36.0, 32, true});

  CHECK_FALSE(filter != Filter{36.0, 32, false});
  CHECK(filter != Filter{35.0, 32, false});
  CHECK(filter != Filter{36.0, 33, false});
  CHECK(filter != Filter{36.0, 32, true});
}
}  // namespace

SPECTRE_TEST_CASE("Unit.Numerical.LinearOperators.Filter",
                  "[NumericalAlgorithms][LinearOperators][Unit]") {
  // Can't do a loop over different alpha and half_power values because matrices
  // are cached in the action.
  const double alpha = 10.0;
  const unsigned half_power = 16;
  invoke_test_exponential_filter_action<1, true>(alpha, half_power, false);
  invoke_test_exponential_filter_action<2, true>(alpha, half_power, false);
  invoke_test_exponential_filter_action<3, true>(alpha, half_power, false);
  invoke_test_exponential_filter_action<1, false>(alpha, half_power, false);
  invoke_test_exponential_filter_action<2, false>(alpha, half_power, false);
  invoke_test_exponential_filter_action<3, false>(alpha, half_power, false);

  invoke_test_exponential_filter_action<1, true>(alpha, half_power, true);
  invoke_test_exponential_filter_action<2, true>(alpha, half_power, true);
  invoke_test_exponential_filter_action<3, true>(alpha, half_power, true);
  invoke_test_exponential_filter_action<1, false>(alpha, half_power, true);
  invoke_test_exponential_filter_action<2, false>(alpha, half_power, true);
  invoke_test_exponential_filter_action<3, false>(alpha, half_power, true);

  test_exponential_filter_creation<1>();
  test_exponential_filter_creation<2>();
  test_exponential_filter_creation<3>();
}
