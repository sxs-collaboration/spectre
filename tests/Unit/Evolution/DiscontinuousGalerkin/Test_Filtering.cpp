// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "tests/Unit/TestingFramework.hpp"

#include <array>
#include <cmath>
#include <cstddef>
#include <string>

#include "DataStructures/DataBox/DataBox.hpp"
#include "DataStructures/DataBox/DataBoxTag.hpp"
#include "DataStructures/DataVector.hpp"  // IWYU pragma: keep
#include "DataStructures/Matrix.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "DataStructures/Variables.hpp"
#include "DataStructures/VariablesHelpers.hpp"
#include "Domain/Mesh.hpp"
#include "Domain/Tags.hpp"  // IWYU pragma: keep
#include "Evolution/DiscontinuousGalerkin/Filtering.hpp"
#include "NumericalAlgorithms/LinearOperators/ApplyMatrices.hpp"
#include "NumericalAlgorithms/Spectral/Filtering.hpp"
#include "NumericalAlgorithms/Spectral/Spectral.hpp"
#include "Parallel/ParallelComponentHelpers.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/Requires.hpp"
#include "Utilities/TMPL.hpp"
#include "Utilities/TaggedTuple.hpp"
#include "tests/Unit/ActionTesting.hpp"
#include "tests/Unit/TestCreation.hpp"

// IWYU pragma: no_forward_declare dg::Actions::ExponentialFilter

namespace {
namespace Tags {
struct ScalarVar : db::SimpleTag {
  static std::string name() noexcept { return "Scalar"; }
  using type = Scalar<DataVector>;
};

template <size_t Dim>
struct VectorVar : db::SimpleTag {
  static std::string name() noexcept { return "Vector"; }
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
  /// [action_list_example]
  using action_list = tmpl::conditional_t<
      metavariables::filter_individually,
      tmpl::list<
          dg::Actions::ExponentialFilter<0, tmpl::list<Tags::ScalarVar>>,
          dg::Actions::ExponentialFilter<1, tmpl::list<Tags::VectorVar<dim>>>>,
      tmpl::list<dg::Actions::ExponentialFilter<
          0, tmpl::list<Tags::VectorVar<dim>, Tags::ScalarVar>>>>;
  /// [action_list_example]
  using const_global_cache_tag_list =
      Parallel::get_const_global_cache_tags<action_list>;
  using simple_tags =
      db::AddSimpleTags<::Tags::Mesh<dim>,
                        typename metavariables::system::variables_tag>;
  using initial_databox = db::compute_databox_type<simple_tags>;
};

template <size_t Dim, bool FilterIndividually>
struct Metavariables {
  static constexpr bool filter_individually = FilterIndividually;

  using system = System<Dim>;
  static constexpr bool local_time_stepping = true;
  using component_list = tmpl::list<Component<Metavariables>>;
  using const_global_cache_tag_list = tmpl::list<>;
};

template <typename Metavariables,
          Requires<Metavariables::filter_individually> = nullptr>
typename ActionTesting::MockRuntimeSystem<Metavariables>::CacheTuple
create_cache_tuple(const double alpha, const unsigned half_power,
                   const bool disable_for_debugging) noexcept {
  constexpr size_t dim = Metavariables::system::volume_dim;
  return {dg::Actions::ExponentialFilter<0, tmpl::list<Tags::ScalarVar>>{
              alpha, half_power, disable_for_debugging},
          dg::Actions::ExponentialFilter<1, tmpl::list<Tags::VectorVar<dim>>>{
              2.0 * alpha, 2 * half_power, disable_for_debugging}};
}

template <typename Metavariables,
          Requires<not Metavariables::filter_individually> = nullptr>
typename ActionTesting::MockRuntimeSystem<Metavariables>::CacheTuple
create_cache_tuple(const double alpha, const unsigned half_power,
                   const bool disable_for_debugging) noexcept {
  constexpr size_t dim = Metavariables::system::volume_dim;
  return {dg::Actions::ExponentialFilter<
      0, tmpl::list<Tags::VectorVar<dim>, Tags::ScalarVar>>{
      alpha, half_power, disable_for_debugging}};
}

template <size_t Dim, Spectral::Basis BasisType,
          Spectral::Quadrature QuadratureType, bool FilterIndividually>
void test_exponential_filter_action(const double alpha,
                                    const unsigned half_power,
                                    const bool disable_for_debugging) noexcept {
  CAPTURE(BasisType);
  CAPTURE(QuadratureType);
  CAPTURE(disable_for_debugging);

  using metavariables = Metavariables<Dim, FilterIndividually>;
  using component = Component<metavariables>;
  using MockRuntimeSystem = ActionTesting::MockRuntimeSystem<metavariables>;
  using MockDistributedObjectsTag =
      typename MockRuntimeSystem::template MockDistributedObjectsTag<component>;

  for (size_t num_pts =
           Spectral::minimum_number_of_points<BasisType, QuadratureType>;
       num_pts < Spectral::maximum_number_of_points<BasisType>; ++num_pts) {
    CAPTURE(num_pts);
    const Mesh<Dim> mesh(num_pts, BasisType, QuadratureType);

    typename MockRuntimeSystem::TupleOfMockDistributedObjects dist_objects{};

    Variables<tmpl::list<Tags::ScalarVar, Tags::VectorVar<Dim>>> initial_vars(
        mesh.number_of_grid_points());
    for (size_t i = 0; i < mesh.number_of_grid_points(); ++i) {
      get(get<Tags::ScalarVar>(initial_vars))[i] = pow(i, num_pts) * 0.5;
      for (size_t d = 0; d < Dim; ++d) {
        get<Tags::VectorVar<Dim>>(initial_vars).get(d)[i] =
            d + pow(i, num_pts) * 0.75;
      }
    }

    tuples::get<MockDistributedObjectsTag>(dist_objects)
        .emplace(0, ActionTesting::MockDistributedObject<component>{
                        db::create<typename component::simple_tags>(
                            mesh, initial_vars)});

    MockRuntimeSystem runner(create_cache_tuple<metavariables>(
                                 alpha, half_power, disable_for_debugging),
                             std::move(dist_objects));

    auto& box =
        runner.template algorithms<component>()
            .at(0)
            .template get_databox<typename component::initial_databox>();

    runner.template next_action<component>(0);
    if (FilterIndividually) {
      runner.template next_action<component>(0);
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
    apply_matrices(&get(expected_scalar), filter_scalar,
                   get(get<Tags::ScalarVar>(initial_vars)), mesh.extents());
    for (size_t d = 0; d < Dim; d++) {
      apply_matrices(&expected_vector.get(d), filter_vector,
                     get<Tags::VectorVar<Dim>>(initial_vars).get(d),
                     mesh.extents());
    }
    CHECK_ITERABLE_APPROX(expected_scalar, db::get<Tags::ScalarVar>(box));
    CHECK_ITERABLE_APPROX(expected_vector, db::get<Tags::VectorVar<Dim>>(box));
  }
}

template <size_t Dim, bool FilterIndividually>
void invoke_test_exponential_filter_action(
    const double alpha, const unsigned half_power,
    const bool disable_for_debugging) noexcept {
  test_exponential_filter_action<1, Spectral::Basis::Legendre,
                                 Spectral::Quadrature::GaussLobatto,
                                 FilterIndividually>(alpha, half_power,
                                                     disable_for_debugging);
  test_exponential_filter_action<1, Spectral::Basis::Legendre,
                                 Spectral::Quadrature::Gauss,
                                 FilterIndividually>(alpha, half_power,
                                                     disable_for_debugging);
  test_exponential_filter_action<1, Spectral::Basis::Chebyshev,
                                 Spectral::Quadrature::GaussLobatto,
                                 FilterIndividually>(alpha, half_power,
                                                     disable_for_debugging);
  test_exponential_filter_action<1, Spectral::Basis::Chebyshev,
                                 Spectral::Quadrature::Gauss,
                                 FilterIndividually>(alpha, half_power,
                                                     disable_for_debugging);
}

template <size_t Dim>
void test_exponential_filter_creation() noexcept {
  using Filter = dg::Actions::ExponentialFilter<
      0, tmpl::list<Tags::ScalarVar, Tags::VectorVar<Dim>>>;

  const Filter filter = test_creation<Filter>(
      "  Alpha: 36\n"
      "  HalfPower: 32\n");

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

SPECTRE_TEST_CASE("Unit.Evolution.dG.ExponentialFilter", "[Unit][Evolution]") {
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
