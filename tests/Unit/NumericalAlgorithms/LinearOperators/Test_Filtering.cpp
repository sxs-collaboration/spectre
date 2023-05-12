// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <array>
#include <cmath>
#include <cstddef>
#include <string>
#include <unordered_set>

#include "DataStructures/ApplyMatrices.hpp"
#include "DataStructures/DataBox/DataBox.hpp"
#include "DataStructures/DataBox/Tag.hpp"
#include "DataStructures/DataVector.hpp"  // IWYU pragma: keep
#include "DataStructures/Matrix.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "DataStructures/Variables.hpp"
#include "DataStructures/VariablesTag.hpp"
#include "Domain/CoordinateMaps/CoordinateMap.hpp"
#include "Domain/CoordinateMaps/CoordinateMap.tpp"
#include "Domain/CoordinateMaps/Identity.hpp"
#include "Domain/Creators/DomainCreator.hpp"
#include "Domain/Creators/Tags/Domain.hpp"
#include "Domain/Structure/Element.hpp"
#include "Domain/Tags.hpp"  // IWYU pragma: keep
#include "Evolution/Tags/Filter.hpp"
#include "Framework/ActionTesting.hpp"
#include "Framework/TestCreation.hpp"
#include "NumericalAlgorithms/LinearOperators/ExponentialFilter.hpp"
#include "NumericalAlgorithms/LinearOperators/FilterAction.hpp"
#include "NumericalAlgorithms/Spectral/Filtering.hpp"
#include "NumericalAlgorithms/Spectral/Mesh.hpp"
#include "NumericalAlgorithms/Spectral/Spectral.hpp"
#include "Parallel/ParallelComponentHelpers.hpp"
#include "Parallel/Phase.hpp"
#include "Parallel/PhaseDependentActionList.hpp"  // IWYU pragma: keep
#include "Utilities/GetOutput.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/Requires.hpp"
#include "Utilities/Serialization/RegisterDerivedClassesWithCharm.hpp"
#include "Utilities/StdArrayHelpers.hpp"
#include "Utilities/TMPL.hpp"

// IWYU pragma: no_forward_declare ActionTesting::InitializeDataBox
// IWYU pragma: no_forward_declare dg::Actions::ExponentialFilter

namespace {
// Blocks 0-2 do filtering (if enabled). Block 3 doesn't
constexpr size_t num_blocks = 4;

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
  using const_global_cache_tags = tmpl::list<domain::Tags::Domain<dim>>;
  using simple_tags =
      db::AddSimpleTags<domain::Tags::Mesh<dim>, domain::Tags::Element<dim>,
                        typename metavariables::system::variables_tag>;

  using phase_dependent_action_list = tmpl::list<
      Parallel::PhaseActions<
          Parallel::Phase::Initialization,
          tmpl::list<ActionTesting::InitializeDataBox<simple_tags>>>,
      // [action_list_example]
      Parallel::PhaseActions<
          Parallel::Phase::Testing,
          tmpl::conditional_t<
              metavariables::filter_individually,
              tmpl::list<dg::Actions::Filter<Filters::Exponential<0>,
                                             tmpl::list<Tags::ScalarVar>>,
                         dg::Actions::Filter<Filters::Exponential<1>,
                                             tmpl::list<Tags::VectorVar<dim>>>>,
              tmpl::list<dg::Actions::Filter<
                  Filters::Exponential<0>,
                  tmpl::list<Tags::VectorVar<dim>, Tags::ScalarVar>>>>>>;
  // [action_list_example]
};

template <size_t Dim, bool FilterIndividually>
struct Metavariables {
  static constexpr bool filter_individually = FilterIndividually;
  static constexpr size_t dim = Dim;

  using system = System<Dim>;
  static constexpr bool local_time_stepping = true;
  using component_list = tmpl::list<Component<Metavariables>>;
};

std::vector<std::string> domain_block_names() {
  std::vector<std::string> block_names{num_blocks};
  for (size_t i = 0; i < num_blocks; i++) {
    block_names[i] = "Block" + get_output(i);
  }

  return block_names;
}

std::unordered_map<std::string, std::unordered_set<std::string>>
domain_block_groups() {
  std::unordered_map<std::string, std::unordered_set<std::string>> groups{};
  groups["Group1"] = std::unordered_set<std::string>{{"Block1"s}};
  groups["Group2"] = std::unordered_set<std::string>{{"Block1"s}, {"Block2"s}};

  return groups;
}

template <size_t Dim>
Domain<Dim> make_domain() {
  using Identity = domain::CoordinateMaps::Identity<Dim>;
  using Map =
      domain::CoordinateMap<Frame::BlockLogical, Frame::Inertial, Identity>;
  register_classes_with_charm(tmpl::list<Map>{});
  std::vector<std::unique_ptr<
      domain::CoordinateMapBase<Frame::BlockLogical, Frame::Inertial, Dim>>>
      maps{num_blocks};
  for (size_t i = 0; i < num_blocks; i++) {
    maps[i] = std::make_unique<Map>(Identity{});
  }

  return Domain<Dim>{
      std::move(maps), {}, domain_block_names(), domain_block_groups()};
}

std::optional<std::vector<std::string>> get_block_names() {
  std::optional<std::vector<std::string>> names{
      {{"Block0"}, {"Group1"}, {"Group2"}}};
  return names;
}

template <typename Metavariables,
          Requires<Metavariables::filter_individually> = nullptr>
typename ActionTesting::MockRuntimeSystem<Metavariables>::CacheTuple
create_cache_tuple(const double alpha, const unsigned half_power,
                   const bool enable) {
  return {make_domain<Metavariables::dim>(),
          Filters::Exponential<0>{alpha, half_power, enable, get_block_names()},
          Filters::Exponential<1>{2.0 * alpha, 2 * half_power, enable,
                                  get_block_names()}};
}

template <typename Metavariables,
          Requires<not Metavariables::filter_individually> = nullptr>
typename ActionTesting::MockRuntimeSystem<Metavariables>::CacheTuple
create_cache_tuple(const double alpha, const unsigned half_power,
                   const bool enable) {
  return {
      make_domain<Metavariables::dim>(),
      Filters::Exponential<0>{alpha, half_power, enable, get_block_names()}};
}

template <size_t Dim, Spectral::Basis BasisType,
          Spectral::Quadrature QuadratureType, bool FilterIndividually>
void test_exponential_filter_action(const double alpha,
                                    const unsigned half_power,
                                    const bool enable) {
  CAPTURE(BasisType);
  CAPTURE(QuadratureType);
  CAPTURE(enable);

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
        create_cache_tuple<metavariables>(alpha, half_power, enable));
    for (size_t block = 0; block < num_blocks; block++) {
      ActionTesting::emplace_component_and_initialize<component>(
          &runner, block,
          {mesh, Element{ElementId<Dim>{block}, {}}, initial_vars});
    }
    ActionTesting::set_phase(make_not_null(&runner), Parallel::Phase::Testing);

    for (size_t block = 0; block < num_blocks; block++) {
      CAPTURE(block);
      ActionTesting::next_action<component>(make_not_null(&runner), block);
      if (FilterIndividually) {
        ActionTesting::next_action<component>(make_not_null(&runner), block);
      }

      std::array<Matrix, Dim> filter_scalar{};
      std::array<Matrix, Dim> filter_vector{};
      for (size_t d = 0; d < Dim; d++) {
        if (enable and block < num_blocks - 1) {
          gsl::at(filter_scalar, d) = Spectral::filtering::exponential_filter(
              mesh.slice_through(d), alpha, half_power);
          if (FilterIndividually) {
            gsl::at(filter_vector, d) = Spectral::filtering::exponential_filter(
                mesh.slice_through(d), 2.0 * alpha, 2 * half_power);
          } else {
            gsl::at(filter_vector, d) = gsl::at(filter_scalar, d);
          }
        } else {
          gsl::at(filter_scalar, d) = Matrix{};
          gsl::at(filter_vector, d) = Matrix{};
        }
      }

      Scalar<DataVector> expected_scalar(mesh.number_of_grid_points(), 0.0);
      tnsr::I<DataVector, Dim> expected_vector(mesh.number_of_grid_points(),
                                               0.0);
      apply_matrices(make_not_null(&get(expected_scalar)), filter_scalar,
                     get(get<Tags::ScalarVar>(initial_vars)), mesh.extents());
      for (size_t d = 0; d < Dim; d++) {
        apply_matrices(make_not_null(&expected_vector.get(d)), filter_vector,
                       get<Tags::VectorVar<Dim>>(initial_vars).get(d),
                       mesh.extents());
      }
      CHECK_ITERABLE_CUSTOM_APPROX(
          expected_scalar,
          (ActionTesting::get_databox_tag<component, Tags::ScalarVar>(runner,
                                                                      block)),
          custom_approx);
      CHECK_ITERABLE_CUSTOM_APPROX(
          expected_vector,
          (ActionTesting::get_databox_tag<component, Tags::VectorVar<Dim>>(
              runner, block)),
          custom_approx);
    }
  }
}

template <size_t Dim, bool FilterIndividually>
void invoke_test_exponential_filter_action(const double alpha,
                                           const unsigned half_power,
                                           const bool enable) {
  test_exponential_filter_action<Dim, Spectral::Basis::Legendre,
                                 Spectral::Quadrature::GaussLobatto,
                                 FilterIndividually>(alpha, half_power, enable);
  test_exponential_filter_action<Dim, Spectral::Basis::Legendre,
                                 Spectral::Quadrature::Gauss,
                                 FilterIndividually>(alpha, half_power, enable);
  test_exponential_filter_action<Dim, Spectral::Basis::Chebyshev,
                                 Spectral::Quadrature::GaussLobatto,
                                 FilterIndividually>(alpha, half_power, enable);
  test_exponential_filter_action<Dim, Spectral::Basis::Chebyshev,
                                 Spectral::Quadrature::Gauss,
                                 FilterIndividually>(alpha, half_power, enable);
}

template <size_t Dim>
class TestCreator : public DomainCreator<Dim> {
 public:
  TestCreator(const bool use_block_names = true)
      : use_block_names_(use_block_names) {}

  Domain<Dim> create_domain() const override { return make_domain<Dim>(); }
  std::vector<std::string> block_names() const override {
    return use_block_names_ ? domain_block_names() : std::vector<std::string>{};
  }
  std::unordered_map<std::string, std::unordered_set<std::string>>
  block_groups() const override {
    return use_block_names_
               ? domain_block_groups()
               : std::unordered_map<std::string,
                                    std::unordered_set<std::string>>{};
  }
  std::vector<DirectionMap<
      Dim, std::unique_ptr<domain::BoundaryConditions::BoundaryCondition>>>
  external_boundary_conditions() const override {
    ERROR("");
  }
  std::vector<std::array<size_t, Dim>> initial_extents() const override {
    ERROR("");
  }
  std::vector<std::array<size_t, Dim>> initial_refinement_levels()
      const override {
    ERROR("");
  }
  auto functions_of_time(const std::unordered_map<std::string, double>&
                         /*initial_expiration_times*/
                         = {}) const
      -> std::unordered_map<
          std::string,
          std::unique_ptr<domain::FunctionsOfTime::FunctionOfTime>> override {
    ERROR("");
  }

 private:
  bool use_block_names_;
};

template <size_t Dim>
struct Metavars {
  static constexpr size_t volume_dim = Dim;
  struct factory_creation {
    using factory_classes = tmpl::map<
        tmpl::pair<::DomainCreator<Dim>, tmpl::list<TestCreator<Dim>>>>;
  };
};

template <size_t Dim>
void test_exponential_filter_creation() {
  using Filter = Filters::Exponential<0>;
  using AnotherFilter = Filters::Exponential<1>;

  using tags =
      tmpl::list<OptionTags::Filter<Filter>, OptionTags::Filter<AnotherFilter>,
                 domain::OptionTags::DomainCreator<Dim>>;
  Options::Parser<tags> options("");
  options.parse(
      "DomainCreator:\n"
      "  TestCreator\n"
      // [multiple_exponential_filters]
      "Filtering:\n"
      "  ExpFilter0:\n"
      "    Alpha: 36\n"
      "    HalfPower: 32\n"
      "    Enable: True\n"
      "    BlocksToFilter: All\n"
      "  ExpFilter1:\n"
      "    Alpha: 36\n"
      "    HalfPower: 12\n"
      "    Enable: True\n"
      "    BlocksToFilter:\n"
      "      - Block0\n"
      "      - BlockGroup1\n"
      // [multiple_exponential_filters]
  );
  const auto filter =
      options.template get<OptionTags::Filter<Filter>, Metavars<Dim>>();

  CHECK(filter == Filter{36.0, 32, true, {}});
  CHECK_FALSE(filter == Filter{35.0, 32, true, {}});
  CHECK_FALSE(filter == Filter{36.0, 33, true, {}});
  CHECK_FALSE(filter == Filter{36.0, 32, false, {}});
  CHECK_FALSE(filter == Filter{36.0, 32, true, {{"Block0"}}});

  CHECK_FALSE(filter != Filter{36.0, 32, true, {}});
  CHECK(filter != Filter{35.0, 32, true, {}});
  CHECK(filter != Filter{36.0, 33, true, {}});
  CHECK(filter != Filter{36.0, 32, false, {}});
  CHECK(filter != Filter{36.0, 32, true, {{"Block0"}}});

  const auto another_filter =
      options.template get<OptionTags::Filter<AnotherFilter>, Metavars<Dim>>();

  CHECK(another_filter ==
        AnotherFilter{36.0, 12, true, {{"Block0", "BlockGroup1"}}});
  CHECK_FALSE(another_filter !=
              AnotherFilter{36.0, 12, true, {{"Block0", "BlockGroup1"}}});

  CHECK_FALSE(another_filter == AnotherFilter{36.0, 12, true, {}});
  CHECK(another_filter != AnotherFilter{36.0, 12, true, {}});

  {
    Options::Parser<tmpl::pop_front<tags>> error_options("");
    error_options.parse(
        "DomainCreator:\n"
        "  TestCreator\n"
        "Filtering:\n"
        "  ExpFilter1:\n"
        "    Alpha: 36\n"
        "    HalfPower: 12\n"
        "    Enable: True\n"
        "    BlocksToFilter:\n"
        "      - Block0\n"
        "      - Block0\n");

    CHECK_THROWS_WITH(
        (error_options
             .template get<OptionTags::Filter<AnotherFilter>, Metavars<Dim>>()),
        Catch::Contains("Duplicate block name"));

    CHECK_THROWS_WITH(
        (Filters::Tags::Filter<AnotherFilter>::create_from_options<
            Metavars<Dim>>(another_filter,
                           std::make_unique<TestCreator<Dim>>())),
        Catch::Contains(
            "is not a block name or a block group. Existing blocks are"));

    // These two are to check that we can pass just a block name or just a block
    // group and the tag will create things correctly
    CHECK_NOTHROW(
        (Filters::Tags::Filter<Filter>::create_from_options<Metavars<Dim>>(
            Filter{26.0, 23, true, {{"Block0"}}},
            std::make_unique<TestCreator<Dim>>())));
    CHECK_NOTHROW(
        (Filters::Tags::Filter<Filter>::create_from_options<Metavars<Dim>>(
            Filter{26.0, 23, true, {{"Group1"}}},
            std::make_unique<TestCreator<Dim>>())));

    CHECK_THROWS_WITH(
        (Filters::Tags::Filter<AnotherFilter>::create_from_options<
            Metavars<Dim>>(another_filter,
                           std::make_unique<TestCreator<Dim>>(false))),
        Catch::Contains("The domain chosen doesn't use block names"));
  }
}
}  // namespace

SPECTRE_TEST_CASE("Unit.Numerical.LinearOperators.Filter",
                  "[NumericalAlgorithms][LinearOperators][Unit]") {
  // Can't do a loop over different alpha and half_power values because matrices
  // are cached in the action.
  const double alpha = 10.0;
  const unsigned half_power = 16;
  tmpl::for_each<tmpl::make_sequence<tmpl::size_t<1>, 3>>(
      [&alpha, &half_power](auto dim_v) {
        constexpr size_t Dim = tmpl::type_from<decltype(dim_v)>::value;
        for (const bool enable : make_array(true, false)) {
          invoke_test_exponential_filter_action<Dim, true>(alpha, half_power,
                                                           enable);
          invoke_test_exponential_filter_action<Dim, false>(alpha, half_power,
                                                            enable);
        }
      });

  test_exponential_filter_creation<1>();
  test_exponential_filter_creation<2>();
  test_exponential_filter_creation<3>();
}
