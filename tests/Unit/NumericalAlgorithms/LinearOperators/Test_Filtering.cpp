// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <array>
#include <cmath>
#include <cstddef>
#include <memory>
#include <string>
#include <unordered_set>
#include <vector>

#include "DataStructures/ApplyMatrices.hpp"
#include "DataStructures/DataBox/DataBox.hpp"
#include "DataStructures/DataBox/Tag.hpp"
#include "DataStructures/DataVector.hpp"
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
#include "Domain/Tags.hpp"
#include "Evolution/Tags/Filter.hpp"
#include "Framework/ActionTesting.hpp"
#include "Framework/TestCreation.hpp"
#include "NumericalAlgorithms/LinearOperators/ExponentialFilter.hpp"
#include "NumericalAlgorithms/Spectral/Basis.hpp"
#include "NumericalAlgorithms/Spectral/Filtering.hpp"
#include "NumericalAlgorithms/Spectral/MaximumNumberOfPoints.hpp"
#include "NumericalAlgorithms/Spectral/Mesh.hpp"
#include "NumericalAlgorithms/Spectral/MinimumNumberOfPoints.hpp"
#include "NumericalAlgorithms/Spectral/Quadrature.hpp"
#include "Options/Protocols/FactoryCreation.hpp"
#include "Parallel/ParallelComponentHelpers.hpp"
#include "Parallel/Phase.hpp"
#include "Parallel/PhaseDependentActionList.hpp"
#include "ParallelAlgorithms/Actions/FilterAction.hpp"
#include "ParallelAlgorithms/Actions/InitializeItems.hpp"
#include "Utilities/GetOutput.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/ProtocolHelpers.hpp"
#include "Utilities/Serialization/RegisterDerivedClassesWithCharm.hpp"
#include "Utilities/StdArrayHelpers.hpp"
#include "Utilities/TMPL.hpp"

namespace {
// Blocks 0-2 do filtering (if enabled). Block 3 doesn't
constexpr size_t num_blocks = 4;
struct FilterEvolvedVariables {};
struct FilterScalarVariables {};
struct FilterVectorVariables {};

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
          tmpl::list<
              ActionTesting::InitializeDataBox<simple_tags>,
              Initialization::Actions::InitializeItems<tmpl::conditional_t<
                  metavariables::filter_individually,
                  tmpl::list<
                      dg::Actions::InitializeFilters<FilterScalarVariables>,
                      dg::Actions::InitializeFilters<FilterVectorVariables>>,
                  tmpl::list<dg::Actions::InitializeFilters<
                      FilterEvolvedVariables>>>>>>,
      // [action_list_example]
      Parallel::PhaseActions<
          Parallel::Phase::Testing,
          tmpl::conditional_t<
              metavariables::filter_individually,
              tmpl::list<dg::Actions::Filter<FilterScalarVariables,
                                             tmpl::list<Tags::ScalarVar>>,
                         dg::Actions::Filter<FilterVectorVariables,
                                             tmpl::list<Tags::VectorVar<dim>>>>,
              tmpl::list<dg::Actions::Filter<
                  FilterEvolvedVariables,
                  tmpl::list<Tags::VectorVar<dim>, Tags::ScalarVar>>>>>>;
  // [action_list_example]
  using simple_tags_from_options = Parallel::get_simple_tags_from_options<
      Parallel::get_initialization_actions_list<phase_dependent_action_list>>;
};

template <size_t Dim, bool FilterIndividually>
struct Metavariables {
  static constexpr bool filter_individually = FilterIndividually;
  static constexpr size_t dim = Dim;

  using system = System<Dim>;
  static constexpr bool local_time_stepping = true;
  struct factory_creation
      : tt::ConformsTo<Options::protocols::FactoryCreation> {
    using factory_classes = tmpl::map<
        tmpl::pair<Filters::Filter, tmpl::list<Filters::Exponential<Dim>>>>;
  };
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

template <typename Metavariables>
auto create_filters(const double alpha, const unsigned half_power,
                    const bool enable) {
  constexpr size_t dim = Metavariables::system::volume_dim;
  if constexpr (Metavariables::filter_individually) {
    if (not enable) {
      return std::make_tuple(std::vector<std::unique_ptr<Filters::Filter>>{},
                             std::vector<std::unique_ptr<Filters::Filter>>{});
    }
    std::vector<std::unique_ptr<Filters::Filter>> scalar_filters{};
    scalar_filters.emplace_back(std::make_unique<Filters::Exponential<dim>>(
        alpha, half_power, get_block_names()));
    std::vector<std::unique_ptr<Filters::Filter>> vector_filters{};
    vector_filters.emplace_back(std::make_unique<Filters::Exponential<dim>>(
        2.0 * alpha, 2 * half_power, get_block_names()));
    return std::make_tuple(std::move(scalar_filters),
                           std::move(vector_filters));
  } else {
    std::vector<std::unique_ptr<Filters::Filter>> filters{};
    if (enable) {
      filters.emplace_back(std::make_unique<Filters::Exponential<dim>>(
          alpha, half_power, get_block_names()));
    }
    return std::make_tuple(std::move(filters));
  }
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
  register_factory_classes_with_charm<metavariables>();

  // Division by Dim to reduce time of test
  const size_t max_pts =
      BasisType == Spectral::Basis::Fourier
          ? Spectral::maximum_number_of_points<BasisType> / (3 * Dim)
          : Spectral::maximum_number_of_points<BasisType> / Dim;
  for (size_t num_pts =
           Spectral::minimum_number_of_points<BasisType, QuadratureType>;
       num_pts < max_pts; ++num_pts) {
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

    ActionTesting::MockRuntimeSystem<metavariables> runner{
        {make_domain<Dim>()}};
    for (size_t block = 0; block < num_blocks; block++) {
      auto filters = create_filters<metavariables>(alpha, half_power, enable);
      std::apply(
          [&runner, &mesh, &initial_vars, block](auto&&... local_filters) {
            ActionTesting::emplace_component_and_initialize<component>(
                &runner, block,
                {mesh, Element{ElementId<Dim>{block}, {}}, initial_vars},
                std::move(local_filters)...);
          },
          std::move(filters));
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
  test_exponential_filter_action<Dim, Spectral::Basis::Fourier,
                                 Spectral::Quadrature::Equiangular,
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
  struct factory_creation
      : tt::ConformsTo<Options::protocols::FactoryCreation> {
    using factory_classes = tmpl::map<
        tmpl::pair<Filters::Filter, tmpl::list<Filters::Exponential<Dim>>>,
        tmpl::pair<::DomainCreator<Dim>, tmpl::list<TestCreator<Dim>>>>;
  };
};

template <size_t Dim>
void test_exponential_filter_creation() {
  using Filter = Filters::Exponential<Dim>;
  using AnotherFilter = Filters::Exponential<Dim>;
  register_factory_classes_with_charm<Metavars<Dim>>();

  using tags = tmpl::list<OptionTags::FilterList<FilterScalarVariables>,
                          OptionTags::FilterList<FilterVectorVariables>,
                          domain::OptionTags::DomainCreator<Dim>>;
  Options::Parser<tags> options("");
  options.parse(
      "DomainCreator:\n"
      "  TestCreator\n"
      // [multiple_exponential_filters]
      "Filtering:\n"
      "  FilterScalarVariables:\n"
      "    - ExponentialFilter:\n"
      "        Alpha: 36\n"
      "        HalfPower: 32\n"
      "        BlocksToFilter: All\n"
      "  FilterVectorVariables:\n"
      "    - ExponentialFilter:\n"
      "        Alpha: 36\n"
      "        HalfPower: 12\n"
      "        BlocksToFilter:\n"
      "          - Block0\n"
      "          - Group1\n"
      // [multiple_exponential_filters]
  );
  const auto& filters =
      options.template get<OptionTags::FilterList<FilterScalarVariables>,
                           Metavars<Dim>>();
  REQUIRE(filters.size() == 1);
  const auto* filter = dynamic_cast<const Filter*>(filters[0].get());
  REQUIRE(filter != nullptr);
  CHECK(*filter == Filter{36.0, 32, {}});

  const auto& another_filters =
      options.template get<OptionTags::FilterList<FilterVectorVariables>,
                           Metavars<Dim>>();
  REQUIRE(another_filters.size() == 1);
  const auto* another_filter =
      dynamic_cast<const AnotherFilter*>(another_filters[0].get());
  REQUIRE(another_filter != nullptr);
  CHECK(*another_filter == AnotherFilter{36.0, 12, {{"Block0", "Group1"}}});

  {
    Options::Parser<tmpl::list<OptionTags::FilterList<FilterVectorVariables>,
                               domain::OptionTags::DomainCreator<Dim>>>
        error_options("");
    error_options.parse(
        "DomainCreator:\n"
        "  TestCreator\n"
        "Filtering:\n"
        "  FilterVectorVariables:\n"
        "    - ExponentialFilter:\n"
        "        Alpha: 36\n"
        "        HalfPower: 12\n"
        "        BlocksToFilter:\n"
        "          - Block0\n"
        "          - Block0\n");

    CHECK_THROWS_WITH(
        (error_options.template get<
            OptionTags::FilterList<FilterVectorVariables>, Metavars<Dim>>()),
        Catch::Matchers::ContainsSubstring("Duplicate block name"));

    std::vector<std::unique_ptr<Filters::Filter>> invalid_filter{};
    invalid_filter.emplace_back(std::make_unique<AnotherFilter>(
        36.0, 12, std::vector<std::string>{"BlockGroup1"}));
    CHECK_THROWS_WITH(
        (Filters::Tags::FilterList<FilterVectorVariables>::
             template create_from_options<Metavars<Dim>>(
                 invalid_filter, std::make_unique<TestCreator<Dim>>())),
        Catch::Matchers::ContainsSubstring(
            "is not a block name or a block group. Existing blocks are"));

    // These two checks ensure both block names and block groups validate.
    std::vector<std::unique_ptr<Filters::Filter>> block_name_filter{};
    block_name_filter.emplace_back(
        std::make_unique<Filter>(26.0, 23, std::vector<std::string>{"Block0"}));
    CHECK_NOTHROW(
        (Filters::Tags::FilterList<FilterScalarVariables>::
             template create_from_options<Metavars<Dim>>(
                 block_name_filter, std::make_unique<TestCreator<Dim>>())));

    std::vector<std::unique_ptr<Filters::Filter>> block_group_filter{};
    block_group_filter.emplace_back(
        std::make_unique<Filter>(26.0, 23, std::vector<std::string>{"Group1"}));
    CHECK_NOTHROW(
        (Filters::Tags::FilterList<FilterScalarVariables>::
             template create_from_options<Metavars<Dim>>(
                 block_group_filter, std::make_unique<TestCreator<Dim>>())));

    CHECK_THROWS_WITH(
        (Filters::Tags::FilterList<FilterVectorVariables>::
             template create_from_options<Metavars<Dim>>(
                 another_filters, std::make_unique<TestCreator<Dim>>(false))),
        Catch::Matchers::ContainsSubstring(
            "The domain chosen doesn't use block names"));
  }
}

void test_cartoon_exponential_filter() {
  const auto filter = Filters::Exponential<1>{0, 0, {}};
  CHECK(filter.filter_matrix({1, Spectral::Basis::Cartoon,
                              Spectral::Quadrature::AxialSymmetry}) ==
        Matrix(1, 1, 1.0));
  CHECK(filter.filter_matrix({1, Spectral::Basis::Cartoon,
                              Spectral::Quadrature::SphericalSymmetry}) ==
        Matrix(1, 1, 1.0));
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

  test_cartoon_exponential_filter();
}
