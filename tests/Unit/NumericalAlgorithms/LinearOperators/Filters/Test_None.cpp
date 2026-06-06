// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <cstddef>
#include <memory>
#include <optional>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "DataStructures/DataBox/Tag.hpp"
#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "DataStructures/Variables.hpp"
#include "Domain/CoordinateMaps/CoordinateMap.hpp"
#include "Domain/CoordinateMaps/CoordinateMap.tpp"
#include "Domain/CoordinateMaps/Identity.hpp"
#include "Domain/Creators/DomainCreator.hpp"
#include "Domain/Creators/OptionTags.hpp"
#include "Domain/Creators/Tags/Domain.hpp"
#include "Domain/Domain.hpp"
#include "Evolution/Tags/Filter.hpp"
#include "Framework/TestCreation.hpp"
#include "Framework/TestHelpers.hpp"
#include "NumericalAlgorithms/LinearOperators/Filters/Factory.hpp"
#include "NumericalAlgorithms/LinearOperators/Filters/Filter.hpp"
#include "NumericalAlgorithms/LinearOperators/Filters/Hypercube.hpp"
#include "NumericalAlgorithms/LinearOperators/Filters/Hypercube.tpp"
#include "NumericalAlgorithms/LinearOperators/Filters/None.hpp"
#include "NumericalAlgorithms/LinearOperators/Filters/None.tpp"
#include "NumericalAlgorithms/Spectral/Basis.hpp"
#include "NumericalAlgorithms/Spectral/Mesh.hpp"
#include "NumericalAlgorithms/Spectral/Quadrature.hpp"
#include "Utilities/GetOutput.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/Serialization/RegisterDerivedClassesWithCharm.hpp"
#include "Utilities/TMPL.hpp"

namespace {
using namespace std::string_literals;

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
using TagList = tmpl::list<Tags::ScalarVar, Tags::VectorVar<Dim>>;

template <size_t Dim>
using NoneFilter = Filters::None<Dim, TagList<Dim>>;

std::vector<std::string> domain_block_names() {
  std::vector<std::string> block_names(num_blocks);
  for (size_t i = 0; i < num_blocks; ++i) {
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
  for (size_t i = 0; i < num_blocks; ++i) {
    maps[i] = std::make_unique<Map>(Identity{});
  }
  return Domain<Dim>{
      std::move(maps), {}, domain_block_names(), domain_block_groups()};
}

template <size_t Dim>
class TestCreator : public DomainCreator<Dim> {
 public:
  explicit TestCreator(const bool use_block_names = true)
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
    ERROR("TestCreator does not implement external_boundary_conditions");
  }
  std::vector<std::array<size_t, Dim>> initial_extents() const override {
    ERROR("TestCreator does not implement initial_extents");
  }
  std::vector<std::array<size_t, Dim>> initial_refinement_levels()
      const override {
    ERROR("TestCreator does not implement initial_refinement_levels");
  }
  auto functions_of_time(const std::unordered_map<std::string, double>&
                         /*initial_expiration_times*/
                         = {}) const
      -> std::unordered_map<
          std::string,
          std::unique_ptr<domain::FunctionsOfTime::FunctionOfTime>> override {
    ERROR("TestCreator does not implement functions_of_time");
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
void test_construction_and_accessors() {
  INFO("Construction and accessors");
  CAPTURE(Dim);

  // Default-constructed: all blocks, no restrictions.
  const NoneFilter<Dim> default_filter{};
  CHECK_FALSE(default_filter.need_jacobians());
  CHECK_FALSE(default_filter.blocks_to_filter().has_value());

  // All cadence predicates always return false.
  CHECK_FALSE(default_filter.apply_volume_filter_on_substep());
  CHECK_FALSE(default_filter.apply_boundary_filter_on_substep());
  for (const size_t step : {size_t{0}, size_t{1}, size_t{7}, size_t{100}}) {
    CHECK_FALSE(default_filter.apply_volume_filter_on_this_step(step));
    CHECK_FALSE(default_filter.apply_boundary_filter_on_this_step(step));
  }

  // supports_mesh returns true for any basis/quadrature combination.
  CHECK(default_filter.supports_mesh(
      Mesh<Dim>{3, Spectral::Basis::Legendre, Spectral::Quadrature::Gauss}));
  CHECK(default_filter.supports_mesh(Mesh<Dim>{
      3, Spectral::Basis::Legendre, Spectral::Quadrature::GaussLobatto}));
  CHECK(default_filter.supports_mesh(
      Mesh<Dim>{3, Spectral::Basis::Chebyshev, Spectral::Quadrature::Gauss}));
  CHECK(default_filter.supports_mesh(Mesh<Dim>{
      3, Spectral::Basis::Chebyshev, Spectral::Quadrature::GaussLobatto}));
  CHECK(default_filter.supports_mesh(Mesh<Dim>{
      4, Spectral::Basis::Fourier, Spectral::Quadrature::Equiangular}));
  // FiniteDifference is not supported by Hypercube, but None accepts it.
  CHECK(default_filter.supports_mesh(
      Mesh<Dim>{3, Spectral::Basis::FiniteDifference,
                Spectral::Quadrature::CellCentered}));

  // With a block list: resolved ids after set_blocks_to_filter.
  const std::vector<std::string> block_strings{"Block0", "Group1"};
  const NoneFilter<Dim> restricted_filter{
      std::optional<std::vector<std::string>>{block_strings}};
  // Before resolution the ids are not yet available.
  CHECK_FALSE(restricted_filter.blocks_to_filter().has_value());

  auto resolved = restricted_filter;
  resolved.set_blocks_to_filter(domain_block_names(), domain_block_groups());
  REQUIRE(resolved.blocks_to_filter().has_value());
  // NOLINTBEGIN(bugprone-unchecked-optional-access)
  CHECK(resolved.blocks_to_filter().value() == std::vector<size_t>{0, 1});
  // NOLINTEND(bugprone-unchecked-optional-access)

  // Without a block list: set_blocks_to_filter leaves it as nullopt.
  auto unrestricted = default_filter;
  unrestricted.set_blocks_to_filter(domain_block_names(),
                                    domain_block_groups());
  CHECK_FALSE(unrestricted.blocks_to_filter().has_value());

  // Equality: same blocks_and_groups means equal.
  const NoneFilter<Dim> filter_a{
      std::optional<std::vector<std::string>>{{"Block0", "Block1"}}};
  const NoneFilter<Dim> filter_b{
      std::optional<std::vector<std::string>>{{"Block0", "Block1"}}};
  const NoneFilter<Dim> filter_c{
      std::optional<std::vector<std::string>>{{"Block0"}}};
  CHECK(filter_a == filter_b);
  CHECK_FALSE(filter_a != filter_b);
  CHECK(filter_a != filter_c);
  CHECK(filter_a != default_filter);

  // Duplicate block names rejected at construction.
  CHECK_THROWS_WITH((NoneFilter<Dim>{std::optional<std::vector<std::string>>{
                        {"Block0", "Block0"}}}),
                    Catch::Matchers::ContainsSubstring("Duplicate block name"));
}

template <size_t Dim>
void test_pup_round_trip() {
  INFO("Serialization");
  CAPTURE(Dim);
  const NoneFilter<Dim> filter{
      std::optional<std::vector<std::string>>{{"Block0", "Group2"}}};
  ::test_serialization(filter);

  // Round-trip through the abstract base pointer.
  using Base = Filters::Filter<Dim, TagList<Dim>>;
  using Derived = NoneFilter<Dim>;
  register_classes_with_charm<Derived>();
  const std::unique_ptr<Base> base = std::make_unique<Derived>(filter);
  const std::unique_ptr<Base> pupped_base = serialize_and_deserialize(base);
  REQUIRE(dynamic_cast<const Derived*>(pupped_base.get()) != nullptr);
  CHECK(dynamic_cast<const Derived&>(*pupped_base) == filter);
}

template <size_t Dim>
void test_apply_noop() {
  INFO("apply_in_volume and apply_on_boundary are no-ops");
  CAPTURE(Dim);
  const NoneFilter<Dim> filter{};

  const Mesh<Dim> vol_mesh{3, Spectral::Basis::Legendre,
                           Spectral::Quadrature::GaussLobatto};
  Variables<TagList<Dim>> vol_vars(vol_mesh.number_of_grid_points());
  for (size_t i = 0; i < vol_mesh.number_of_grid_points(); ++i) {
    get(get<Tags::ScalarVar>(vol_vars))[i] = static_cast<double>(i) + 1.0;
    for (size_t d = 0; d < Dim; ++d) {
      get<Tags::VectorVar<Dim>>(vol_vars).get(d)[i] =
          static_cast<double>(d * 10 + i) + 0.5;
    }
  }
  const auto initial_vol_vars = vol_vars;
  filter.apply_in_volume(make_not_null(&vol_vars), vol_mesh, std::nullopt,
                         std::nullopt);
  CHECK(vol_vars == initial_vol_vars);

  if constexpr (Dim >= 2) {
    const Mesh<Dim - 1> face_mesh = vol_mesh.slice_away(0);
    Variables<TagList<Dim>> face_vars(face_mesh.number_of_grid_points());
    for (size_t i = 0; i < face_mesh.number_of_grid_points(); ++i) {
      get(get<Tags::ScalarVar>(face_vars))[i] = static_cast<double>(i) + 2.0;
      for (size_t d = 0; d < Dim; ++d) {
        get<Tags::VectorVar<Dim>>(face_vars).get(d)[i] =
            static_cast<double>(d * 5 + i) + 0.25;
      }
    }
    const auto initial_face_vars = face_vars;
    filter.apply_on_boundary(make_not_null(&face_vars), face_mesh, std::nullopt,
                             std::nullopt);
    CHECK(face_vars == initial_face_vars);
  }
}

template <size_t Dim>
void test_is_equal() {
  INFO("is_equal");
  CAPTURE(Dim);
  using Base = Filters::Filter<Dim, TagList<Dim>>;
  const std::optional<std::vector<std::string>> blocks{
      std::vector<std::string>{"Block0", "Block1"}};

  const NoneFilter<Dim> a{blocks};
  const NoneFilter<Dim> b{blocks};
  const NoneFilter<Dim> c{std::nullopt};

  CHECK(a.is_equal(b));
  CHECK(b.is_equal(a));
  CHECK_FALSE(a.is_equal(c));

  // Different concrete type returns false.
  const Filters::Hypercube<Dim, TagList<Dim>> hypercube_filter{
      4, true, std::nullopt, false, false, std::nullopt, std::nullopt};
  CHECK_FALSE(a.is_equal(hypercube_filter));

  // Via abstract base pointer (the primary AMR use case).
  const std::unique_ptr<Base> pa = std::make_unique<NoneFilter<Dim>>(a);
  const std::unique_ptr<Base> pb = std::make_unique<NoneFilter<Dim>>(b);
  const std::unique_ptr<Base> pc = std::make_unique<NoneFilter<Dim>>(c);
  const std::unique_ptr<Base> phyper =
      std::make_unique<Filters::Hypercube<Dim, TagList<Dim>>>(hypercube_filter);
  CHECK(pa->is_equal(*pb));
  CHECK_FALSE(pa->is_equal(*pc));
  CHECK_FALSE(pa->is_equal(*phyper));
}

template <size_t Dim>
void test_option_parsing() {
  INFO("Option parsing");
  CAPTURE(Dim);
  using Filter = NoneFilter<Dim>;
  using tags = tmpl::list<OptionTags::Filter<Filter>,
                          domain::OptionTags::DomainCreator<Dim>>;

  // BlocksToFilter: All -> nullopt.
  Options::Parser<tags> all_parser("");
  all_parser.parse(
      "DomainCreator:\n"
      "  TestCreator\n"
      "Filtering:\n"
      "  None:\n"
      "    BlocksToFilter: All\n");
  const auto all_parsed =
      all_parser.template get<OptionTags::Filter<Filter>, Metavars<Dim>>();
  CHECK_FALSE(all_parsed.blocks_to_filter().has_value());

  // All cadence predicates still false after parsing.
  CHECK_FALSE(all_parsed.apply_volume_filter_on_substep());
  CHECK_FALSE(all_parsed.apply_volume_filter_on_this_step(0));
  CHECK_FALSE(all_parsed.apply_boundary_filter_on_substep());
  CHECK_FALSE(all_parsed.apply_boundary_filter_on_this_step(0));

  // Explicit block list.
  Options::Parser<tags> explicit_parser("");
  explicit_parser.parse(
      "DomainCreator:\n"
      "  TestCreator\n"
      "Filtering:\n"
      "  None:\n"
      "    BlocksToFilter:\n"
      "      - Block0\n"
      "      - Group1\n");
  const auto explicit_parsed =
      explicit_parser.template get<OptionTags::Filter<Filter>, Metavars<Dim>>();
  // Before set_blocks_to_filter, resolved ids not yet available.
  CHECK_FALSE(explicit_parsed.blocks_to_filter().has_value());
  const NoneFilter<Dim> expected{
      std::optional<std::vector<std::string>>{{"Block0", "Group1"}}};
  CHECK(explicit_parsed == expected);

  // Duplicate block names rejected via Options framework.
  Options::Parser<tags> dup_parser("");
  dup_parser.parse(
      "DomainCreator:\n"
      "  TestCreator\n"
      "Filtering:\n"
      "  None:\n"
      "    BlocksToFilter:\n"
      "      - Block0\n"
      "      - Block0\n");
  CHECK_THROWS_WITH(
      (dup_parser.template get<OptionTags::Filter<Filter>, Metavars<Dim>>()),
      Catch::Matchers::ContainsSubstring("Duplicate block name"));

  // Invalid block name caught by set_blocks_to_filter.
  auto invalid_filter =
      NoneFilter<Dim>{std::optional<std::vector<std::string>>{{"NotABlock"}}};
  CHECK_THROWS_AS(invalid_filter.set_blocks_to_filter(domain_block_names(),
                                                      domain_block_groups()),
                  std::invalid_argument);

  // Domain without block names but filter has block names -> ERROR.
  CHECK_THROWS_WITH(
      NoneFilter<Dim>{std::optional<std::vector<std::string>>{{"Block0"}}}
          .set_blocks_to_filter({}, {}),
      Catch::Matchers::ContainsSubstring("doesn't use block names"));
}
}  // namespace

// Verify all_filters contains exactly Hypercube and None (no SphericalShell).
static_assert(std::is_same_v<Filters::all_filters<1, TagList<1>>,
                             tmpl::list<Filters::Hypercube<1, TagList<1>>,
                                        Filters::None<1, TagList<1>>>>);
static_assert(std::is_same_v<Filters::all_filters<2, TagList<2>>,
                             tmpl::list<Filters::Hypercube<2, TagList<2>>,
                                        Filters::None<2, TagList<2>>>>);
static_assert(std::is_same_v<Filters::all_filters<3, TagList<3>>,
                             tmpl::list<Filters::Hypercube<3, TagList<3>>,
                                        Filters::None<3, TagList<3>>>>);

SPECTRE_TEST_CASE("Unit.Numerical.LinearOperators.Filter.None",
                  "[NumericalAlgorithms][LinearOperators][Unit]") {
  test_construction_and_accessors<1>();
  test_construction_and_accessors<2>();
  test_construction_and_accessors<3>();

  test_pup_round_trip<1>();
  test_pup_round_trip<2>();
  test_pup_round_trip<3>();

  test_is_equal<1>();
  test_is_equal<2>();
  test_is_equal<3>();

  test_apply_noop<1>();
  test_apply_noop<2>();
  test_apply_noop<3>();

  test_option_parsing<1>();
  test_option_parsing<2>();
  test_option_parsing<3>();
}
