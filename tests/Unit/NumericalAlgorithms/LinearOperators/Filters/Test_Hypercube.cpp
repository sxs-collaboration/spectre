// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <array>
#include <cmath>
#include <cstddef>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "DataStructures/ApplyMatrices.hpp"
#include "DataStructures/DataBox/Tag.hpp"
#include "DataStructures/DataVector.hpp"
#include "DataStructures/Matrix.hpp"
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
#include "NumericalAlgorithms/LinearOperators/Filters/Filter.hpp"
#include "NumericalAlgorithms/LinearOperators/Filters/Hypercube.hpp"
#include "NumericalAlgorithms/LinearOperators/Filters/Hypercube.tpp"
#include "NumericalAlgorithms/Spectral/Basis.hpp"
#include "NumericalAlgorithms/Spectral/Filtering.hpp"
#include "NumericalAlgorithms/Spectral/MaximumNumberOfPoints.hpp"
#include "NumericalAlgorithms/Spectral/Mesh.hpp"
#include "NumericalAlgorithms/Spectral/MinimumNumberOfPoints.hpp"
#include "NumericalAlgorithms/Spectral/Quadrature.hpp"
#include "Utilities/GetOutput.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/Serialization/RegisterDerivedClassesWithCharm.hpp"
#include "Utilities/StdArrayHelpers.hpp"
#include "Utilities/TMPL.hpp"

namespace {
using namespace std::string_literals;

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
using TagList = tmpl::list<Tags::ScalarVar, Tags::VectorVar<Dim>>;

template <size_t Dim>
using HypercubeFilter = Filters::Hypercube<Dim, TagList<Dim>>;

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
void test_construction_and_accessors() {
  INFO("Construction and accessors");
  CAPTURE(Dim);
  const std::vector<std::string> blocks{"Block0", "Group1"};
  const std::optional<std::vector<std::string>> blocks_opt{blocks};

  const auto filter =
      HypercubeFilter<Dim>(16, true, blocks_opt, false, true, std::nullopt, 3);

  CHECK_FALSE(filter.need_jacobians());
  // Before set_blocks_to_filter the resolved IDs are not available.
  CHECK_FALSE(filter.blocks_to_filter().has_value());

  // After resolution: Block0->0, Group1 expands to Block1->1 -> sorted {0,1}.
  auto filter_resolved = filter;
  filter_resolved.set_blocks_to_filter(domain_block_names(),
                                       domain_block_groups());
  REQUIRE(filter_resolved.blocks_to_filter().has_value());
  // NOLINTBEGIN(bugprone-unchecked-optional-access)
  CHECK(filter_resolved.blocks_to_filter().value() ==
        std::vector<size_t>{0, 1});
  // NOLINTEND(bugprone-unchecked-optional-access)

  CHECK_FALSE(filter.apply_volume_filter_on_substep());
  CHECK(filter.apply_boundary_filter_on_substep());

  // volume_filter_every_n_steps is nullopt -> always false
  for (const size_t step : {size_t{0}, size_t{1}, size_t{7}, size_t{42}}) {
    CHECK_FALSE(filter.apply_volume_filter_on_this_step(step));
  }
  // boundary_filter_every_n_steps == 3 -> step % 3 == 0
  CHECK(filter.apply_boundary_filter_on_this_step(0));
  CHECK_FALSE(filter.apply_boundary_filter_on_this_step(1));
  CHECK_FALSE(filter.apply_boundary_filter_on_this_step(2));
  CHECK(filter.apply_boundary_filter_on_this_step(3));
  CHECK(filter.apply_boundary_filter_on_this_step(6));

  // Filter without any blocks restriction
  const auto unrestricted =
      HypercubeFilter<Dim>(8, true, std::nullopt, true, false, 1, std::nullopt);
  CHECK_FALSE(unrestricted.blocks_to_filter().has_value());
  CHECK(unrestricted.apply_volume_filter_on_substep());
  CHECK_FALSE(unrestricted.apply_boundary_filter_on_substep());
  // every_n_steps == 1 -> always true
  CHECK(unrestricted.apply_volume_filter_on_this_step(0));
  CHECK(unrestricted.apply_volume_filter_on_this_step(1));
  CHECK(unrestricted.apply_volume_filter_on_this_step(99));

  // Filter disabled
  const auto disabled =
      HypercubeFilter<Dim>(8, false, std::nullopt, true, true, 1, 1);
  CHECK_FALSE(disabled.apply_volume_filter_on_substep());
  CHECK_FALSE(disabled.apply_boundary_filter_on_substep());
  CHECK_FALSE(disabled.apply_volume_filter_on_this_step(0));
  CHECK_FALSE(disabled.apply_volume_filter_on_this_step(1));
  CHECK_FALSE(disabled.apply_volume_filter_on_this_step(99));

  // Equality / inequality: each constructor parameter independently flips.
  const auto base =
      HypercubeFilter<Dim>(4, true, blocks_opt, false, false, 2, 5);
  CHECK(base == HypercubeFilter<Dim>(4, true, blocks_opt, false, false, 2, 5));
  CHECK_FALSE(base !=
              HypercubeFilter<Dim>(4, true, blocks_opt, false, false, 2, 5));
  CHECK(base != HypercubeFilter<Dim>(5, true, blocks_opt, false, false, 2, 5));
  CHECK(base != HypercubeFilter<Dim>(4, false, blocks_opt, false, false, 2, 5));
  CHECK(base !=
        HypercubeFilter<Dim>(4, true, std::nullopt, false, false, 2, 5));
  CHECK(base != HypercubeFilter<Dim>(4, true, blocks_opt, true, false, 2, 5));
  CHECK(base != HypercubeFilter<Dim>(4, true, blocks_opt, false, true, 2, 5));
  CHECK(base != HypercubeFilter<Dim>(4, true, blocks_opt, false, false,
                                     std::nullopt, 5));
  CHECK(base != HypercubeFilter<Dim>(4, true, blocks_opt, false, false, 2,
                                     std::nullopt));

  // Duplicate block names rejected at construction time.
  CHECK_THROWS_WITH(
      (HypercubeFilter<Dim>{
          16, true,
          std::optional<std::vector<std::string>>{{"Block0", "Block0"}}, false,
          false, std::nullopt, std::nullopt}),
      Catch::Matchers::ContainsSubstring("Duplicate block name"));
}

template <size_t Dim>
void test_pup_round_trip() {
  INFO("Serialization");
  CAPTURE(Dim);
  const auto filter = HypercubeFilter<Dim>(
      8, true, std::optional<std::vector<std::string>>{{"Block0", "Group2"}},
      true, false, 4, std::nullopt);
  ::test_serialization(filter);

  // Round-trip through the abstract base pointer.
  using Base = Filters::Filter<Dim, TagList<Dim>>;
  using Derived = HypercubeFilter<Dim>;
  register_classes_with_charm<Derived>();
  const std::unique_ptr<Base> base = std::make_unique<Derived>(filter);
  const std::unique_ptr<Base> pupped_base = serialize_and_deserialize(base);
  REQUIRE(dynamic_cast<const Derived*>(pupped_base.get()) != nullptr);
  CHECK(dynamic_cast<const Derived&>(*pupped_base) == filter);
}

// Build the analytic post-filter expectation by directly invoking
// Spectral::filtering::exponential_filter with the same alpha (36) and
// half_power that Hypercube uses internally.
template <size_t Dim, size_t MatrixDim>
Variables<TagList<Dim>> expected_filtered(
    const Mesh<MatrixDim>& mesh, const Variables<TagList<Dim>>& initial,
    const unsigned half_power) {
  std::array<Matrix, MatrixDim> filter_matrices{};
  for (size_t d = 0; d < MatrixDim; ++d) {
    gsl::at(filter_matrices, d) = Spectral::filtering::exponential_filter(
        mesh.slice_through(d), 36.0, half_power);
  }
  return apply_matrices(filter_matrices, initial, mesh.extents());
}

template <size_t Dim>
Variables<TagList<Dim>> deterministic_vars(const Mesh<Dim>& mesh) {
  Variables<TagList<Dim>> vars(mesh.number_of_grid_points());
  for (size_t i = 0; i < mesh.number_of_grid_points(); ++i) {
    get(get<Tags::ScalarVar>(vars))[i] = (pow(i, 3) * 0.5) + 1.25;
    for (size_t d = 0; d < Dim; ++d) {
      get<Tags::VectorVar<Dim>>(vars).get(d)[i] =
          static_cast<double>(d) + (pow(i, 3) * 0.75);
    }
  }
  return vars;
}

template <size_t FaceDim, size_t Dim>
Variables<TagList<Dim>> deterministic_vars(const Mesh<FaceDim>& mesh) {
  Variables<TagList<Dim>> vars(mesh.number_of_grid_points());
  for (size_t i = 0; i < mesh.number_of_grid_points(); ++i) {
    get(get<Tags::ScalarVar>(vars))[i] = (pow(i, 2) * 0.25) + 0.5;
    for (size_t d = 0; d < Dim; ++d) {
      get<Tags::VectorVar<Dim>>(vars).get(d)[i] =
          static_cast<double>(d) + 1.0 + (pow(i, 2) * 0.4);
    }
  }
  return vars;
}

// All apply_* tests for a given `Hypercube<Dim, TagList>` instantiation must
// share a single `half_power_` because `Hypercube::filter_matrix` caches
// matrices in a per-instantiation static cache that is keyed by the first
// observed `half_power_`. Mixing half_power values across calls in the same
// process aborts with "Filter was cached with half power = ...". Each Dim
// gets its own static cache, but for cleanliness we use the same value
// throughout the suite.
constexpr unsigned kFilterHalfPower = 4u;

template <size_t Dim, Spectral::Basis BasisType,
          Spectral::Quadrature QuadratureType>
void test_apply_in_volume() {
  INFO("apply_in_volume");
  CAPTURE(Dim);
  CAPTURE(BasisType);
  CAPTURE(QuadratureType);
  const Approx custom_approx = Approx::custom().epsilon(5.0e-13);

  const size_t max_pts =
      BasisType == Spectral::Basis::Fourier
          ? Spectral::maximum_number_of_points<BasisType> / (3 * Dim)
          : Spectral::maximum_number_of_points<BasisType> / Dim;
  for (size_t num_pts =
           Spectral::minimum_number_of_points<BasisType, QuadratureType>;
       num_pts < max_pts; ++num_pts) {
    if constexpr (BasisType == Spectral::Basis::Fourier) {
      if (num_pts % 2 == 0) {
        continue;
      }
    }
    CAPTURE(num_pts);
    const Mesh<Dim> mesh(num_pts, BasisType, QuadratureType);
    const auto initial_vars = deterministic_vars<Dim>(mesh);

    // The Hypercube class itself is unconditional in apply_in_volume —
    // the cadence/enable gating lives in the driving action. Verify
    // that calling apply_in_volume always rescales the modes,
    // regardless of how `enable` is set.
    for (const bool enable : {true, false}) {
      CAPTURE(enable);
      const auto filter =
          HypercubeFilter<Dim>(kFilterHalfPower, enable, std::nullopt, false,
                               false, std::nullopt, std::nullopt);
      auto vars = initial_vars;
      filter.apply_in_volume(make_not_null(&vars), mesh, std::nullopt,
                             std::nullopt);
      const auto expected =
          expected_filtered<Dim, Dim>(mesh, initial_vars, kFilterHalfPower);
      CHECK_VARIABLES_CUSTOM_APPROX(vars, expected, custom_approx);
    }
  }
}

template <size_t Dim, Spectral::Basis BasisType,
          Spectral::Quadrature QuadratureType>
void test_apply_on_boundary() {
  static_assert(Dim >= 2,
                "Boundary filtering requires a non-degenerate face mesh.");
  INFO("apply_on_boundary");
  CAPTURE(Dim);
  CAPTURE(BasisType);
  CAPTURE(QuadratureType);
  const Approx custom_approx = Approx::custom().epsilon(5.0e-13);

  const size_t max_pts =
      BasisType == Spectral::Basis::Fourier
          ? Spectral::maximum_number_of_points<BasisType> / (3 * Dim)
          : Spectral::maximum_number_of_points<BasisType> / Dim;
  for (size_t num_pts =
           Spectral::minimum_number_of_points<BasisType, QuadratureType>;
       num_pts < max_pts; ++num_pts) {
    if constexpr (BasisType == Spectral::Basis::Fourier) {
      if (num_pts % 2 == 0) {
        continue;
      }
    }
    CAPTURE(num_pts);
    const Mesh<Dim> volume_mesh(num_pts, BasisType, QuadratureType);
    const Mesh<Dim - 1> face_mesh = volume_mesh.slice_away(0);
    const auto initial_face_vars = deterministic_vars<Dim - 1, Dim>(face_mesh);

    const auto filter =
        HypercubeFilter<Dim>(kFilterHalfPower, true, std::nullopt, false, false,
                             std::nullopt, std::nullopt);
    auto face_vars = initial_face_vars;
    filter.apply_on_boundary(make_not_null(&face_vars), face_mesh, std::nullopt,
                             std::nullopt);
    const auto expected = expected_filtered<Dim, Dim - 1>(
        face_mesh, initial_face_vars, kFilterHalfPower);
    CHECK_VARIABLES_CUSTOM_APPROX(face_vars, expected, custom_approx);
  }
}

template <size_t Dim>
void test_invoke_apply() {
  test_apply_in_volume<Dim, Spectral::Basis::Legendre,
                       Spectral::Quadrature::Gauss>();
  test_apply_in_volume<Dim, Spectral::Basis::Legendre,
                       Spectral::Quadrature::GaussLobatto>();
  test_apply_in_volume<Dim, Spectral::Basis::Chebyshev,
                       Spectral::Quadrature::Gauss>();
  test_apply_in_volume<Dim, Spectral::Basis::Chebyshev,
                       Spectral::Quadrature::GaussLobatto>();
  test_apply_in_volume<Dim, Spectral::Basis::Fourier,
                       Spectral::Quadrature::Equiangular>();

  if constexpr (Dim >= 2) {
    test_apply_on_boundary<Dim, Spectral::Basis::Legendre,
                           Spectral::Quadrature::Gauss>();
    test_apply_on_boundary<Dim, Spectral::Basis::Legendre,
                           Spectral::Quadrature::GaussLobatto>();
    test_apply_on_boundary<Dim, Spectral::Basis::Chebyshev,
                           Spectral::Quadrature::Gauss>();
    test_apply_on_boundary<Dim, Spectral::Basis::Chebyshev,
                           Spectral::Quadrature::GaussLobatto>();
    test_apply_on_boundary<Dim, Spectral::Basis::Fourier,
                           Spectral::Quadrature::Equiangular>();
  }
}

template <size_t Dim>
void test_option_parsing() {
  INFO("Option parsing");
  CAPTURE(Dim);
  using Filter = HypercubeFilter<Dim>;
  using tags = tmpl::list<OptionTags::Filter<Filter>,
                          domain::OptionTags::DomainCreator<Dim>>;

  // Full option set with explicit blocks.
  Options::Parser<tags> parser("");
  parser.parse(
      "DomainCreator:\n"
      "  TestCreator\n"
      "Filtering:\n"
      "  Hypercube:\n"
      "    HalfPower: 16\n"
      "    Enable: True\n"
      "    BlocksToFilter:\n"
      "      - Block0\n"
      "      - Group1\n"
      "    VolumeFilterOnSubstep: False\n"
      "    BoundaryCorrectionFilterOnSubstep: True\n"
      "    VolumeFilterEveryNSteps: 5\n"
      "    BoundaryCorrectionFilterEveryNSteps: None\n");

  const auto parsed =
      parser.template get<OptionTags::Filter<Filter>, Metavars<Dim>>();
  const auto expected = HypercubeFilter<Dim>(
      16, true, std::optional<std::vector<std::string>>{{"Block0", "Group1"}},
      false, true, 5, std::nullopt);
  CHECK(parsed == expected);
  CHECK_FALSE(parsed != expected);

  // BlocksToFilter: All -> nullopt.
  Options::Parser<tags> all_parser("");
  all_parser.parse(
      "DomainCreator:\n"
      "  TestCreator\n"
      "Filtering:\n"
      "  Hypercube:\n"
      "    HalfPower: 4\n"
      "    Enable: False\n"
      "    BlocksToFilter: All\n"
      "    VolumeFilterOnSubstep: True\n"
      "    BoundaryCorrectionFilterOnSubstep: False\n"
      "    VolumeFilterEveryNSteps: None\n"
      "    BoundaryCorrectionFilterEveryNSteps: 7\n");
  const auto all_parsed =
      all_parser.template get<OptionTags::Filter<Filter>, Metavars<Dim>>();
  CHECK_FALSE(all_parsed.blocks_to_filter().has_value());
  // Enable: false disables scheduling regardless of other settings
  CHECK_FALSE(all_parsed.apply_volume_filter_on_substep());
  CHECK_FALSE(all_parsed.apply_boundary_filter_on_substep());
  CHECK_FALSE(all_parsed.apply_volume_filter_on_this_step(0));
  CHECK_FALSE(all_parsed.apply_boundary_filter_on_this_step(0));
  CHECK_FALSE(all_parsed.apply_boundary_filter_on_this_step(7));
  CHECK_FALSE(all_parsed.apply_boundary_filter_on_this_step(8));

  // Duplicate block names parsed via the Options framework -> PARSE_ERROR.
  Options::Parser<tags> dup_parser("");
  dup_parser.parse(
      "DomainCreator:\n"
      "  TestCreator\n"
      "Filtering:\n"
      "  Hypercube:\n"
      "    HalfPower: 4\n"
      "    Enable: True\n"
      "    BlocksToFilter:\n"
      "      - Block0\n"
      "      - Block0\n"
      "    VolumeFilterOnSubstep: False\n"
      "    BoundaryCorrectionFilterOnSubstep: False\n"
      "    VolumeFilterEveryNSteps: None\n"
      "    BoundaryCorrectionFilterEveryNSteps: None\n");
  CHECK_THROWS_WITH(
      (dup_parser.template get<OptionTags::Filter<Filter>, Metavars<Dim>>()),
      Catch::Matchers::ContainsSubstring("Duplicate block name"));

  // Invalid block name caught by set_blocks_to_filter.
  auto invalid_filter = HypercubeFilter<Dim>(
      4, true, std::optional<std::vector<std::string>>{{"NotABlock"}}, false,
      false, std::nullopt, std::nullopt);
  CHECK_THROWS_AS(invalid_filter.set_blocks_to_filter(domain_block_names(),
                                                      domain_block_groups()),
                  std::invalid_argument);

  // Domain that doesn't expose block names but a filter does -> ERROR.
  CHECK_THROWS_WITH(
      HypercubeFilter<Dim>(4, true,
                           std::optional<std::vector<std::string>>{{"Block0"}},
                           false, false, std::nullopt, std::nullopt)
          .set_blocks_to_filter({}, {}),
      Catch::Matchers::ContainsSubstring("doesn't use block names"));

  // Block-only and group-only specifications resolve without throwing.
  CHECK_NOTHROW(
      HypercubeFilter<Dim>(4, true,
                           std::optional<std::vector<std::string>>{{"Block0"}},
                           false, false, std::nullopt, std::nullopt)
          .set_blocks_to_filter(domain_block_names(), domain_block_groups()));
  CHECK_NOTHROW(
      HypercubeFilter<Dim>(4, true,
                           std::optional<std::vector<std::string>>{{"Group1"}},
                           false, false, std::nullopt, std::nullopt)
          .set_blocks_to_filter(domain_block_names(), domain_block_groups()));
}

void test_supports_mesh() {
  INFO("supports_mesh");
  // Reuse kFilterHalfPower so the static cache stays consistent across tests.
  const auto f1 = HypercubeFilter<1>(kFilterHalfPower, true, std::nullopt,
                                     false, false, std::nullopt, std::nullopt);
  const auto f2 = HypercubeFilter<2>(kFilterHalfPower, true, std::nullopt,
                                     false, false, std::nullopt, std::nullopt);
  const auto f3 = HypercubeFilter<3>(kFilterHalfPower, true, std::nullopt,
                                     false, false, std::nullopt, std::nullopt);

  // All valid (Basis, Quadrature) pairs.
  CHECK(f1.supports_mesh(
      Mesh<1>{3, Spectral::Basis::Legendre, Spectral::Quadrature::Gauss}));
  CHECK(f1.supports_mesh(Mesh<1>{3, Spectral::Basis::Legendre,
                                 Spectral::Quadrature::GaussLobatto}));
  CHECK(f1.supports_mesh(
      Mesh<1>{3, Spectral::Basis::Chebyshev, Spectral::Quadrature::Gauss}));
  CHECK(f1.supports_mesh(Mesh<1>{3, Spectral::Basis::Chebyshev,
                                 Spectral::Quadrature::GaussLobatto}));
  CHECK(f1.supports_mesh(
      Mesh<1>{4, Spectral::Basis::Fourier, Spectral::Quadrature::Equiangular}));
  CHECK(f1.supports_mesh(Mesh<1>{1, Spectral::Basis::Cartoon,
                                 Spectral::Quadrature::AxialSymmetry}));
  CHECK(f1.supports_mesh(Mesh<1>{1, Spectral::Basis::Cartoon,
                                 Spectral::Quadrature::SphericalSymmetry}));

  // Invalid cross-combos: wrong quadrature for the basis.
  CHECK_FALSE(f1.supports_mesh(Mesh<1>{3, Spectral::Basis::Legendre,
                                       Spectral::Quadrature::AxialSymmetry}));
  CHECK_FALSE(f1.supports_mesh(Mesh<1>{
      3, Spectral::Basis::Legendre, Spectral::Quadrature::SphericalSymmetry}));
  CHECK_FALSE(f1.supports_mesh(Mesh<1>{3, Spectral::Basis::Chebyshev,
                                       Spectral::Quadrature::AxialSymmetry}));
  CHECK_FALSE(f1.supports_mesh(Mesh<1>{
      3, Spectral::Basis::Chebyshev, Spectral::Quadrature::SphericalSymmetry}));
  CHECK_FALSE(f1.supports_mesh(
      Mesh<1>{1, Spectral::Basis::Cartoon, Spectral::Quadrature::Gauss}));
  CHECK_FALSE(f1.supports_mesh(Mesh<1>{1, Spectral::Basis::Cartoon,
                                       Spectral::Quadrature::GaussLobatto}));
  CHECK_FALSE(f1.supports_mesh(
      Mesh<1>{4, Spectral::Basis::Fourier, Spectral::Quadrature::Gauss}));

  // Unsupported basis entirely.
  CHECK_FALSE(f1.supports_mesh(Mesh<1>{3, Spectral::Basis::FiniteDifference,
                                       Spectral::Quadrature::CellCentered}));
  CHECK_FALSE(f1.supports_mesh(Mesh<1>{3, Spectral::Basis::SphericalHarmonic,
                                       Spectral::Quadrature::Gauss}));
  CHECK_FALSE(f1.supports_mesh(Mesh<1>{4, Spectral::Basis::SphericalHarmonic,
                                       Spectral::Quadrature::Equiangular}));
  CHECK_FALSE(f1.supports_mesh(Mesh<1>{3, Spectral::Basis::ZernikeB1,
                                       Spectral::Quadrature::GaussRadauUpper}));
  CHECK_FALSE(f1.supports_mesh(Mesh<1>{3, Spectral::Basis::ZernikeB2,
                                       Spectral::Quadrature::GaussRadauUpper}));
  CHECK_FALSE(f1.supports_mesh(Mesh<1>{3, Spectral::Basis::ZernikeB3,
                                       Spectral::Quadrature::GaussRadauUpper}));

  // Multi-dim: all dims valid -> true.
  CHECK(f2.supports_mesh(
      Mesh<2>{3, Spectral::Basis::Legendre, Spectral::Quadrature::Gauss}));
  CHECK(f3.supports_mesh(Mesh<3>{3, Spectral::Basis::Chebyshev,
                                 Spectral::Quadrature::GaussLobatto}));

  // Multi-dim: one unsupported dim -> false.
  CHECK_FALSE(f2.supports_mesh(Mesh<2>{
      std::array<size_t, 2>{3, 3},
      std::array<Spectral::Basis, 2>{Spectral::Basis::Legendre,
                                     Spectral::Basis::FiniteDifference},
      std::array<Spectral::Quadrature, 2>{
          Spectral::Quadrature::Gauss, Spectral::Quadrature::CellCentered}}));
  CHECK_FALSE(f3.supports_mesh(Mesh<3>{
      std::array<size_t, 3>{3, 3, 3},
      std::array<Spectral::Basis, 3>{Spectral::Basis::Legendre,
                                     Spectral::Basis::Chebyshev,
                                     Spectral::Basis::Legendre},
      std::array<Spectral::Quadrature, 3>{
          Spectral::Quadrature::Gauss, Spectral::Quadrature::GaussLobatto,
          Spectral::Quadrature::AxialSymmetry}}));
}

template <size_t Dim>
void test_is_equal() {
  INFO("is_equal");
  CAPTURE(Dim);
  using Base = Filters::Filter<Dim, TagList<Dim>>;
  const std::optional<std::vector<std::string>> blocks{
      std::vector<std::string>{"Block0", "Group1"}};

  const auto a =
      HypercubeFilter<Dim>(8, true, blocks, false, false, 2, std::nullopt);
  const auto b =
      HypercubeFilter<Dim>(8, true, blocks, false, false, 2, std::nullopt);
  const auto c =
      HypercubeFilter<Dim>(16, true, blocks, false, false, 2, std::nullopt);

  CHECK(a.is_equal(b));
  CHECK(b.is_equal(a));
  CHECK_FALSE(a.is_equal(c));

  // Via abstract base pointer (the primary AMR use case).
  const std::unique_ptr<Base> pa = std::make_unique<HypercubeFilter<Dim>>(a);
  const std::unique_ptr<Base> pb = std::make_unique<HypercubeFilter<Dim>>(b);
  const std::unique_ptr<Base> pc = std::make_unique<HypercubeFilter<Dim>>(c);
  CHECK(pa->is_equal(*pb));
  CHECK_FALSE(pa->is_equal(*pc));
}

void test_cartoon() {
  INFO("Cartoon basis");
  // Cartoon meshes are 1-point and the resulting 1x1 filter matrix is the
  // identity, so the data should pass through unchanged regardless of
  // half_power. Reuse `kFilterHalfPower` so `Hypercube<1, ...>::filter_matrix`
  // shares its static cache with `test_apply_in_volume<1, ...>`.
  const auto filter =
      HypercubeFilter<1>(kFilterHalfPower, true, std::nullopt, false, false,
                         std::nullopt, std::nullopt);
  for (const auto quadrature : {Spectral::Quadrature::AxialSymmetry,
                                Spectral::Quadrature::SphericalSymmetry}) {
    CAPTURE(quadrature);
    const Mesh<1> mesh{1, Spectral::Basis::Cartoon, quadrature};
    Variables<TagList<1>> vars(mesh.number_of_grid_points());
    get(get<Tags::ScalarVar>(vars))[0] = 3.14;
    get<Tags::VectorVar<1>>(vars).get(0)[0] = -2.71;
    const auto initial = vars;
    filter.apply_in_volume(make_not_null(&vars), mesh, std::nullopt,
                           std::nullopt);
    CHECK(get(get<Tags::ScalarVar>(vars))[0] ==
          approx(get(get<Tags::ScalarVar>(initial))[0]));
    CHECK(get<Tags::VectorVar<1>>(vars).get(0)[0] ==
          approx(get<Tags::VectorVar<1>>(initial).get(0)[0]));
  }
}
}  // namespace

SPECTRE_TEST_CASE("Unit.Numerical.LinearOperators.Filter.Cube",
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

  tmpl::for_each<tmpl::make_sequence<tmpl::size_t<1>, 3>>([](auto dim_v) {
    constexpr size_t Dim = tmpl::type_from<decltype(dim_v)>::value;
    test_invoke_apply<Dim>();
  });

  test_option_parsing<1>();
  test_option_parsing<2>();
  test_option_parsing<3>();

  test_supports_mesh();
  test_cartoon();
}
