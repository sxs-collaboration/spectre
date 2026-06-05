// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <array>
#include <cstddef>
#include <functional>
#include <memory>
#include <optional>
#include <random>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "DataStructures/ApplyMatrices.hpp"
#include "DataStructures/DataBox/Tag.hpp"
#include "DataStructures/DataVector.hpp"
#include "DataStructures/Matrix.hpp"
#include "DataStructures/Tensor/EagerMath/DeterminantAndInverse.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "DataStructures/Tensor/TypeAliases.hpp"
#include "DataStructures/Variables.hpp"
#include "Domain/CoordinateMaps/CoordinateMap.hpp"
#include "Domain/CoordinateMaps/CoordinateMap.tpp"
#include "Domain/CoordinateMaps/Identity.hpp"
#include "Domain/Creators/DomainCreator.hpp"
#include "Domain/Creators/OptionTags.hpp"
#include "Domain/Creators/Tags/Domain.hpp"
#include "Domain/Domain.hpp"
#include "Evolution/Systems/CurvedScalarWave/ApplyTensorYlmFilter.hpp"
#include "Evolution/Systems/CurvedScalarWave/Tags.hpp"
#include "Evolution/Tags/Filter.hpp"
#include "Framework/TestCreation.hpp"
#include "Framework/TestHelpers.hpp"
#include "NumericalAlgorithms/LinearOperators/Filters/Filter.hpp"
#include "NumericalAlgorithms/LinearOperators/Filters/None.hpp"
#include "NumericalAlgorithms/LinearOperators/Filters/None.tpp"
#include "NumericalAlgorithms/LinearOperators/Filters/SphericalShell.hpp"
#include "NumericalAlgorithms/LinearOperators/Filters/SphericalShell.tpp"
#include "NumericalAlgorithms/Spectral/Basis.hpp"
#include "NumericalAlgorithms/Spectral/Filtering.hpp"
#include "NumericalAlgorithms/Spectral/Mesh.hpp"
#include "NumericalAlgorithms/Spectral/Quadrature.hpp"
#include "NumericalAlgorithms/SphericalHarmonics/Spherepack.hpp"
#include "NumericalAlgorithms/SphericalHarmonics/SpherepackCache.hpp"
#include "NumericalAlgorithms/SphericalHarmonics/SpherepackIterator.hpp"
#include "Utilities/GetOutput.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/Serialization/RegisterDerivedClassesWithCharm.hpp"
#include "Utilities/TMPL.hpp"

namespace {
using namespace std::string_literals;

constexpr size_t volume_dim = 3;
constexpr size_t num_blocks = 4;

using TagList = ylm::TensorYlm::filter_detail::sw_vars_list<Frame::Inertial>;
using SphericalShellFilter = Filters::SphericalShell<TagList>;

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

Domain<volume_dim> make_domain() {
  using Identity = domain::CoordinateMaps::Identity<volume_dim>;
  using Map =
      domain::CoordinateMap<Frame::BlockLogical, Frame::Inertial, Identity>;
  register_classes_with_charm(tmpl::list<Map>{});
  std::vector<std::unique_ptr<domain::CoordinateMapBase<
      Frame::BlockLogical, Frame::Inertial, volume_dim>>>
      maps{num_blocks};
  for (size_t i = 0; i < num_blocks; ++i) {
    maps[i] = std::make_unique<Map>(Identity{});
  }
  return Domain<volume_dim>{
      std::move(maps), {}, domain_block_names(), domain_block_groups()};
}

class TestCreator : public DomainCreator<volume_dim> {
 public:
  explicit TestCreator(const bool use_block_names = true)
      : use_block_names_(use_block_names) {}

  Domain<volume_dim> create_domain() const override { return make_domain(); }
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
      volume_dim,
      std::unique_ptr<domain::BoundaryConditions::BoundaryCondition>>>
  external_boundary_conditions() const override {
    ERROR("");
  }
  std::vector<std::array<size_t, volume_dim>> initial_extents() const override {
    ERROR("");
  }
  std::vector<std::array<size_t, volume_dim>> initial_refinement_levels()
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

struct Metavars {
  static constexpr size_t volume_dim = ::volume_dim;
  struct factory_creation {
    using factory_classes = tmpl::map<
        tmpl::pair<::DomainCreator<volume_dim>, tmpl::list<TestCreator>>>;
  };
};

SphericalShellFilter make_filter(
    const size_t num_modes_to_kill,
    const std::optional<size_t> angular_half_power,
    const std::optional<size_t> radial_half_power, const bool enable,
    const std::optional<std::vector<std::string>>& blocks_to_filter,
    const bool volume_filter_on_substep, const bool boundary_filter_on_substep,
    const std::optional<size_t> volume_filter_every_n_steps,
    const std::optional<size_t> boundary_filter_every_n_steps) {
  return SphericalShellFilter{num_modes_to_kill,
                              angular_half_power,
                              radial_half_power,
                              enable,
                              blocks_to_filter,
                              volume_filter_on_substep,
                              boundary_filter_on_substep,
                              volume_filter_every_n_steps,
                              boundary_filter_every_n_steps};
}

void test_is_equal() {
  INFO("is_equal");
  using Base = Filters::Filter<3, TagList>;
  const std::optional<std::vector<std::string>> blocks{
      std::vector<std::string>{"Block0", "Block1"}};

  const auto a = make_filter(1, std::nullopt, std::nullopt, true, blocks, false,
                             false, std::nullopt, std::nullopt);
  const auto b = make_filter(1, std::nullopt, std::nullopt, true, blocks, false,
                             false, std::nullopt, std::nullopt);
  const auto c = make_filter(2, std::nullopt, std::nullopt, true, blocks, false,
                             false, std::nullopt, std::nullopt);

  CHECK(a.is_equal(b));
  CHECK(b.is_equal(a));
  CHECK_FALSE(a.is_equal(c));

  // Different concrete type returns false.
  const Filters::None<3, TagList> none_filter{};
  CHECK_FALSE(a.is_equal(none_filter));

  // Via abstract base pointer (the primary AMR use case).
  const std::unique_ptr<Base> pa = std::make_unique<SphericalShellFilter>(a);
  const std::unique_ptr<Base> pb = std::make_unique<SphericalShellFilter>(b);
  const std::unique_ptr<Base> pc = std::make_unique<SphericalShellFilter>(c);
  const std::unique_ptr<Base> pnone =
      std::make_unique<Filters::None<3, TagList>>(none_filter);
  CHECK(pa->is_equal(*pb));
  CHECK_FALSE(pa->is_equal(*pc));
  CHECK_FALSE(pa->is_equal(*pnone));
}

void test_construction_and_accessors() {
  INFO("Construction and accessors");
  const std::vector<std::string> blocks{"Block0", "Group1"};
  const std::optional<std::vector<std::string>> blocks_opt{blocks};

  const auto filter =
      make_filter(2, 16, 8, true, blocks_opt, false, true, std::nullopt, 3);

  CHECK(filter.need_jacobians());
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

  for (const size_t step : {size_t{0}, size_t{1}, size_t{7}, size_t{42}}) {
    CHECK_FALSE(filter.apply_volume_filter_on_this_step(step));
  }
  CHECK(filter.apply_boundary_filter_on_this_step(0));
  CHECK_FALSE(filter.apply_boundary_filter_on_this_step(1));
  CHECK_FALSE(filter.apply_boundary_filter_on_this_step(2));
  CHECK(filter.apply_boundary_filter_on_this_step(3));
  CHECK(filter.apply_boundary_filter_on_this_step(6));

  const auto unrestricted =
      make_filter(0, std::nullopt, std::nullopt, true, std::nullopt, true,
                  false, 1, std::nullopt);
  CHECK_FALSE(unrestricted.blocks_to_filter().has_value());
  CHECK(unrestricted.apply_volume_filter_on_substep());
  CHECK_FALSE(unrestricted.apply_boundary_filter_on_substep());
  CHECK(unrestricted.apply_volume_filter_on_this_step(0));
  CHECK(unrestricted.apply_volume_filter_on_this_step(99));

  // Each constructor parameter independently flips.
  const auto base = make_filter(2, 16, 8, true, blocks_opt, false, false, 2, 5);
  CHECK(base == make_filter(2, 16, 8, true, blocks_opt, false, false, 2, 5));
  CHECK_FALSE(base !=
              make_filter(2, 16, 8, true, blocks_opt, false, false, 2, 5));
  CHECK(base != make_filter(3, 16, 8, true, blocks_opt, false, false, 2, 5));
  CHECK(base != make_filter(2, 12, 8, true, blocks_opt, false, false, 2, 5));
  CHECK(base !=
        make_filter(2, std::nullopt, 8, true, blocks_opt, false, false, 2, 5));
  CHECK(base != make_filter(2, 16, 4, true, blocks_opt, false, false, 2, 5));
  CHECK(base !=
        make_filter(2, 16, std::nullopt, true, blocks_opt, false, false, 2, 5));
  CHECK(base != make_filter(2, 16, 8, false, blocks_opt, false, false, 2, 5));
  CHECK(base != make_filter(2, 16, 8, true, std::nullopt, false, false, 2, 5));
  CHECK(base != make_filter(2, 16, 8, true, blocks_opt, true, false, 2, 5));
  CHECK(base != make_filter(2, 16, 8, true, blocks_opt, false, true, 2, 5));
  CHECK(base !=
        make_filter(2, 16, 8, true, blocks_opt, false, false, std::nullopt, 5));
  CHECK(base !=
        make_filter(2, 16, 8, true, blocks_opt, false, false, 2, std::nullopt));

  // Duplicate block names rejected at construction.
  CHECK_THROWS_WITH(
      (SphericalShellFilter{
          2, 16, 8, true,
          std::optional<std::vector<std::string>>{{"Block0", "Block0"}}, false,
          false, std::nullopt, std::nullopt}),
      Catch::Matchers::ContainsSubstring("Duplicate block name"));
}

void test_pup_round_trip() {
  INFO("Serialization");
  const auto filter =
      make_filter(2, 16, 8, true,
                  std::optional<std::vector<std::string>>{{"Block0", "Group2"}},
                  true, false, 4, std::nullopt);
  ::test_serialization(filter);

  using Base = Filters::Filter<volume_dim, TagList>;
  using Derived = SphericalShellFilter;
  register_classes_with_charm<Derived>();
  const std::unique_ptr<Base> base = std::make_unique<Derived>(filter);
  const std::unique_ptr<Base> pupped_base = serialize_and_deserialize(base);
  REQUIRE(dynamic_cast<const Derived*>(pupped_base.get()) != nullptr);
  CHECK(dynamic_cast<const Derived&>(*pupped_base) == filter);

  // A filter with both half-powers as nullopt also round-trips cleanly.
  const auto heaviside =
      make_filter(1, std::nullopt, std::nullopt, true, std::nullopt, false,
                  false, std::nullopt, std::nullopt);
  ::test_serialization(heaviside);
}

// Build a Mesh<3> with extents matching a Spherepack grid: radial along
// logical direction 0 (Legendre/GaussLobatto), n_theta = ell_max + 1 along
// direction 1, and n_phi = 2*ell_max + 1 along direction 2. The angular
// directions use Legendre as a placeholder; SphericalShell only consumes the
// extents (and the radial basis) so the angular basis choice is inert.
Mesh<volume_dim> make_shell_mesh(const size_t radial_extents,
                                 const size_t ell_max) {
  return Mesh<volume_dim>{{{radial_extents, ell_max + 1, (2 * ell_max) + 1}},
                          Spectral::Basis::Legendre,
                          Spectral::Quadrature::GaussLobatto};
}

// Random Spherepack-modal Variables, restricted to valid (l <= ell_max - rank)
// modes, then transformed to the nodal collocation grid.
Variables<TagList> random_nodal_vars(const size_t radial_extents,
                                     const size_t ell_max,
                                     std::mt19937& generator) {
  const auto& ylm = ::ylm::get_spherepack_cache(ell_max);
  Variables<TagList> modal_vars(ylm.spectral_size() * radial_extents, 0.0);
  std::uniform_real_distribution<double> dist{-1.0, 1.0};
  ylm::SpherepackIterator it(ell_max, ell_max, radial_extents, true);
  tmpl::for_each<TagList>(
      [&modal_vars, &it, &dist, &generator, ell_max](auto tag_v) {
        using Tag = tmpl::type_from<decltype(tag_v)>;
        auto& tensor = get<Tag>(modal_vars);
        for (auto& component : tensor) {
          for (size_t offset = 0; offset < it.stride(); ++offset) {
            for (it.reset(); it; ++it) {
              if (it.l() <= ell_max - tensor.rank()) {
                component[it() + offset] = dist(generator);
              }
            }
          }
        }
      });
  Variables<TagList> nodal_vars(ylm.physical_size() * radial_extents);
  ylm::TensorYlm::filter_detail::modal_to_nodal_ylm(
      make_not_null(&nodal_vars), modal_vars, ylm, radial_extents);
  return nodal_vars;
}

// Build a random diagonally-dominant grid->inertial Jacobian and invert it.
void make_jacobians(
    gsl::not_null<InverseJacobian<DataVector, 3, Frame::Grid, Frame::Inertial>*>
        inv_jac_grid_to_inertial,
    gsl::not_null<Jacobian<DataVector, 3, Frame::Grid, Frame::Inertial>*>
        jac_grid_to_inertial,
    const size_t physical_grid_points, std::mt19937& generator) {
  std::uniform_real_distribution<double> dist{-1.0, 1.0};
  std::uniform_real_distribution<double> positive_dist{0.5, 1.0};
  *jac_grid_to_inertial = Jacobian<DataVector, 3, Frame::Grid, Frame::Inertial>(
      physical_grid_points);
  for (size_t i = 0; i < 3; ++i) {
    for (size_t j = 0; j < 3; ++j) {
      jac_grid_to_inertial->get(i, j) = 0.05 * dist(generator);
    }
    jac_grid_to_inertial->get(i, i) += positive_dist(generator);
  }
  Scalar<DataVector> det(physical_grid_points);
  determinant_and_inverse(make_not_null(&det), inv_jac_grid_to_inertial,
                          *jac_grid_to_inertial);
}

void test_apply_in_volume() {
  INFO("apply_in_volume");
  const Approx custom_approx = Approx::custom().epsilon(5.0e-12);

  constexpr size_t radial_extents = 4;
  constexpr size_t ell_max = 6;
  const auto& ylm = ::ylm::get_spherepack_cache(ell_max);
  const size_t physical_grid_points = ylm.physical_size() * radial_extents;
  const Mesh<volume_dim> mesh = make_shell_mesh(radial_extents, ell_max);
  REQUIRE(mesh.number_of_grid_points() == physical_grid_points);

  MAKE_GENERATOR(generator);
  const auto initial_vars =
      random_nodal_vars(radial_extents, ell_max, generator);

  InverseJacobian<DataVector, 3, Frame::Grid, Frame::Inertial>
      inv_jac_grid_to_inertial(physical_grid_points);
  Jacobian<DataVector, 3, Frame::Grid, Frame::Inertial> jac_grid_to_inertial(
      physical_grid_points);
  make_jacobians(make_not_null(&inv_jac_grid_to_inertial),
                 make_not_null(&jac_grid_to_inertial), physical_grid_points,
                 generator);

  // Helper: directly compute the expected output via the same building blocks
  // SphericalShell uses internally.
  const auto expected_after = [&](const size_t num_modes_to_kill,
                                  const std::optional<size_t> angular_half,
                                  const std::optional<size_t> radial_half) {
    Variables<TagList> vars = initial_vars;
    Variables<TagList> temp(ylm.spectral_size() * radial_extents);
    ylm::TensorYlm::FilterMatrixHolder filter_matrices;
    ylm::TensorYlm::fill_tensor_ylm_filters<TagList>(
        make_not_null(&filter_matrices), ell_max, num_modes_to_kill,
        angular_half, ylm::TensorYlm::CoefficientNormalization::Spherepack);
    ylm::TensorYlm::apply_tensor_ylm_filter(
        make_not_null(&vars), make_not_null(&temp), jac_grid_to_inertial,
        inv_jac_grid_to_inertial, filter_matrices, ell_max, radial_extents);
    if (radial_half.has_value()) {
      const Matrix radial_matrix = Spectral::filtering::exponential_filter(
          mesh.slice_through(0), 36.0, static_cast<unsigned>(*radial_half));
      const Matrix empty{};
      const std::array<std::reference_wrapper<const Matrix>, 3> direction{
          std::cref(radial_matrix), std::cref(empty), std::cref(empty)};
      vars = apply_matrices(direction, vars, mesh.extents());
    }
    return vars;
  };

  const auto run_filter = [&](const SphericalShellFilter& filter) {
    Variables<TagList> vars = initial_vars;
    filter.apply_in_volume(make_not_null(&vars), mesh, inv_jac_grid_to_inertial,
                           jac_grid_to_inertial);
    return vars;
  };

  // Case 1: angular Heaviside-only (no smooth roll-off, no radial filter).
  {
    INFO("Heaviside-only angular filter");
    const auto filter =
        make_filter(2, std::nullopt, std::nullopt, true, std::nullopt, false,
                    false, std::nullopt, std::nullopt);
    const auto vars = run_filter(filter);
    const auto expected = expected_after(2, std::nullopt, std::nullopt);
    CHECK_VARIABLES_CUSTOM_APPROX(vars, expected, custom_approx);
  }

  // Case 2: angular smooth roll-off + radial filter both active.
  {
    INFO("Both angular and radial smooth filters");
    const auto filter = make_filter(2, 16, 8, true, std::nullopt, false, false,
                                    std::nullopt, std::nullopt);
    const auto vars = run_filter(filter);
    const auto expected = expected_after(2, 16, 8);
    CHECK_VARIABLES_CUSTOM_APPROX(vars, expected, custom_approx);
  }

  // Case 3: radial filter only, no angular cutoff.
  {
    INFO("Radial-only filter (num_modes_to_kill = 0)");
    const auto filter = make_filter(0, std::nullopt, 4, true, std::nullopt,
                                    false, false, std::nullopt, std::nullopt);
    const auto vars = run_filter(filter);
    const auto expected = expected_after(0, std::nullopt, 4);
    CHECK_VARIABLES_CUSTOM_APPROX(vars, expected, custom_approx);
  }

  // Case 4: SphericalShell::apply_in_volume errors when jacobians are absent.
#ifdef SPECTRE_DEBUG
  {
    INFO("Missing jacobian arguments cause an error");
    const auto filter =
        make_filter(0, std::nullopt, std::nullopt, true, std::nullopt, false,
                    false, std::nullopt, std::nullopt);
    Variables<TagList> vars = initial_vars;
    CHECK_THROWS_WITH(
        filter.apply_in_volume(make_not_null(&vars), mesh, std::nullopt,
                               jac_grid_to_inertial),
        Catch::Matchers::ContainsSubstring("inv_jac_grid_to_inertial"));
    CHECK_THROWS_WITH(
        filter.apply_in_volume(make_not_null(&vars), mesh,
                               inv_jac_grid_to_inertial, std::nullopt),
        Catch::Matchers::ContainsSubstring("jac_grid_to_inertial"));
  }
#endif
}

void test_apply_on_boundary() {
  INFO("apply_on_boundary");
  const Approx custom_approx = Approx::custom().epsilon(5.0e-12);

  constexpr size_t ell_max = 4;
  const auto& ylm = ::ylm::get_spherepack_cache(ell_max);
  const size_t physical_grid_points = ylm.physical_size();

  // Inner/outer radial face of a shell element: theta x phi, both
  // SphericalHarmonic.
  const Mesh<2> face_mesh{
      std::array<size_t, 2>{ell_max + 1, 2 * ell_max + 1},
      std::array<Spectral::Basis, 2>{Spectral::Basis::SphericalHarmonic,
                                     Spectral::Basis::SphericalHarmonic},
      std::array<Spectral::Quadrature, 2>{Spectral::Quadrature::Gauss,
                                          Spectral::Quadrature::Equiangular}};
  REQUIRE(face_mesh.number_of_grid_points() == physical_grid_points);

  MAKE_GENERATOR(generator);
  // radial_extents = 1 gives physical_size grid points, matching the face mesh.
  const auto initial_vars = random_nodal_vars(1, ell_max, generator);

  InverseJacobian<DataVector, 3, Frame::Grid, Frame::Inertial>
      inv_jac_grid_to_inertial(physical_grid_points);
  Jacobian<DataVector, 3, Frame::Grid, Frame::Inertial> jac_grid_to_inertial(
      physical_grid_points);
  make_jacobians(make_not_null(&inv_jac_grid_to_inertial),
                 make_not_null(&jac_grid_to_inertial), physical_grid_points,
                 generator);

  // Direct computation of expected output: angular filter with radial_extents=1
  // (no radial filter on a face).
  const auto expected_after = [&](const size_t num_modes_to_kill,
                                  const std::optional<size_t> angular_half) {
    Variables<TagList> vars = initial_vars;
    Variables<TagList> temp(ylm.spectral_size());
    ylm::TensorYlm::FilterMatrixHolder filter_matrices;
    ylm::TensorYlm::fill_tensor_ylm_filters<TagList>(
        make_not_null(&filter_matrices), ell_max, num_modes_to_kill,
        angular_half, ylm::TensorYlm::CoefficientNormalization::Spherepack);
    ylm::TensorYlm::apply_tensor_ylm_filter(
        make_not_null(&vars), make_not_null(&temp), jac_grid_to_inertial,
        inv_jac_grid_to_inertial, filter_matrices, ell_max, 1);
    return vars;
  };

  const auto run_filter = [&](const SphericalShellFilter& filter) {
    Variables<TagList> vars = initial_vars;
    filter.apply_on_boundary(make_not_null(&vars), face_mesh,
                             inv_jac_grid_to_inertial, jac_grid_to_inertial);
    return vars;
  };

  // Case 1: Heaviside-only angular filter on boundary.
  {
    INFO("Heaviside-only angular filter");
    const auto filter =
        make_filter(2, std::nullopt, std::nullopt, true, std::nullopt, false,
                    false, std::nullopt, std::nullopt);
    CHECK_VARIABLES_CUSTOM_APPROX(
        run_filter(filter), expected_after(2, std::nullopt), custom_approx);
  }

  // Case 2: smooth angular roll-off on boundary (RadialHalfPower is ignored on
  // a face).
  {
    INFO("Smooth angular filter");
    const auto filter = make_filter(2, 16, 8, true, std::nullopt, false, false,
                                    std::nullopt, std::nullopt);
    CHECK_VARIABLES_CUSTOM_APPROX(run_filter(filter), expected_after(2, 16),
                                  custom_approx);
  }

#ifdef SPECTRE_DEBUG
  {
    INFO("Non-SphericalHarmonic face mesh errors");
    const Mesh<2> legendre_face{{{4, 5}},
                                Spectral::Basis::Legendre,
                                Spectral::Quadrature::GaussLobatto};
    Variables<TagList> vars = initial_vars;
    const auto filter =
        make_filter(2, std::nullopt, std::nullopt, true, std::nullopt, false,
                    false, std::nullopt, std::nullopt);
    CHECK_THROWS_WITH(
        filter.apply_on_boundary(make_not_null(&vars), legendre_face,
                                 std::nullopt, std::nullopt),
        Catch::Matchers::ContainsSubstring("SphericalHarmonic"));
  }
  {
    INFO("Missing jacobian arguments cause an error");
    const auto filter =
        make_filter(2, std::nullopt, std::nullopt, true, std::nullopt, false,
                    false, std::nullopt, std::nullopt);
    Variables<TagList> vars = initial_vars;
    CHECK_THROWS_WITH(
        filter.apply_on_boundary(make_not_null(&vars), face_mesh, std::nullopt,
                                 jac_grid_to_inertial),
        Catch::Matchers::ContainsSubstring("inv_jac_grid_to_inertial"));
    CHECK_THROWS_WITH(
        filter.apply_on_boundary(make_not_null(&vars), face_mesh,
                                 inv_jac_grid_to_inertial, std::nullopt),
        Catch::Matchers::ContainsSubstring("jac_grid_to_inertial"));
  }
#endif
}

void test_supports_mesh() {
  INFO("supports_mesh");
  const auto filter = make_filter(2, 16, 8, true, std::nullopt, false, false,
                                  std::nullopt, std::nullopt);

  // Helper to build a mixed Mesh<3>: radial dim 0, angular dims 1 & 2.
  const auto shell_mesh = [](const Spectral::Basis radial_basis,
                             const Spectral::Quadrature radial_quadrature) {
    // ell_max=1: n_theta=2, n_phi=3
    return Mesh<3>{std::array<size_t, 3>{4, 2, 3},
                   std::array<Spectral::Basis, 3>{
                       radial_basis, Spectral::Basis::SphericalHarmonic,
                       Spectral::Basis::SphericalHarmonic},
                   std::array<Spectral::Quadrature, 3>{
                       radial_quadrature, Spectral::Quadrature::Gauss,
                       Spectral::Quadrature::Equiangular}};
  };

  // Valid radial bases: Legendre and Chebyshev, Gauss and GaussLobatto.
  CHECK(filter.supports_mesh(
      shell_mesh(Spectral::Basis::Legendre, Spectral::Quadrature::Gauss)));
  CHECK(filter.supports_mesh(shell_mesh(Spectral::Basis::Legendre,
                                        Spectral::Quadrature::GaussLobatto)));
  CHECK(filter.supports_mesh(
      shell_mesh(Spectral::Basis::Chebyshev, Spectral::Quadrature::Gauss)));
  CHECK(filter.supports_mesh(shell_mesh(Spectral::Basis::Chebyshev,
                                        Spectral::Quadrature::GaussLobatto)));

  // Invalid radial basis.
  CHECK_FALSE(filter.supports_mesh(
      shell_mesh(Spectral::Basis::Fourier, Spectral::Quadrature::Equiangular)));

  // Wrong angular quadrature in dim 1 (should be Gauss, not GaussLobatto).
  CHECK_FALSE(filter.supports_mesh(Mesh<3>{
      std::array<size_t, 3>{4, 2, 3},
      std::array<Spectral::Basis, 3>{Spectral::Basis::Legendre,
                                     Spectral::Basis::SphericalHarmonic,
                                     Spectral::Basis::SphericalHarmonic},
      std::array<Spectral::Quadrature, 3>{Spectral::Quadrature::Gauss,
                                          Spectral::Quadrature::GaussLobatto,
                                          Spectral::Quadrature::Equiangular}}));

  // Wrong angular quadrature in dim 2 (should be Equiangular, not Gauss).
  CHECK_FALSE(filter.supports_mesh(Mesh<3>{
      std::array<size_t, 3>{4, 2, 3},
      std::array<Spectral::Basis, 3>{Spectral::Basis::Legendre,
                                     Spectral::Basis::SphericalHarmonic,
                                     Spectral::Basis::SphericalHarmonic},
      std::array<Spectral::Quadrature, 3>{Spectral::Quadrature::Gauss,
                                          Spectral::Quadrature::Gauss,
                                          Spectral::Quadrature::Gauss}}));

  // Non-SphericalHarmonic basis in an angular dim.
  CHECK_FALSE(filter.supports_mesh(Mesh<3>{
      std::array<size_t, 3>{4, 3, 3},
      std::array<Spectral::Basis, 3>{Spectral::Basis::Legendre,
                                     Spectral::Basis::Legendre,
                                     Spectral::Basis::SphericalHarmonic},
      std::array<Spectral::Quadrature, 3>{Spectral::Quadrature::Gauss,
                                          Spectral::Quadrature::Gauss,
                                          Spectral::Quadrature::Equiangular}}));

  // All Legendre (a Hypercube mesh, not a shell mesh) -> false.
  CHECK_FALSE(filter.supports_mesh(
      Mesh<3>{4, Spectral::Basis::Legendre, Spectral::Quadrature::Gauss}));
}

void test_option_parsing() {
  INFO("Option parsing");
  using Filter = SphericalShellFilter;
  using tags = tmpl::list<OptionTags::Filter<Filter>,
                          domain::OptionTags::DomainCreator<volume_dim>>;

  Options::Parser<tags> parser("");
  parser.parse(
      "DomainCreator:\n"
      "  TestCreator\n"
      "Filtering:\n"
      "  SphericalShell:\n"
      "    NumModesToKill: 2\n"
      "    AngularHalfPower: 16\n"
      "    RadialHalfPower: 8\n"
      "    Enable: True\n"
      "    BlocksToFilter:\n"
      "      - Block0\n"
      "      - Group1\n"
      "    VolumeFilterOnSubstep: False\n"
      "    BoundaryCorrectionFilterOnSubstep: True\n"
      "    VolumeFilterEveryNSteps: 5\n"
      "    BoundaryCorrectionFilterEveryNSteps: None\n");
  const auto parsed =
      parser.template get<OptionTags::Filter<Filter>, Metavars>();
  const auto expected =
      make_filter(2, 16, 8, true,
                  std::optional<std::vector<std::string>>{{"Block0", "Group1"}},
                  false, true, 5, std::nullopt);
  CHECK(parsed == expected);
  CHECK_FALSE(parsed != expected);

  // Heaviside-only: AngularHalfPower and RadialHalfPower both None.
  Options::Parser<tags> heaviside_parser("");
  heaviside_parser.parse(
      "DomainCreator:\n"
      "  TestCreator\n"
      "Filtering:\n"
      "  SphericalShell:\n"
      "    NumModesToKill: 1\n"
      "    AngularHalfPower: None\n"
      "    RadialHalfPower: None\n"
      "    Enable: True\n"
      "    BlocksToFilter: All\n"
      "    VolumeFilterOnSubstep: True\n"
      "    BoundaryCorrectionFilterOnSubstep: False\n"
      "    VolumeFilterEveryNSteps: None\n"
      "    BoundaryCorrectionFilterEveryNSteps: 7\n");
  const auto heaviside =
      heaviside_parser.template get<OptionTags::Filter<Filter>, Metavars>();
  CHECK_FALSE(heaviside.blocks_to_filter().has_value());
  CHECK(heaviside.apply_volume_filter_on_substep());
  CHECK_FALSE(heaviside.apply_boundary_filter_on_substep());
  CHECK(heaviside.apply_boundary_filter_on_this_step(7));
  CHECK_FALSE(heaviside.apply_boundary_filter_on_this_step(8));

  // Duplicate block names rejected by the option parser.
  Options::Parser<tags> dup_parser("");
  dup_parser.parse(
      "DomainCreator:\n"
      "  TestCreator\n"
      "Filtering:\n"
      "  SphericalShell:\n"
      "    NumModesToKill: 2\n"
      "    AngularHalfPower: 16\n"
      "    RadialHalfPower: 8\n"
      "    Enable: True\n"
      "    BlocksToFilter:\n"
      "      - Block0\n"
      "      - Block0\n"
      "    VolumeFilterOnSubstep: False\n"
      "    BoundaryCorrectionFilterOnSubstep: False\n"
      "    VolumeFilterEveryNSteps: None\n"
      "    BoundaryCorrectionFilterEveryNSteps: None\n");
  CHECK_THROWS_WITH(
      (dup_parser.template get<OptionTags::Filter<Filter>, Metavars>()),
      Catch::Matchers::ContainsSubstring("Duplicate block name"));

  // Invalid block name caught by set_blocks_to_filter.
  auto invalid_filter =
      make_filter(0, std::nullopt, std::nullopt, true,
                  std::optional<std::vector<std::string>>{{"NotABlock"}}, false,
                  false, std::nullopt, std::nullopt);
  CHECK_THROWS_AS(invalid_filter.set_blocks_to_filter(domain_block_names(),
                                                      domain_block_groups()),
                  std::invalid_argument);

  // A domain that doesn't expose block names is rejected when blocks are
  // specified.
  CHECK_THROWS_WITH(
      make_filter(0, std::nullopt, std::nullopt, true,
                  std::optional<std::vector<std::string>>{{"Block0"}}, false,
                  false, std::nullopt, std::nullopt)
          .set_blocks_to_filter({}, {}),
      Catch::Matchers::ContainsSubstring("doesn't use block names"));
}
}  // namespace

// [[TimeOut, 20]]
SPECTRE_TEST_CASE(
    "Unit.NumericalAlgorithms.LinearOperators.Filter.SphericalShell",
    "[NumericalAlgorithms][LinearOperators][Unit]") {
  test_is_equal();
  test_construction_and_accessors();
  test_pup_round_trip();
  test_apply_in_volume();
  test_apply_on_boundary();
  test_supports_mesh();
  test_option_parsing();
}
