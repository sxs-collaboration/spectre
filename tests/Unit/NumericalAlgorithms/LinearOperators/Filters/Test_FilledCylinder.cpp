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
#include "NumericalAlgorithms/LinearOperators/Filters/FilledCylinder.hpp"
#include "NumericalAlgorithms/LinearOperators/Filters/FilledCylinder.tpp"
#include "NumericalAlgorithms/LinearOperators/Filters/Filter.hpp"
#include "NumericalAlgorithms/LinearOperators/Filters/None.hpp"
#include "NumericalAlgorithms/LinearOperators/Filters/None.tpp"
#include "NumericalAlgorithms/Spectral/Basis.hpp"
#include "NumericalAlgorithms/Spectral/Filtering.hpp"
#include "NumericalAlgorithms/Spectral/FilteringB2.hpp"
#include "NumericalAlgorithms/Spectral/FilteringB2.tpp"
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

struct VectorVar : db::SimpleTag {
  using type = tnsr::I<DataVector, 3>;
};
}  // namespace Tags

using TagList = tmpl::list<Tags::ScalarVar, Tags::VectorVar>;
using CylinderFilter = Filters::FilledCylinder<TagList>;

std::optional<unsigned> to_unsigned(const std::optional<size_t> half_power) {
  if (not half_power.has_value()) {
    return std::nullopt;
  }
  return static_cast<unsigned>(*half_power);
}

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

Domain<3> make_domain() {
  using Identity = domain::CoordinateMaps::Identity<3>;
  using Map =
      domain::CoordinateMap<Frame::BlockLogical, Frame::Inertial, Identity>;
  register_classes_with_charm(tmpl::list<Map>{});
  std::vector<std::unique_ptr<
      domain::CoordinateMapBase<Frame::BlockLogical, Frame::Inertial, 3>>>
      maps{num_blocks};
  for (size_t i = 0; i < num_blocks; ++i) {
    maps[i] = std::make_unique<Map>(Identity{});
  }
  return Domain<3>{
      std::move(maps), {}, domain_block_names(), domain_block_groups()};
}

class TestCreator : public DomainCreator<3> {
 public:
  TestCreator() = default;

  Domain<3> create_domain() const override { return make_domain(); }
  std::vector<std::string> block_names() const override {
    return domain_block_names();
  }
  std::unordered_map<std::string, std::unordered_set<std::string>>
  block_groups() const override {
    return domain_block_groups();
  }
  std::vector<DirectionMap<
      3, std::unique_ptr<domain::BoundaryConditions::BoundaryCondition>>>
  external_boundary_conditions() const override {
    ERROR("");
  }
  std::vector<std::array<size_t, 3>> initial_extents() const override {
    ERROR("");
  }
  std::vector<std::array<size_t, 3>> initial_refinement_levels()
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
};

struct Metavars {
  [[maybe_unused]] static constexpr size_t volume_dim = 3;
  struct factory_creation {
    using factory_classes =
        tmpl::map<tmpl::pair<::DomainCreator<3>, tmpl::list<TestCreator>>>;
  };
};

// Construct the canonical filled-cylinder mesh:
//   dim 0 (radial) ZernikeB2/GaussRadauUpper, dim 1 (angular)
//   ZernikeB2/Equiangular, dim 2 (z) Legendre. The angular extent must be odd
//   and M = n_phi / 2 <= 2 * n_r - 2.
Mesh<3> filled_mesh() {
  return Mesh<3>{
      std::array<size_t, 3>{5, 7, 4},
      std::array<Spectral::Basis, 3>{Spectral::Basis::ZernikeB2,
                                     Spectral::Basis::ZernikeB2,
                                     Spectral::Basis::Legendre},
      std::array<Spectral::Quadrature, 3>{Spectral::Quadrature::GaussRadauUpper,
                                          Spectral::Quadrature::Equiangular,
                                          Spectral::Quadrature::GaussLobatto}};
}

template <size_t Dim>
Variables<TagList> deterministic_vars(const Mesh<Dim>& mesh) {
  Variables<TagList> vars(mesh.number_of_grid_points());
  for (size_t i = 0; i < mesh.number_of_grid_points(); ++i) {
    get(get<Tags::ScalarVar>(vars))[i] = (pow(i, 3) * 0.5) + 1.25;
    for (size_t d = 0; d < 3; ++d) {
      get<Tags::VectorVar>(vars).get(d)[i] =
          static_cast<double>(d) + (pow(i, 3) * 0.75);
    }
  }
  return vars;
}

// Apply a single 1-D matrix in logical direction `dir`. The other directions
// use a default-constructed (empty) Matrix, which apply_matrices treats as the
// identity.
template <size_t Dim>
Variables<TagList> apply_one(const Mesh<Dim>& mesh,
                             const Variables<TagList>& in, const size_t dir,
                             const Matrix& matrix) {
  std::array<Matrix, Dim> filter{};
  gsl::at(filter, dir) = matrix;
  return apply_matrices(filter, in, mesh.extents());
}

void test_construction_and_accessors() {
  INFO("Construction and accessors");
  const std::optional<std::vector<std::string>> blocks_opt{
      std::vector<std::string>{"Block0", "Group1"}};

  const auto filter =
      CylinderFilter(1, 4, 4, true, blocks_opt, false, true, std::nullopt, 3);

  CHECK_FALSE(filter.need_jacobians());
  CHECK(filter.name() == "FilledCylinder");
  CHECK_FALSE(filter.blocks_to_filter().has_value());

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
  for (const size_t step : {size_t{0}, size_t{1}, size_t{7}}) {
    CHECK_FALSE(filter.apply_volume_filter_on_this_step(step));
  }
  CHECK(filter.apply_boundary_filter_on_this_step(0));
  CHECK_FALSE(filter.apply_boundary_filter_on_this_step(1));
  CHECK(filter.apply_boundary_filter_on_this_step(3));

  const auto unrestricted = CylinderFilter(
      0, std::nullopt, 8, true, std::nullopt, true, false, 1, std::nullopt);
  CHECK_FALSE(unrestricted.blocks_to_filter().has_value());
  CHECK(unrestricted.apply_volume_filter_on_substep());
  CHECK(unrestricted.apply_volume_filter_on_this_step(0));
  CHECK(unrestricted.apply_volume_filter_on_this_step(99));

  const auto disabled =
      CylinderFilter(1, 4, 4, false, std::nullopt, true, true, 1, 1);
  CHECK_FALSE(disabled.apply_volume_filter_on_substep());
  CHECK_FALSE(disabled.apply_boundary_filter_on_substep());
  CHECK_FALSE(disabled.apply_volume_filter_on_this_step(0));

  // Equality / inequality: each constructor parameter independently flips.
  const auto base =
      CylinderFilter(1, 4, 4, true, blocks_opt, false, false, 2, 5);
  CHECK(base == CylinderFilter(1, 4, 4, true, blocks_opt, false, false, 2, 5));
  CHECK_FALSE(base !=
              CylinderFilter(1, 4, 4, true, blocks_opt, false, false, 2, 5));
  CHECK(base != CylinderFilter(2, 4, 4, true, blocks_opt, false, false, 2, 5));
  CHECK(base != CylinderFilter(1, std::nullopt, 4, true, blocks_opt, false,
                               false, 2, 5));
  CHECK(base != CylinderFilter(1, 4, std::nullopt, true, blocks_opt, false,
                               false, 2, 5));
  CHECK(base != CylinderFilter(1, 4, 4, false, blocks_opt, false, false, 2, 5));
  CHECK(base !=
        CylinderFilter(1, 4, 4, true, std::nullopt, false, false, 2, 5));
  CHECK(base != CylinderFilter(1, 4, 4, true, blocks_opt, true, false, 2, 5));
  CHECK(base != CylinderFilter(1, 4, 4, true, blocks_opt, false, true, 2, 5));
  CHECK(base != CylinderFilter(1, 4, 4, true, blocks_opt, false, false,
                               std::nullopt, 5));
  CHECK(base != CylinderFilter(1, 4, 4, true, blocks_opt, false, false, 2,
                               std::nullopt));

  CHECK_THROWS_WITH((CylinderFilter{1, 4, 4, true,
                                    std::optional<std::vector<std::string>>{
                                        {"Block0", "Block0"}},
                                    false, false, std::nullopt, std::nullopt}),
                    Catch::Matchers::ContainsSubstring("Duplicate block name"));
}

void test_pup_round_trip() {
  INFO("Serialization");
  const auto filter = CylinderFilter(
      1, 4, std::nullopt, true,
      std::optional<std::vector<std::string>>{{"Block0", "Group2"}}, true,
      false, 4, std::nullopt);
  ::test_serialization(filter);

  using Base = Filters::Filter<3, TagList>;
  register_classes_with_charm<CylinderFilter>();
  const std::unique_ptr<Base> base = std::make_unique<CylinderFilter>(filter);
  const std::unique_ptr<Base> pupped_base = serialize_and_deserialize(base);
  REQUIRE(dynamic_cast<const CylinderFilter*>(pupped_base.get()) != nullptr);
  CHECK(dynamic_cast<const CylinderFilter&>(*pupped_base) == filter);
}

void test_is_equal() {
  INFO("is_equal");
  using Base = Filters::Filter<3, TagList>;
  const std::optional<std::vector<std::string>> blocks{
      std::vector<std::string>{"Block0", "Group1"}};
  const auto a =
      CylinderFilter(1, 4, 4, true, blocks, false, false, 2, std::nullopt);
  const auto b =
      CylinderFilter(1, 4, 4, true, blocks, false, false, 2, std::nullopt);
  const auto c =
      CylinderFilter(2, 4, 4, true, blocks, false, false, 2, std::nullopt);
  CHECK(a.is_equal(b));
  CHECK_FALSE(a.is_equal(c));

  const Filters::None<3, TagList> none_filter{};
  CHECK_FALSE(a.is_equal(none_filter));

  const std::unique_ptr<Base> pa = std::make_unique<CylinderFilter>(a);
  const std::unique_ptr<Base> pb = std::make_unique<CylinderFilter>(b);
  const std::unique_ptr<Base> pnone =
      std::make_unique<Filters::None<3, TagList>>(none_filter);
  CHECK(pa->is_equal(*pb));
  CHECK_FALSE(pa->is_equal(*pnone));
}

void test_supports_mesh() {
  INFO("supports_mesh");
  const auto filter = CylinderFilter(1, 4, 4, true, std::nullopt, false, false,
                                     std::nullopt, std::nullopt);

  // The canonical filled-cylinder mesh is supported, including the Chebyshev
  // variant in the axial direction.
  CHECK(filter.supports_mesh(filled_mesh()));
  CHECK(filter.supports_mesh(Mesh<3>{
      std::array<size_t, 3>{5, 7, 4},
      std::array<Spectral::Basis, 3>{Spectral::Basis::ZernikeB2,
                                     Spectral::Basis::ZernikeB2,
                                     Spectral::Basis::Chebyshev},
      std::array<Spectral::Quadrature, 3>{Spectral::Quadrature::GaussRadauUpper,
                                          Spectral::Quadrature::Equiangular,
                                          Spectral::Quadrature::Gauss}}));

  // A hollow-cylinder (Fourier) mesh is not a filled cylinder.
  CHECK_FALSE(filter.supports_mesh(Mesh<3>{
      std::array<size_t, 3>{5, 7, 4},
      std::array<Spectral::Basis, 3>{Spectral::Basis::Legendre,
                                     Spectral::Basis::Fourier,
                                     Spectral::Basis::Legendre},
      std::array<Spectral::Quadrature, 3>{
          Spectral::Quadrature::Gauss, Spectral::Quadrature::Equiangular,
          Spectral::Quadrature::GaussLobatto}}));

  // A plain Legendre hypercube is not a filled cylinder.
  CHECK_FALSE(filter.supports_mesh(
      Mesh<3>{5, Spectral::Basis::Legendre, Spectral::Quadrature::Gauss}));

  // Radial ZernikeB2 with the wrong (Equiangular) quadrature.
  CHECK_FALSE(filter.supports_mesh(Mesh<3>{
      std::array<size_t, 3>{5, 7, 4},
      std::array<Spectral::Basis, 3>{Spectral::Basis::ZernikeB2,
                                     Spectral::Basis::ZernikeB2,
                                     Spectral::Basis::Legendre},
      std::array<Spectral::Quadrature, 3>{
          Spectral::Quadrature::Equiangular, Spectral::Quadrature::Equiangular,
          Spectral::Quadrature::GaussLobatto}}));

  // Axial direction is not Legendre/Chebyshev.
  CHECK_FALSE(filter.supports_mesh(Mesh<3>{
      std::array<size_t, 3>{5, 7, 7},
      std::array<Spectral::Basis, 3>{Spectral::Basis::ZernikeB2,
                                     Spectral::Basis::ZernikeB2,
                                     Spectral::Basis::ZernikeB2},
      std::array<Spectral::Quadrature, 3>{Spectral::Quadrature::GaussRadauUpper,
                                          Spectral::Quadrature::Equiangular,
                                          Spectral::Quadrature::Equiangular}}));
}

void test_apply_in_volume() {
  INFO("apply_in_volume");
  const Approx custom_approx = Approx::custom().epsilon(5.0e-13);
  const Mesh<3> mesh = filled_mesh();
  const auto initial_vars = deterministic_vars<3>(mesh);

  struct Params {
    size_t num_modes_to_kill;
    std::optional<size_t> radial_angular_half;
    std::optional<size_t> z_half;
  };
  const std::vector<Params> cases{
      {1, 4, 4},                         // everything active
      {2, std::nullopt, std::nullopt},   // only the angular cutoff
      {0, 4, std::nullopt},              // only disk exponential
      {0, std::nullopt, 6},              // only axial
      {0, std::nullopt, std::nullopt}};  // identity

  for (const auto& p : cases) {
    CAPTURE(p.num_modes_to_kill);
    const auto filter = CylinderFilter(
        p.num_modes_to_kill, p.radial_angular_half, p.z_half, true,
        std::nullopt, false, false, std::nullopt, std::nullopt);
    auto vars = initial_vars;
    filter.apply_in_volume(make_not_null(&vars), mesh, std::nullopt,
                           std::nullopt);
    auto expected = initial_vars;
    Spectral::filtering::zernike_b2_cylinder_filter(
        make_not_null(&expected), mesh, 36.0,
        to_unsigned(p.radial_angular_half), to_unsigned(p.z_half),
        p.num_modes_to_kill);
    CHECK_VARIABLES_CUSTOM_APPROX(vars, expected, custom_approx);
  }

  // Buffer reuse: applying the same filter object twice to identical input
  // must give identical output.
  {
    INFO("Buffer reuse");
    const auto filter = CylinderFilter(1, 4, 4, true, std::nullopt, false,
                                       false, std::nullopt, std::nullopt);
    auto vars1 = initial_vars;
    filter.apply_in_volume(make_not_null(&vars1), mesh, std::nullopt,
                           std::nullopt);
    auto vars2 = initial_vars;
    filter.apply_in_volume(make_not_null(&vars2), mesh, std::nullopt,
                           std::nullopt);
    CHECK_VARIABLES_CUSTOM_APPROX(vars1, vars2, custom_approx);
  }
  // The top-mode cutoff genuinely removes the highest angular modes: with a
  // nonzero NumModesToKill the result differs from the unfiltered data.
  const auto cutoff_filter =
      CylinderFilter(2, std::nullopt, std::nullopt, true, std::nullopt, false,
                     false, std::nullopt, std::nullopt);
  auto cutoff_vars = initial_vars;
  cutoff_filter.apply_in_volume(make_not_null(&cutoff_vars), mesh, std::nullopt,
                                std::nullopt);
  CHECK_FALSE(get(get<Tags::ScalarVar>(cutoff_vars)) ==
              get(get<Tags::ScalarVar>(initial_vars)));
}

void test_apply_on_boundary() {
  INFO("apply_on_boundary");
  const Approx custom_approx = Approx::custom().epsilon(5.0e-13);
  const Mesh<3> volume_mesh = filled_mesh();
  const size_t num_modes_to_kill = 1;
  const std::optional<size_t> radial_angular_half = 4;
  const std::optional<size_t> z_half = 5;
  const auto filter =
      CylinderFilter(num_modes_to_kill, radial_angular_half, z_half, true,
                     std::nullopt, false, false, std::nullopt, std::nullopt);

  // Axial face: slice away dim 2 -> face dims (ZernikeB2 radial, ZernikeB2
  // angular). A full disk.
  {
    const Mesh<2> face = volume_mesh.slice_away(2);
    const auto initial = deterministic_vars<2>(face);
    auto vars = initial;
    filter.apply_on_boundary(make_not_null(&vars), face, std::nullopt,
                             std::nullopt);
    auto expected = initial;
    Spectral::filtering::zernike_b2_disk_filter(
        make_not_null(&expected), face, 36.0, to_unsigned(radial_angular_half),
        num_modes_to_kill);
    CHECK_VARIABLES_CUSTOM_APPROX(vars, expected, custom_approx);
  }

  // Mantle face: slice away dim 0, then mortar-converted to Fourier/Equiangular
  // in the angular direction -> face dims (Fourier/Equiangular angular,
  // Legendre z).
  {
    const Mesh<2> face = volume_mesh.on_interface(0);
    const Mesh<1> fourier_angular{face.extents(0), Spectral::Basis::Fourier,
                                  Spectral::Quadrature::Equiangular};
    const auto initial = deterministic_vars<2>(face);
    auto vars = initial;
    filter.apply_on_boundary(make_not_null(&vars), face, std::nullopt,
                             std::nullopt);
    auto expected = initial;
    expected = apply_one(face, expected, 0,
                         Spectral::filtering::exponential_filter(
                             fourier_angular, 36.0,
                             static_cast<unsigned>(*radial_angular_half)));
    expected = apply_one(face, expected, 0,
                         Spectral::filtering::zero_highest_modes(
                             fourier_angular, num_modes_to_kill));
    expected = apply_one(
        face, expected, 1,
        Spectral::filtering::exponential_filter(
            face.slice_through(1), 36.0, static_cast<unsigned>(*z_half)));
    CHECK_VARIABLES_CUSTOM_APPROX(vars, expected, custom_approx);
  }

  // Mantle (radial) face with no disk exponential roll-off: exercises the
  // cutoff-only branch of angular_filter_matrix (RadialAngularHalfPower None,
  // NumModesToKill > 0), so the angular direction is filtered by the top-mode
  // Fourier cutoff alone while z still gets its exponential filter.
  {
    const auto cutoff_filter =
        CylinderFilter(num_modes_to_kill, std::nullopt, z_half, true,
                       std::nullopt, false, false, std::nullopt, std::nullopt);
    const Mesh<2> face{
        std::array<size_t, 2>{volume_mesh.extents(1), volume_mesh.extents(2)},
        std::array<Spectral::Basis, 2>{Spectral::Basis::Fourier,
                                       Spectral::Basis::Legendre},
        std::array<Spectral::Quadrature, 2>{
            Spectral::Quadrature::Equiangular,
            Spectral::Quadrature::GaussLobatto}};
    const Mesh<1> fourier_angular{face.extents(0), Spectral::Basis::Fourier,
                                  Spectral::Quadrature::Equiangular};
    const auto initial = deterministic_vars<2>(face);
    auto vars = initial;
    cutoff_filter.apply_on_boundary(make_not_null(&vars), face, std::nullopt,
                                    std::nullopt);
    auto expected = initial;
    expected = apply_one(face, expected, 0,
                         Spectral::filtering::zero_highest_modes(
                             fourier_angular, num_modes_to_kill));
    expected = apply_one(
        face, expected, 1,
        Spectral::filtering::exponential_filter(
            face.slice_through(1), 36.0, static_cast<unsigned>(*z_half)));
    CHECK_VARIABLES_CUSTOM_APPROX(vars, expected, custom_approx);
  }

  // Angular faces do not exist: the angular direction is periodic, so slicing
  // it away to form a (ZernikeB2/GaussRadauUpper=radial, Legendre=z) boundary
  // face must error.
  {
    const Mesh<2> face = volume_mesh.slice_away(1);
    auto vars = deterministic_vars<2>(face);
    CHECK_THROWS_WITH(
        filter.apply_on_boundary(make_not_null(&vars), face, std::nullopt,
                                 std::nullopt),
        Catch::Matchers::ContainsSubstring(
            "the angular direction is periodic and so has no boundary faces"));
  }

  // All-empty early return: a mantle face filtered by an identity filter (no
  // disk roll-off, no top-mode cutoff, no axial roll-off) leaves both the
  // angular and z directions unfiltered, so both face filters are empty and
  // apply_on_boundary returns the data unchanged.
  {
    const auto identity_filter =
        CylinderFilter(0, std::nullopt, std::nullopt, true, std::nullopt, false,
                       false, std::nullopt, std::nullopt);
    const Mesh<2> face{
        std::array<size_t, 2>{volume_mesh.extents(1), volume_mesh.extents(2)},
        std::array<Spectral::Basis, 2>{Spectral::Basis::Fourier,
                                       Spectral::Basis::Legendre},
        std::array<Spectral::Quadrature, 2>{
            Spectral::Quadrature::Equiangular,
            Spectral::Quadrature::GaussLobatto}};
    const auto initial = deterministic_vars<2>(face);
    auto vars = initial;
    identity_filter.apply_on_boundary(make_not_null(&vars), face, std::nullopt,
                                      std::nullopt);
    CHECK_VARIABLES_APPROX(vars, initial);
  }
}

void test_option_parsing() {
  INFO("Option parsing");
  using Filter = CylinderFilter;
  using tags = tmpl::list<OptionTags::Filter<Filter>,
                          domain::OptionTags::DomainCreator<3>>;

  Options::Parser<tags> parser("");
  parser.parse(
      "DomainCreator:\n"
      "  TestCreator\n"
      "Filtering:\n"
      "  FilledCylinder:\n"
      "    NumModesToKill: 1\n"
      "    RadialAngularHalfPower: None\n"
      "    ZHalfPower: 6\n"
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
  const auto expected = CylinderFilter(
      1, std::nullopt, 6, true,
      std::optional<std::vector<std::string>>{{"Block0", "Group1"}}, false,
      true, 5, std::nullopt);
  CHECK(parsed == expected);

  // BlocksToFilter: All -> nullopt.
  Options::Parser<tags> all_parser("");
  all_parser.parse(
      "DomainCreator:\n"
      "  TestCreator\n"
      "Filtering:\n"
      "  FilledCylinder:\n"
      "    NumModesToKill: 0\n"
      "    RadialAngularHalfPower: 4\n"
      "    ZHalfPower: None\n"
      "    Enable: False\n"
      "    BlocksToFilter: All\n"
      "    VolumeFilterOnSubstep: True\n"
      "    BoundaryCorrectionFilterOnSubstep: False\n"
      "    VolumeFilterEveryNSteps: None\n"
      "    BoundaryCorrectionFilterEveryNSteps: 7\n");
  const auto all_parsed =
      all_parser.template get<OptionTags::Filter<Filter>, Metavars>();
  CHECK_FALSE(all_parsed.blocks_to_filter().has_value());
  CHECK_FALSE(all_parsed.apply_volume_filter_on_substep());

  // Invalid block name caught by set_blocks_to_filter.
  auto invalid_filter =
      CylinderFilter(0, 4, std::nullopt, true,
                     std::optional<std::vector<std::string>>{{"NotABlock"}},
                     false, false, std::nullopt, std::nullopt);
  CHECK_THROWS_AS(invalid_filter.set_blocks_to_filter(domain_block_names(),
                                                      domain_block_groups()),
                  std::invalid_argument);
}

void test_errors() {
  INFO("Errors");
#ifdef SPECTRE_DEBUG
  // Zeroing more angular m-modes than the Fourier modal space resolves trips
  // the assertion at apply time: the angular extent of filled_mesh() is 7
  // (M = 3 m-modes), so killing 4 is invalid. FilledCylinder does no
  // construction-time validation of NumModesToKill against any extent.
  const Mesh<3> mesh = filled_mesh();
  const auto filter =
      CylinderFilter(4, std::nullopt, std::nullopt, true, std::nullopt, false,
                     false, std::nullopt, std::nullopt);
  auto vars = deterministic_vars<3>(mesh);
  CHECK_THROWS_WITH(filter.apply_in_volume(make_not_null(&vars), mesh,
                                           std::nullopt, std::nullopt),
                    Catch::Matchers::ContainsSubstring("angular modes"));
#endif  // SPECTRE_DEBUG
}
}  // namespace

SPECTRE_TEST_CASE("Unit.Numerical.LinearOperators.Filter.FilledCylinder",
                  "[NumericalAlgorithms][LinearOperators][Unit]") {
  test_construction_and_accessors();
  test_pup_round_trip();
  test_is_equal();
  test_supports_mesh();
  test_apply_in_volume();
  test_apply_on_boundary();
  test_option_parsing();
  test_errors();
}
