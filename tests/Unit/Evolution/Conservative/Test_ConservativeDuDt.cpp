// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "tests/Unit/TestingFramework.hpp"

#include <array>
#include <cstddef>
#include <string>

#include "DataStructures/DataBox/Prefixes.hpp"  // IWYU pragma: keep
#include "DataStructures/DataBox/Tag.hpp"
#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "DataStructures/Variables.hpp"
#include "Domain/CoordinateMaps/Affine.hpp"
#include "Domain/CoordinateMaps/CoordinateMap.hpp"
#include "Domain/CoordinateMaps/CoordinateMap.tpp"
#include "Domain/CoordinateMaps/ProductMaps.hpp"
#include "Domain/CoordinateMaps/ProductMaps.tpp"
#include "Domain/LogicalCoordinates.hpp"
#include "Domain/Mesh.hpp"
#include "Evolution/Conservative/ConservativeDuDt.hpp"
#include "NumericalAlgorithms/LinearOperators/Divergence.tpp"  // IWYU pragma: keep
#include "PointwiseFunctions/MathFunctions/MathFunction.hpp"
#include "PointwiseFunctions/MathFunctions/PowX.hpp"
#include "PointwiseFunctions/MathFunctions/TensorProduct.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/TMPL.hpp"
#include "Utilities/TypeTraits.hpp"

// IWYU pragma: no_forward_declare Tags::Flux
// IWYU pragma: no_forward_declare Tags::div

namespace {
struct Var1 : db::SimpleTag {
  using type = Scalar<DataVector>;
};

template <size_t Dim>
struct Var2 : db::SimpleTag {
  using type = tnsr::i<DataVector, Dim, Frame::Inertial>;
};

template <typename Tag>
struct Functions;

template <size_t Dim>
struct Functions<Tags::Flux<Var1, tmpl::size_t<Dim>, Frame::Inertial>> {
  static auto flux(
      const MathFunctions::TensorProduct<Dim>& f,
      const tnsr::I<DataVector, Dim, Frame::Inertial>& x) noexcept {
    auto result =
        make_with_value<tnsr::I<DataVector, Dim, Frame::Inertial>>(x, 0.);
    const auto f_of_x = f(x);
    for (size_t d = 0; d < Dim; ++d) {
      result.get(d) = (d + 0.5) * get(f_of_x);
    }
    return result;
  }
  static auto divergence_of_flux(
      const MathFunctions::TensorProduct<Dim>& f,
      const tnsr::I<DataVector, Dim, Frame::Inertial>& x) noexcept {
    auto result = make_with_value<Scalar<DataVector>>(x, 0.);
    const auto df = f.first_derivatives(x);
    for (size_t d = 0; d < Dim; ++d) {
      get(result) += (d + 0.5) * df.get(d);
    }
    return result;
  }
};

template <size_t Dim>
struct Functions<Tags::Flux<Var2<Dim>, tmpl::size_t<Dim>, Frame::Inertial>> {
  static auto flux(
      const MathFunctions::TensorProduct<Dim>& f,
      const tnsr::I<DataVector, Dim, Frame::Inertial>& x) noexcept {
    auto result =
        make_with_value<tnsr::Ij<DataVector, Dim, Frame::Inertial>>(x, 0.);
    const auto f_of_x = f(x);
    for (size_t d = 0; d < Dim; ++d) {
      for (size_t j = 0; j < Dim; ++j) {
        result.get(d, j) = (d + 0.5) * (j + 0.25) * get(f_of_x);
      }
    }
    return result;
  }
  static auto divergence_of_flux(
      const MathFunctions::TensorProduct<Dim>& f,
      const tnsr::I<DataVector, Dim, Frame::Inertial>& x) noexcept {
    auto result =
        make_with_value<tnsr::i<DataVector, Dim, Frame::Inertial>>(x, 0.);
    const auto df = f.first_derivatives(x);
    for (size_t j = 0; j < Dim; ++j) {
      for (size_t d = 0; d < Dim; ++d) {
        result.get(j) += (d + 0.5) * (j + 0.25) * df.get(d);
      }
    }
    return result;
  }
};

template <typename SourcedVariables, size_t Dim>
struct System {
  using variables_tag = Tags::Variables<tmpl::list<Var1, Var2<Dim>>>;
  using sourced_variables = SourcedVariables;
  static constexpr size_t volume_dim = Dim;
};

template <typename SourcedVariables, size_t Dim>
using expected_argument_tags = tmpl::list<
    domain::Tags::Mesh<Dim>,
    domain::Tags::InverseJacobian<Dim, Frame::Logical, Frame::Inertial>,
    db::add_tag_prefix<Tags::Flux,
                       typename System<SourcedVariables, Dim>::variables_tag,
                       tmpl::size_t<Dim>, Frame::Inertial>,
    db::add_tag_prefix<Tags::Source,
                       typename System<SourcedVariables, Dim>::variables_tag>>;

static_assert(
    cpp17::is_same_v<ConservativeDuDt<System<tmpl::list<>, 1>>::argument_tags,
                     expected_argument_tags<tmpl::list<>, 1>>,
    "Failed testing ConservativeDuDt::argument_tags");
static_assert(
    cpp17::is_same_v<ConservativeDuDt<System<tmpl::list<>, 2>>::argument_tags,
                     expected_argument_tags<tmpl::list<>, 2>>,
    "Failed testing ConservativeDuDt::argument_tags");
static_assert(
    cpp17::is_same_v<ConservativeDuDt<System<tmpl::list<>, 3>>::argument_tags,
                     expected_argument_tags<tmpl::list<>, 3>>,
    "Failed testing ConservativeDuDt::argument_tags");
static_assert(cpp17::is_same_v<
                  ConservativeDuDt<System<tmpl::list<Var1>, 1>>::argument_tags,
                  expected_argument_tags<tmpl::list<Var1>, 1>>,
              "Failed testing ConservativeDuDt::argument_tags");
static_assert(cpp17::is_same_v<
                  ConservativeDuDt<System<tmpl::list<Var1>, 2>>::argument_tags,
                  expected_argument_tags<tmpl::list<Var1>, 2>>,
              "Failed testing ConservativeDuDt::argument_tags");
static_assert(cpp17::is_same_v<
                  ConservativeDuDt<System<tmpl::list<Var1>, 3>>::argument_tags,
                  expected_argument_tags<tmpl::list<Var1>, 3>>,
              "Failed testing ConservativeDuDt::argument_tags");
static_assert(
    cpp17::is_same_v<
        ConservativeDuDt<System<tmpl::list<Var2<1>>, 1>>::argument_tags,
        expected_argument_tags<tmpl::list<Var2<1>>, 1>>,
    "Failed testing ConservativeDuDt::argument_tags");
static_assert(
    cpp17::is_same_v<
        ConservativeDuDt<System<tmpl::list<Var2<2>>, 2>>::argument_tags,
        expected_argument_tags<tmpl::list<Var2<2>>, 2>>,
    "Failed testing ConservativeDuDt::argument_tags");
static_assert(
    cpp17::is_same_v<
        ConservativeDuDt<System<tmpl::list<Var2<3>>, 3>>::argument_tags,
        expected_argument_tags<tmpl::list<Var2<3>>, 3>>,
    "Failed testing ConservativeDuDt::argument_tags");
static_assert(
    cpp17::is_same_v<
        ConservativeDuDt<System<tmpl::list<Var1, Var2<1>>, 1>>::argument_tags,
        expected_argument_tags<tmpl::list<Var1, Var2<1>>, 1>>,
    "Failed testing ConservativeDuDt::argument_tags");
static_assert(
    cpp17::is_same_v<
        ConservativeDuDt<System<tmpl::list<Var1, Var2<2>>, 2>>::argument_tags,
        expected_argument_tags<tmpl::list<Var1, Var2<2>>, 2>>,
    "Failed testing ConservativeDuDt::argument_tags");
static_assert(
    cpp17::is_same_v<
        ConservativeDuDt<System<tmpl::list<Var1, Var2<3>>, 3>>::argument_tags,
        expected_argument_tags<tmpl::list<Var1, Var2<3>>, 3>>,
    "Failed testing ConservativeDuDt::argument_tags");

using Affine = domain::CoordinateMaps::Affine;
using Affine2D = domain::CoordinateMaps::ProductOf2Maps<Affine, Affine>;
using Affine3D = domain::CoordinateMaps::ProductOf3Maps<Affine, Affine, Affine>;

template <size_t VolumeDim>
auto make_affine_map() noexcept;

template <>
auto make_affine_map<1>() noexcept {
  return domain::make_coordinate_map<Frame::Logical, Frame::Inertial>(
      Affine{-1.0, 1.0, -0.3, 0.7});
}

template <>
auto make_affine_map<2>() noexcept {
  return domain::make_coordinate_map<Frame::Logical, Frame::Inertial>(
      Affine2D{Affine{-1.0, 1.0, -0.3, 0.7}, Affine{-1.0, 1.0, 0.3, 0.55}});
}

template <>
auto make_affine_map<3>() noexcept {
  return domain::make_coordinate_map<Frame::Logical, Frame::Inertial>(
      Affine3D{Affine{-1.0, 1.0, -0.3, 0.7}, Affine{-1.0, 1.0, 0.3, 0.55},
               Affine{-1.0, 1.0, 2.3, 2.8}});
}

template <size_t Dim>
void test(
    const Mesh<Dim>& mesh,
    std::array<std::unique_ptr<MathFunction<1>>, Dim> functions) noexcept {
  using flux_tags = tmpl::list<
      db::add_tag_prefix<Tags::Flux, Var1, tmpl::size_t<Dim>, Frame::Inertial>,
      db::add_tag_prefix<Tags::Flux, Var2<Dim>, tmpl::size_t<Dim>,
                         Frame::Inertial>>;
  using Flux1 = tmpl::at_c<flux_tags, 0>;
  using Flux2 = tmpl::at_c<flux_tags, 1>;

  const auto coord_map = make_affine_map<Dim>();
  const auto logical_coords = logical_coordinates(mesh);
  const auto inertial_coords = coord_map(logical_coords);
  const auto inverse_jacobian = coord_map.inv_jacobian(logical_coords);

  Variables<tmpl::list<Tags::dt<Var1>, Tags::dt<Var2<Dim>>>> dt_vars(
      mesh.number_of_grid_points());
  Variables<flux_tags> fluxes(mesh.number_of_grid_points());
  Variables<tmpl::list<Tags::Source<Var1>, Tags::Source<Var2<Dim>>>> sources(
      mesh.number_of_grid_points());
  Variables<db::wrap_tags_in<Tags::div, flux_tags>> expected_div_fluxes(
      mesh.number_of_grid_points());

  MathFunctions::TensorProduct<Dim> f(1.0, std::move(functions));

  tmpl::for_each<flux_tags>(
      [&inertial_coords, &f, &fluxes, &expected_div_fluxes](auto tag) noexcept {
        using FluxTag = tmpl::type_from<decltype(tag)>;
        get<FluxTag>(fluxes) = Functions<FluxTag>::flux(f, inertial_coords);
        using DivFluxTag = Tags::div<FluxTag>;
        get<DivFluxTag>(expected_div_fluxes) =
            Functions<FluxTag>::divergence_of_flux(f, inertial_coords);
      });

  ConservativeDuDt<System<tmpl::list<>, Dim>>::apply(
      make_not_null(&dt_vars), mesh, inverse_jacobian, fluxes, sources);
  CHECK(get(get<Tags::dt<Var1>>(dt_vars)) ==
        -get(get<Tags::div<Flux1>>(expected_div_fluxes)));
  for (size_t i = 0; i < Dim; ++i) {
    CHECK(get<Tags::dt<Var2<Dim>>>(dt_vars).get(i) ==
          -get<Tags::div<Flux2>>(expected_div_fluxes).get(i));
  }

  // Test with Tags::Source<Var1>
  get(get<Tags::Source<Var1>>(sources)) = 0.9 * get<0>(inertial_coords);

  ConservativeDuDt<System<tmpl::list<Var1>, Dim>>::apply(
      make_not_null(&dt_vars), mesh, inverse_jacobian, fluxes, sources);
  CHECK(get(get<Tags::dt<Var1>>(dt_vars)) ==
        -get(get<Tags::div<Flux1>>(expected_div_fluxes)) +
            get(get<Tags::Source<Var1>>(sources)));
  for (size_t i = 0; i < Dim; ++i) {
    CHECK(get<Tags::dt<Var2<Dim>>>(dt_vars).get(i) ==
          -get<Tags::div<Flux2>>(expected_div_fluxes).get(i));
  }

  // Test with Tags::Source<Var2>
  for (size_t i = 0; i < Dim; ++i) {
    get<Tags::Source<Var2<Dim>>>(sources).get(i) =
        0.9 * (i + 1.0) * inertial_coords.get(i);
  }

  ConservativeDuDt<System<tmpl::list<Var2<Dim>>, Dim>>::apply(
      make_not_null(&dt_vars), mesh, inverse_jacobian, fluxes, sources);
  CHECK(get(get<Tags::dt<Var1>>(dt_vars)) ==
        -get(get<Tags::div<Flux1>>(expected_div_fluxes)));
  for (size_t i = 0; i < Dim; ++i) {
    CHECK(get<Tags::dt<Var2<Dim>>>(dt_vars).get(i) ==
          -get<Tags::div<Flux2>>(expected_div_fluxes).get(i) +
              get<Tags::Source<Var2<Dim>>>(sources).get(i));
  }

  // Test with both sources
  ConservativeDuDt<System<tmpl::list<Var1, Var2<Dim>>, Dim>>::apply(
      make_not_null(&dt_vars), mesh, inverse_jacobian, fluxes, sources);
  CHECK(get(get<Tags::dt<Var1>>(dt_vars)) ==
        -get(get<Tags::div<Flux1>>(expected_div_fluxes)) +
            get(get<Tags::Source<Var1>>(sources)));
  for (size_t i = 0; i < Dim; ++i) {
    CHECK(get<Tags::dt<Var2<Dim>>>(dt_vars).get(i) ==
          -get<Tags::div<Flux2>>(expected_div_fluxes).get(i) +
              get<Tags::Source<Var2<Dim>>>(sources).get(i));
  }
}

}  // namespace

SPECTRE_TEST_CASE("Unit.Evolution.ConservativeDuDt", "[Unit][Evolution]") {
  constexpr size_t num_points = 5;

  const Mesh<1> mesh_1d{num_points, Spectral::Basis::Legendre,
                        Spectral::Quadrature::GaussLobatto};
  test(mesh_1d, {{std::make_unique<MathFunctions::PowX>(num_points - 1)}});

  const Mesh<2> mesh_2d{num_points, Spectral::Basis::Legendre,
                        Spectral::Quadrature::GaussLobatto};
  test(mesh_2d, {{std::make_unique<MathFunctions::PowX>(num_points - 1),
                  std::make_unique<MathFunctions::PowX>(num_points - 1)}});

  const Mesh<3> mesh_3d{num_points, Spectral::Basis::Legendre,
                        Spectral::Quadrature::GaussLobatto};
  test(mesh_3d, {{std::make_unique<MathFunctions::PowX>(num_points - 1),
                  std::make_unique<MathFunctions::PowX>(num_points - 1),
                  std::make_unique<MathFunctions::PowX>(num_points - 1)}});
}
