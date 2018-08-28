// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "tests/Unit/TestingFramework.hpp"

#include <array>
#include <cmath>
#include <cstddef>
#include <memory>
#include <pup.h>
#include <string>
#include <type_traits>

#include "DataStructures/DataBox/DataBox.hpp"
#include "DataStructures/DataBox/DataBoxTag.hpp"
#include "DataStructures/DataBox/Prefixes.hpp"  // IWYU pragma: keep
#include "DataStructures/DataVector.hpp"
#include "DataStructures/Index.hpp"
#include "DataStructures/IndexIterator.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "DataStructures/Variables.hpp"  // IWYU pragma: keep
#include "DataStructures/VariablesHelpers.hpp"
#include "Domain/CoordinateMaps/Affine.hpp"
#include "Domain/CoordinateMaps/CoordinateMap.hpp"
#include "Domain/CoordinateMaps/ProductMaps.hpp"
#include "Domain/LogicalCoordinates.hpp"
#include "Domain/Mesh.hpp"
#include "Domain/Tags.hpp"
#include "NumericalAlgorithms/LinearOperators/PartialDerivatives.tpp"
#include "NumericalAlgorithms/Spectral/Spectral.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/MakeArray.hpp"
#include "Utilities/StdArrayHelpers.hpp"
#include "Utilities/TMPL.hpp"

namespace {

template <size_t Dim, class Frame = ::Frame::Grid>
struct Var1 : db::SimpleTag {
  using type = tnsr::i<DataVector, Dim, Frame>;
  static std::string name() noexcept { return "Var1"; }
  static auto f(const std::array<size_t, Dim>& coeffs,
                const tnsr::I<DataVector, Dim, Frame>& x) {
    tnsr::i<DataVector, Dim, Frame> result(x.begin()->size(), 0.);
    for (size_t i = 0; i < Dim; ++i) {
      result.get(i) = (i + 2);
      for (size_t d = 0; d < Dim; ++d) {
        result.get(i) *= pow(x.get(d), gsl::at(coeffs, d));
      }
    }
    return result;
  }
  static auto df(const std::array<size_t, Dim>& coeffs,
                 const tnsr::I<DataVector, Dim, Frame>& x) {
    tnsr::ij<DataVector, Dim, Frame> result(x.begin()->size(), 0.);
    for (size_t i = 0; i < Dim; ++i) {
      for (size_t j = 0; j < Dim; ++j) {
        result.get(i, j) = (j + 2);
        for (size_t d = 0; d < Dim; ++d) {
          if (d == i) {
            if (0 == gsl::at(coeffs, d)) {
              result.get(i, j) = 0.;
            } else {
              result.get(i, j) *=
                  gsl::at(coeffs, d) * pow(x.get(d), gsl::at(coeffs, d) - 1);
            }
          } else {
            result.get(i, j) *= pow(x.get(d), gsl::at(coeffs, d));
          }
        }
      }
    }
    return result;
  }
};

struct Var2 : db::SimpleTag {
  using type = Scalar<DataVector>;
  static std::string name() noexcept { return "Var2"; }
  template <size_t Dim, class Frame>
  static auto f(const std::array<size_t, Dim>& coeffs,
                const tnsr::I<DataVector, Dim, Frame>& x) {
    Scalar<DataVector> result(x.begin()->size(), 1.);
    for (size_t d = 0; d < Dim; ++d) {
      result.get() *= pow(x.get(d), gsl::at(coeffs, d));
    }
    return result;
  }
  template <size_t Dim, class Frame>
  static auto df(const std::array<size_t, Dim>& coeffs,
                 const tnsr::I<DataVector, Dim, Frame>& x) {
    tnsr::i<DataVector, Dim, Frame> result(x.begin()->size(), 1.);
    for (size_t i = 0; i < Dim; ++i) {
      for (size_t d = 0; d < Dim; ++d) {
        if (d == i) {
          if (0 == gsl::at(coeffs, d)) {
            result.get(i) = 0.0;
          } else {
            result.get(i) *=
                gsl::at(coeffs, d) * pow(x.get(d), gsl::at(coeffs, d) - 1);
          }
        } else {
          result.get(i) *= pow(x.get(d), gsl::at(coeffs, d));
        }
      }
    }
    return result;
  }
};

template <size_t Dim>
using two_vars = tmpl::list<Var1<Dim>, Var2>;

template <size_t Dim>
using one_var = tmpl::list<Var1<Dim>>;

template <typename VariableTags, typename GradientTags = VariableTags>
void test_logical_partial_derivatives_1d(const domain::Mesh<1>& mesh) {
  const size_t number_of_grid_points = mesh.number_of_grid_points();
  const DataVector& xi = Spectral::collocation_points(mesh.slice_through(0));
  Variables<VariableTags> u(number_of_grid_points);
  for (size_t a = 0; a < mesh.extents(0); ++a) {
    for (size_t n = 0; n < u.number_of_independent_components; ++n) {
      for (size_t s = 0; s < number_of_grid_points; ++s) {
        u.data()[s + n * number_of_grid_points]  // NOLINT
            = (n + 1) * pow(xi[s], a);
      }
    }

    const auto du = logical_partial_derivatives<GradientTags>(u, mesh);

    for (size_t n = 0;
         n < Variables<GradientTags>::number_of_independent_components; ++n) {
      for (size_t s = 0; s < number_of_grid_points; ++s) {
        const double expected =
            (0 == a ? 0.0 : a * (n + 1) * pow(xi[s], a - 1));
        CHECK(du[0].data()[s + n * number_of_grid_points]  // NOLINT
              == approx(expected));
      }
    }
  }
}

template <typename VariableTags, typename GradientTags = VariableTags>
void test_logical_partial_derivatives_2d(const domain::Mesh<2>& mesh) {
  const size_t number_of_grid_points = mesh.number_of_grid_points();
  const DataVector& xi = Spectral::collocation_points(mesh.slice_through(0));
  const DataVector& eta = Spectral::collocation_points(mesh.slice_through(1));
  Variables<VariableTags> u(mesh.number_of_grid_points());
  const size_t a = mesh.extents(0) - 1;
  const size_t b = mesh.extents(1) - 1;
  for (size_t n = 0; n < u.number_of_independent_components; ++n) {
    for (IndexIterator<2> ii(mesh.extents()); ii; ++ii) {
      u.data()[ii.collapsed_index() + n * number_of_grid_points] =  // NOLINT
          (n + 1) * pow(xi[ii()[0]], a) * pow(eta[ii()[1]], b);
    }
  }

  const auto du = logical_partial_derivatives<GradientTags>(u, mesh);

  for (size_t n = 0;
       n < Variables<GradientTags>::number_of_independent_components; ++n) {
    for (IndexIterator<2> ii(mesh.extents()); ii; ++ii) {
      const double expected_dxi =
          (0 == a
               ? 0.0
               : a * (n + 1) * pow(xi[ii()[0]], a - 1) * pow(eta[ii()[1]], b));
      const double expected_deta = (0 == b ? 0.0
                                           : b * (n + 1) * pow(xi[ii()[0]], a) *
                                                 pow(eta[ii()[1]], b - 1));
      // clang-tidy: pointer arithmetic
      CHECK(du[0].data()[ii.collapsed_index() +         // NOLINT
                         n * number_of_grid_points] ==  // NOLINT
            approx(expected_dxi));
      CHECK(du[1].data()[ii.collapsed_index() +         // NOLINT
                         n * number_of_grid_points] ==  // NOLINT
            approx(expected_deta));
    }
  }
}

template <typename VariableTags, typename GradientTags = VariableTags>
void test_logical_partial_derivatives_3d(const domain::Mesh<3>& mesh) {
  const size_t number_of_grid_points = mesh.number_of_grid_points();
  const DataVector& xi = Spectral::collocation_points(mesh.slice_through(0));
  const DataVector& eta = Spectral::collocation_points(mesh.slice_through(1));
  const DataVector& zeta = Spectral::collocation_points(mesh.slice_through(2));
  Variables<VariableTags> u(number_of_grid_points);
  const size_t a = mesh.extents(0) - 1;
  const size_t b = mesh.extents(1) - 1;
  const size_t c = mesh.extents(2) - 1;
  for (size_t n = 0; n < u.number_of_independent_components; ++n) {
    for (IndexIterator<3> ii(mesh.extents()); ii; ++ii) {
      u.data()[ii.collapsed_index() + n * number_of_grid_points] =  // NOLINT
          (n + 1) * pow(xi[ii()[0]], a) * pow(eta[ii()[1]], b) *
          pow(zeta[ii()[2]], c);
    }
  }

  const auto du = logical_partial_derivatives<GradientTags>(u, mesh);

  for (size_t n = 0;
       n < Variables<GradientTags>::number_of_independent_components; ++n) {
    for (IndexIterator<3> ii(mesh.extents()); ii; ++ii) {
      const double expected_dxi =
          (0 == a ? 0.0
                  : a * (n + 1) * pow(xi[ii()[0]], a - 1) *
                        pow(eta[ii()[1]], b) * pow(zeta[ii()[2]], c));
      const double expected_deta =
          (0 == b ? 0.0
                  : b * (n + 1) * pow(xi[ii()[0]], a) *
                        pow(eta[ii()[1]], b - 1) * pow(zeta[ii()[2]], c));
      const double expected_dzeta =
          (0 == c ? 0.0
                  : c * (n + 1) * pow(xi[ii()[0]], a) * pow(eta[ii()[1]], b) *
                        pow(zeta[ii()[2]], c - 1));
      // clang-tidy: pointer arithmetic
      CHECK(du[0].data()[ii.collapsed_index() +         // NOLINT
                         n * number_of_grid_points] ==  // NOLINT
            approx(expected_dxi));
      CHECK(du[1].data()[ii.collapsed_index() +         // NOLINT
                         n * number_of_grid_points] ==  // NOLINT
            approx(expected_deta));
      CHECK(du[2].data()[ii.collapsed_index() +         // NOLINT
                         n * number_of_grid_points] ==  // NOLINT
            approx(expected_dzeta));
    }
  }
}

template <typename VariableTags, typename GradientTags = VariableTags>
void test_partial_derivatives_1d(const domain::Mesh<1>& mesh) {
  const size_t number_of_grid_points = mesh.number_of_grid_points();
  const domain::CoordinateMaps::Affine x_map{-1.0, 1.0, -0.3, 0.7};
  const auto map_1d =
      domain::make_coordinate_map<Frame::Logical, Frame::Grid>(x_map);
  const auto x = map_1d(logical_coordinates(mesh));
  const InverseJacobian<DataVector, 1, Frame::Logical, Frame::Grid>
      inverse_jacobian(number_of_grid_points, 2.0);

  Variables<VariableTags> u(number_of_grid_points);
  Variables<
      db::wrap_tags_in<Tags::deriv, GradientTags, tmpl::size_t<1>, Frame::Grid>>
      expected_du(number_of_grid_points);
  for (size_t a = 0; a < mesh.extents(0); ++a) {
    tmpl::for_each<VariableTags>([&a, &x, &u ](auto tag) noexcept {
      using Tag = tmpl::type_from<decltype(tag)>;
      get<Tag>(u) = Tag::f({{a}}, x);
    });
    tmpl::for_each<GradientTags>([&a, &x, &expected_du ](auto tag) noexcept {
      using Tag = typename decltype(tag)::type;
      using DerivativeTag = Tags::deriv<Tag, tmpl::size_t<1>, Frame::Grid>;
      get<DerivativeTag>(expected_du) = Tag::df({{a}}, x);
    });

    const auto du =
        partial_derivatives<GradientTags>(u, mesh, inverse_jacobian);

    for (size_t n = 0; n < du.size(); ++n) {
      CAPTURE_PRECISE(du.data()[n] - expected_du.data()[n]);  // NOLINT
      CHECK(du.data()[n] == approx(expected_du.data()[n]));   // NOLINT
    }
  }
}

template <typename VariableTags, typename GradientTags = VariableTags>
void test_partial_derivatives_2d(const domain::Mesh<2>& mesh) {
  using affine_map = domain::CoordinateMaps::Affine;
  using affine_map_2d =
      domain::CoordinateMaps::ProductOf2Maps<affine_map, affine_map>;
  const size_t number_of_grid_points = mesh.number_of_grid_points();
  const auto prod_map2d =
      domain::make_coordinate_map<Frame::Logical, Frame::Grid>(affine_map_2d{
          affine_map{-1.0, 1.0, -0.3, 0.7}, affine_map{-1.0, 1.0, 0.3, 0.55}});
  const auto x = prod_map2d(domain::logical_coordinates(mesh));
  InverseJacobian<DataVector, 2, Frame::Logical, Frame::Grid> inverse_jacobian(
      number_of_grid_points, 0.0);
  inverse_jacobian.get(0, 0) = 2.0;
  inverse_jacobian.get(1, 1) = 8.0;

  Variables<VariableTags> u(number_of_grid_points);
  Variables<
      db::wrap_tags_in<Tags::deriv, GradientTags, tmpl::size_t<2>, Frame::Grid>>
      expected_du(number_of_grid_points);
  for (size_t a = 0; a < mesh.extents(0); ++a) {
    for (size_t b = 0; b < mesh.extents(1); ++b) {
      tmpl::for_each<VariableTags>([&a, &b, &x, &u ](auto tag) noexcept {
        using Tag = typename decltype(tag)::type;
        get<Tag>(u) = Tag::f({{a, b}}, x);
      });
      tmpl::for_each<GradientTags>([&a, &b, &x,
                                    &expected_du ](auto tag) noexcept {
        using Tag = typename decltype(tag)::type;
        using DerivativeTag = Tags::deriv<Tag, tmpl::size_t<2>, Frame::Grid>;
        get<DerivativeTag>(expected_du) = Tag::df({{a, b}}, x);
      });

      const auto du =
          partial_derivatives<GradientTags>(u, mesh, inverse_jacobian);

      for (size_t n = 0; n < du.size(); ++n) {
        CAPTURE_PRECISE(du.data()[n] - expected_du.data()[n]);  // NOLINT
        CHECK(du.data()[n] ==                                   // NOLINT
              approx(expected_du.data()[n]).epsilon(1.e-13));   // NOLINT
      }
    }
  }
}

template <typename VariableTags, typename GradientTags = VariableTags>
void test_partial_derivatives_3d(const domain::Mesh<3>& mesh) {
  using affine_map = domain::CoordinateMaps::Affine;
  using affine_map_3d =
      domain::CoordinateMaps::ProductOf3Maps<affine_map, affine_map,
                                             affine_map>;
  const size_t number_of_grid_points = mesh.number_of_grid_points();
  const auto prod_map3d =
      domain::make_coordinate_map<Frame::Logical, Frame::Grid>(affine_map_3d{
          affine_map{-1.0, 1.0, -0.3, 0.7}, affine_map{-1.0, 1.0, 0.3, 0.55},
          affine_map{-1.0, 1.0, 2.3, 2.8}});
  const auto x = prod_map3d(domain::logical_coordinates(mesh));
  InverseJacobian<DataVector, 3, Frame::Logical, Frame::Grid> inverse_jacobian(
      number_of_grid_points, 0.0);
  inverse_jacobian.get(0, 0) = 2.0;
  inverse_jacobian.get(1, 1) = 8.0;
  inverse_jacobian.get(2, 2) = 4.0;

  Variables<VariableTags> u(number_of_grid_points);
  Variables<
      db::wrap_tags_in<Tags::deriv, GradientTags, tmpl::size_t<3>, Frame::Grid>>
      expected_du(number_of_grid_points);
  for (size_t a = 0; a < mesh.extents(0) / 2; ++a) {
    for (size_t b = 0; b < mesh.extents(1) / 2; ++b) {
      for (size_t c = 0; c < mesh.extents(2) / 2; ++c) {
        tmpl::for_each<VariableTags>([&a, &b, &c, &x, &u ](auto tag) noexcept {
          using Tag = typename decltype(tag)::type;
          get<Tag>(u) = Tag::f({{a, b, c}}, x);
        });
        tmpl::for_each<GradientTags>([&a, &b, &c, &x,
                                      &expected_du ](auto tag) noexcept {
          using Tag = typename decltype(tag)::type;
          using DerivativeTag = Tags::deriv<Tag, tmpl::size_t<3>, Frame::Grid>;
          get<DerivativeTag>(expected_du) = Tag::df({{a, b, c}}, x);
        });

        const auto du =
            partial_derivatives<GradientTags>(u, mesh, inverse_jacobian);

        for (size_t n = 0; n < du.size(); ++n) {
          CAPTURE_PRECISE(du.data()[n] - expected_du.data()[n]);  // NOLINT
          CHECK(du.data()[n] ==                                   // NOLINT
                approx(expected_du.data()[n]).epsilon(1.e-11));
        }
      }
    }
  }
}
}  // namespace

SPECTRE_TEST_CASE("Unit.Numerical.LinearOperators.LogicalDerivs",
                  "[NumericalAlgorithms][LinearOperators][Unit]") {
  constexpr size_t min_points =
      Spectral::minimum_number_of_points<Spectral::Basis::Legendre,
                                         Spectral::Quadrature::GaussLobatto>;
  constexpr size_t max_points =
      Spectral::maximum_number_of_points<Spectral::Basis::Legendre> / 2;
  for (size_t n0 = min_points; n0 <= max_points; ++n0) {
    const domain::Mesh<1> mesh_1d{n0, Spectral::Basis::Legendre,
                                  Spectral::Quadrature::GaussLobatto};
    test_logical_partial_derivatives_1d<two_vars<1>>(mesh_1d);
    test_logical_partial_derivatives_1d<two_vars<1>, one_var<1>>(mesh_1d);
    for (size_t n1 = min_points; n1 <= max_points; ++n1) {
      const domain::Mesh<2> mesh_2d{{{n0, n1}},
                                    Spectral::Basis::Legendre,
                                    Spectral::Quadrature::GaussLobatto};
      test_logical_partial_derivatives_2d<two_vars<2>>(mesh_2d);
      test_logical_partial_derivatives_2d<two_vars<2>, one_var<2>>(mesh_2d);
      for (size_t n2 = min_points; n2 <= max_points; ++n2) {
        const domain::Mesh<3> mesh_3d{{{n0, n1, n2}},
                                      Spectral::Basis::Legendre,
                                      Spectral::Quadrature::GaussLobatto};
        test_logical_partial_derivatives_3d<two_vars<3>>(mesh_3d);
        test_logical_partial_derivatives_3d<two_vars<3>, one_var<3>>(mesh_3d);
      }
    }
  }
}

SPECTRE_TEST_CASE("Unit.Numerical.LinearOperators.PartialDerivs",
                  "[NumericalAlgorithms][LinearOperators][Unit]") {
  const size_t n0 =
      Spectral::maximum_number_of_points<Spectral::Basis::Legendre> / 2;
  const size_t n1 =
      Spectral::maximum_number_of_points<Spectral::Basis::Legendre> / 2 + 1;
  const size_t n2 =
      Spectral::maximum_number_of_points<Spectral::Basis::Legendre> / 2 - 1;
  const domain::Mesh<1> mesh_1d{n0, Spectral::Basis::Legendre,
                                Spectral::Quadrature::GaussLobatto};
  test_partial_derivatives_1d<two_vars<1>>(mesh_1d);
  test_partial_derivatives_1d<two_vars<1>, one_var<1>>(mesh_1d);
  const domain::Mesh<2> mesh_2d{{{n0, n1}},
                                Spectral::Basis::Legendre,
                                Spectral::Quadrature::GaussLobatto};
  test_partial_derivatives_2d<two_vars<2>>(mesh_2d);
  test_partial_derivatives_2d<two_vars<2>, one_var<2>>(mesh_2d);
  const domain::Mesh<3> mesh_3d{{{n0, n1, n2}},
                                Spectral::Basis::Legendre,
                                Spectral::Quadrature::GaussLobatto};
  test_partial_derivatives_3d<two_vars<3>>(mesh_3d);
  test_partial_derivatives_3d<two_vars<3>, one_var<3>>(mesh_3d);
}

namespace {
template <size_t Dim>
void test_logical_derivatives_compute_item(
    const std::array<size_t, Dim> extents_array) noexcept {
  using vars_tags = tmpl::list<Var1<Dim, Frame::Logical>, Var2>;
  using deriv_tag =
      Tags::deriv<vars_tags, vars_tags, std::integral_constant<size_t, Dim>>;

  const std::array<size_t, Dim> array_to_functions{extents_array -
                                                   make_array<Dim>(size_t{1})};
  const domain::Mesh<Dim> mesh{extents_array, Spectral::Basis::Legendre,
                               Spectral::Quadrature::GaussLobatto};
  const size_t num_grid_points = mesh.number_of_grid_points();
  Variables<vars_tags> u(num_grid_points);
  Variables<db::wrap_tags_in<Tags::deriv, vars_tags, tmpl::size_t<Dim>,
                             Frame::Logical>>
      expected_du(num_grid_points);
  const auto x = domain::logical_coordinates(mesh);

  tmpl::for_each<vars_tags>([&array_to_functions, &x, &u ](auto tag) noexcept {
    using Tag = tmpl::type_from<decltype(tag)>;
    get<Tag>(u) = Tag::f(array_to_functions, x);
  });
  tmpl::for_each<vars_tags>([&array_to_functions, &x,
                             &expected_du ](auto tag) noexcept {
    using Tag = typename decltype(tag)::type;
    using DerivativeTag = Tags::deriv<Tag, tmpl::size_t<Dim>, Frame::Logical>;
    get<DerivativeTag>(expected_du) = Tag::df(array_to_functions, x);
  });

  auto box = db::create<
      db::AddSimpleTags<domain::Tags::Mesh<Dim>, Tags::Variables<vars_tags>>,
      db::AddComputeTags<domain::Tags::LogicalCoordinates<Dim>, deriv_tag>>(
      mesh, u);

  const auto& du = db::get<deriv_tag>(box);

  tmpl::for_each<vars_tags>([&du, &expected_du, &mesh ](auto tag) noexcept {
    using Tag = tmpl::type_from<decltype(tag)>;
    using DerivativeTag = Tags::deriv<Tag, tmpl::size_t<Dim>, Frame::Logical>;
    auto& expected_dvariable = get<DerivativeTag>(expected_du);
    for (auto it = expected_dvariable.begin(); it != expected_dvariable.end();
         ++it) {
      const auto deriv_indices = expected_dvariable.get_tensor_index(it);
      const size_t deriv_index = deriv_indices[0];
      const auto tensor_indices =
          all_but_specified_element_of(deriv_indices, 0);
      for (size_t n = 0; n < mesh.number_of_grid_points(); ++n) {
        CAPTURE_PRECISE(get<Tag>(du[deriv_index]).get(tensor_indices)[n] -
                        (*it)[n]);
        CHECK(get<Tag>(du[deriv_index]).get(tensor_indices)[n] ==
              approx((*it)[n]));
      }
    }
  });
}

template <class MapType>
struct MapTag : db::SimpleTag {
  using type = MapType;
  static std::string name() noexcept { return "MapTag"; }
};

template <size_t Dim, typename T>
void test_partial_derivatives_compute_item(
    const std::array<size_t, Dim> extents_array, const T& map) noexcept {
  using vars_tags = tmpl::list<Var1<Dim>, Var2>;
  using map_tag = MapTag<std::decay_t<decltype(map)>>;
  using inv_jac_tag =
      domain::Tags::InverseJacobian<map_tag,
                                    domain::Tags::LogicalCoordinates<Dim>>;
  using deriv_tag = Tags::deriv<vars_tags, vars_tags, inv_jac_tag>;

  const std::array<size_t, Dim> array_to_functions{extents_array -
                                                   make_array<Dim>(size_t{1})};
  const domain::Mesh<Dim> mesh{extents_array, Spectral::Basis::Legendre,
                               Spectral::Quadrature::GaussLobatto};
  const size_t num_grid_points = mesh.number_of_grid_points();
  Variables<vars_tags> u(num_grid_points);
  Variables<
      db::wrap_tags_in<Tags::deriv, vars_tags, tmpl::size_t<Dim>, Frame::Grid>>
      expected_du(num_grid_points);
  const auto x_logical = domain::logical_coordinates(mesh);
  const auto x = map(x_logical);

  tmpl::for_each<vars_tags>([&array_to_functions, &x, &u ](auto tag) noexcept {
    using Tag = tmpl::type_from<decltype(tag)>;
    get<Tag>(u) = Tag::f(array_to_functions, x);
  });
  tmpl::for_each<vars_tags>(
      [&array_to_functions, &x, &expected_du ](auto tag) noexcept {
        using Tag = typename decltype(tag)::type;
        using DerivativeTag = Tags::deriv<Tag, tmpl::size_t<Dim>, Frame::Grid>;
        get<DerivativeTag>(expected_du) = Tag::df(array_to_functions, x);
      });

  auto box =
      db::create<db::AddSimpleTags<domain::Tags::Mesh<Dim>,
                                   Tags::Variables<vars_tags>, map_tag>,
                 db::AddComputeTags<domain::Tags::LogicalCoordinates<Dim>,
                                    inv_jac_tag, deriv_tag>>(mesh, u, map);

  const auto& du = db::get<deriv_tag>(box);

  for (size_t n = 0; n < du.size(); ++n) {
    // clang-tidy: pointer arithmetic
    CAPTURE_PRECISE(du.data()[n] - expected_du.data()[n]);  // NOLINT
    CHECK(du.data()[n] == approx(expected_du.data()[n]));   // NOLINT
  }
}
}  // namespace

SPECTRE_TEST_CASE("Unit.Numerical.LinearOperators.LogicalDerivs.ComputeItems",
                  "[NumericalAlgorithms][LinearOperators][Unit]") {
  Index<3> max_extents{10, 10, 5};

  for (size_t a = 1; a < max_extents[0]; ++a) {
    test_logical_derivatives_compute_item(std::array<size_t, 1>{{a + 1}});
    for (size_t b = 1; b < max_extents[1]; ++b) {
      test_logical_derivatives_compute_item(
          std::array<size_t, 2>{{a + 1, b + 1}});
      for (size_t c = 1; a < max_extents[0] / 2 and b < max_extents[1] / 2 and
                         c < max_extents[2];
           ++c) {
        test_logical_derivatives_compute_item(
            std::array<size_t, 3>{{a + 1, b + 1, c + 1}});
      }
    }
  }
}

SPECTRE_TEST_CASE("Unit.Numerical.LinearOperators.PartialDerivs.ComputeItems",
                  "[NumericalAlgorithms][LinearOperators][Unit]") {
  using Affine = domain::CoordinateMaps::Affine;
  using Affine2d = domain::CoordinateMaps::ProductOf2Maps<Affine, Affine>;
  using Affine3d =
      domain::CoordinateMaps::ProductOf3Maps<Affine, Affine, Affine>;

  Index<3> max_extents{10, 10, 5};

  for (size_t a = 1; a < max_extents[0]; ++a) {
    test_partial_derivatives_compute_item(
        std::array<size_t, 1>{{a + 1}},
        domain::make_coordinate_map<Frame::Logical, Frame::Grid>(
            domain::CoordinateMaps::Affine{-1.0, 1.0, -0.3, 0.7}));
    for (size_t b = 1; b < max_extents[1]; ++b) {
      test_partial_derivatives_compute_item(
          std::array<size_t, 2>{{a + 1, b + 1}},
          domain::make_coordinate_map<Frame::Logical, Frame::Grid>(Affine2d{
              Affine{-1.0, 1.0, -0.3, 0.7}, Affine{-1.0, 1.0, 0.3, 0.55}}));
      for (size_t c = 1; a < max_extents[0] / 2 and b < max_extents[1] / 2 and
                         c < max_extents[2];
           ++c) {
        test_partial_derivatives_compute_item(
            std::array<size_t, 3>{{a + 1, b + 1, c + 1}},
            domain::make_coordinate_map<Frame::Logical, Frame::Grid>(Affine3d{
                Affine{-1.0, 1.0, -0.3, 0.7}, Affine{-1.0, 1.0, 0.3, 0.55},
                Affine{-1.0, 1.0, 2.3, 2.8}}));
      }
    }
  }
}
