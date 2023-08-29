// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <array>
#include <cmath>
#include <cstddef>
#include <memory>
#include <pup.h>
#include <string>
#include <type_traits>

#include "DataStructures/DataBox/DataBox.hpp"
#include "DataStructures/DataBox/PrefixHelpers.hpp"
#include "DataStructures/DataBox/Prefixes.hpp"  // IWYU pragma: keep
#include "DataStructures/DataBox/Tag.hpp"
#include "DataStructures/DataBox/TagName.hpp"
#include "DataStructures/DataVector.hpp"
#include "DataStructures/Index.hpp"
#include "DataStructures/IndexIterator.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "DataStructures/Variables.hpp"  // IWYU pragma: keep
#include "DataStructures/VariablesTag.hpp"
#include "Domain/CoordinateMaps/Affine.hpp"
#include "Domain/CoordinateMaps/CoordinateMap.hpp"
#include "Domain/CoordinateMaps/CoordinateMap.tpp"
#include "Domain/CoordinateMaps/ProductMaps.hpp"
#include "Domain/CoordinateMaps/ProductMaps.tpp"
#include "Domain/Tags.hpp"
#include "Helpers/DataStructures/DataBox/TestHelpers.hpp"
#include "NumericalAlgorithms/LinearOperators/PartialDerivatives.hpp"
#include "NumericalAlgorithms/LinearOperators/PartialDerivatives.tpp"
#include "NumericalAlgorithms/SpatialDiscretization/Mesh.hpp"
#include "NumericalAlgorithms/Spectral/LogicalCoordinates.hpp"
#include "NumericalAlgorithms/Spectral/Spectral.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/MakeArray.hpp"
#include "Utilities/TMPL.hpp"

// IWYU pragma: no_forward_declare Tags::deriv
// IWYU pragma: no_forward_declare Variables

namespace {
using Affine = domain::CoordinateMaps::Affine;
using Affine2D = domain::CoordinateMaps::ProductOf2Maps<Affine, Affine>;
using Affine3D = domain::CoordinateMaps::ProductOf3Maps<Affine, Affine, Affine>;

template <size_t Dim, class Frame = ::Frame::Grid>
struct Var1 : db::SimpleTag {
  using type = tnsr::i<DataVector, Dim, Frame>;
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

template <typename GradientTags, typename VariableTags, size_t Dim>
void test_logical_partial_derivative_per_tensor(
    const std::array<Variables<GradientTags>, Dim>& du,
    const Variables<VariableTags>& u, const Mesh<Dim>& mesh) {
  tmpl::for_each<GradientTags>([&du, &mesh, &u](auto gradient_tag_v) {
    using gradient_tag = tmpl::type_from<decltype(gradient_tag_v)>;
    const auto single_du =
        logical_partial_derivative(get<gradient_tag>(u), mesh);
    for (size_t storage_index = 0; storage_index < get<gradient_tag>(u).size();
         ++storage_index) {
      for (size_t d = 0; d < Dim; ++d) {
        const auto deriv_tensor_index =
            prepend(get<gradient_tag>(u).get_tensor_index(storage_index), d);
        CHECK_ITERABLE_APPROX(single_du.get(deriv_tensor_index),
                              get<gradient_tag>(gsl::at(du, d))[storage_index]);
      }
    }
    std::decay_t<decltype(single_du)> single_du_not_null{};
    DataVector buffer{mesh.number_of_grid_points()};
    gsl::span<double> buffer_view{buffer.data(), buffer.size()};
    logical_partial_derivative(make_not_null(&single_du_not_null),
                               make_not_null(&buffer_view),
                               get<gradient_tag>(u), mesh);
    CHECK_ITERABLE_APPROX(single_du_not_null, single_du);

    // Check we can do derivatives when the components of `u` aren't contiguous
    // in memory.
    // NOLINTNEXTLINE(performance-unnecessary-copy-initialization)
    std::decay_t<decltype(get<gradient_tag>(u))> non_contiguous_u =
        get<gradient_tag>(u);
    const auto non_contiguous_single_du =
        logical_partial_derivative(get<gradient_tag>(u), mesh);
    CHECK_ITERABLE_APPROX(non_contiguous_single_du, single_du);
  });
}

template <typename GradientTags, typename VariableTags, size_t Dim,
          typename DerivativeFrame>
void test_partial_derivative_per_tensor(
    const Variables<GradientTags>& du, const Variables<VariableTags>& u,
    const Mesh<Dim>& mesh,
    const InverseJacobian<DataVector, Dim, Frame::ElementLogical,
                          DerivativeFrame>& inverse_jacobian) {
  tmpl::for_each<GradientTags>(
      [&du, &mesh, &u, &inverse_jacobian](auto gradient_tag_v) {
        using gradient_tag = tmpl::type_from<decltype(gradient_tag_v)>;
        using var_tag = typename gradient_tag::tag;

        const auto single_du =
            partial_derivative(get<var_tag>(u), mesh, inverse_jacobian);

        Approx local_approx = Approx::custom().epsilon(1e-13).scale(1.0);
        CHECK_ITERABLE_CUSTOM_APPROX(single_du, get<gradient_tag>(du),
                                     local_approx);

        // Check we can do derivatives when the components of `u` aren't
        // contiguous in memory.
        // NOLINTNEXTLINE(performance-unnecessary-copy-initialization)
        const auto non_contiguous_u = get<var_tag>(u);
        const auto non_contiguous_single_du =
            partial_derivative(non_contiguous_u, mesh, inverse_jacobian);
        CHECK_ITERABLE_APPROX(non_contiguous_single_du, single_du);
      });
}

template <typename VariableTags, typename GradientTags = VariableTags>
void test_logical_partial_derivatives_1d(const Mesh<1>& mesh) {
  const size_t number_of_grid_points = mesh.number_of_grid_points();
  const DataVector& xi = Spectral::collocation_points(mesh.slice_through(0));
  Variables<VariableTags> u(number_of_grid_points);
  for (size_t a = 0; a < mesh.extents(0); ++a) {
    for (size_t n = 0; n < u.number_of_independent_components; ++n) {
      for (size_t s = 0; s < number_of_grid_points; ++s) {
        u.data()[s + n * number_of_grid_points]  // NOLINT
            = static_cast<double>(n + 1) * pow(xi[s], a);
      }
    }

    const auto helper = [&](const auto& du) {
      for (size_t n = 0;
           n < Variables<GradientTags>::number_of_independent_components; ++n) {
        for (size_t s = 0; s < number_of_grid_points; ++s) {
          const double expected =
              (0 == a ? 0.0
                      : static_cast<double>(a * (n + 1)) * pow(xi[s], a - 1));
          CHECK(du[0].data()[s + n * number_of_grid_points]  // NOLINT
                == approx(expected));
        }
      }
    };
    helper(logical_partial_derivatives<GradientTags>(u, mesh));
    std::array<Variables<GradientTags>, 1> du{};
    logical_partial_derivatives(make_not_null(&du), u, mesh);
    helper(du);
    // We've checked that du is correct, now test that taking derivatives of
    // individual tensors gets the matching result.
    test_logical_partial_derivative_per_tensor(du, u, mesh);
  }
}

template <typename VariableTags, typename GradientTags = VariableTags>
void test_logical_partial_derivatives_2d(const Mesh<2>& mesh) {
  const size_t number_of_grid_points = mesh.number_of_grid_points();
  const DataVector& xi = Spectral::collocation_points(mesh.slice_through(0));
  const DataVector& eta = Spectral::collocation_points(mesh.slice_through(1));
  Variables<VariableTags> u(mesh.number_of_grid_points());
  const size_t a = mesh.extents(0) - 1;
  const size_t b = mesh.extents(1) - 1;
  for (size_t n = 0; n < u.number_of_independent_components; ++n) {
    for (IndexIterator<2> ii(mesh.extents()); ii; ++ii) {
      u.data()[ii.collapsed_index() + n * number_of_grid_points] =  // NOLINT
          static_cast<double>(n + 1) * pow(xi[ii()[0]], a) *
          pow(eta[ii()[1]], b);
    }
  }

  const auto helper = [&](const auto& du) {
    for (size_t n = 0;
         n < Variables<GradientTags>::number_of_independent_components; ++n) {
      for (IndexIterator<2> ii(mesh.extents()); ii; ++ii) {
        const double expected_dxi =
            (0 == a ? 0.0
                    : static_cast<double>(a * (n + 1)) *
                          pow(xi[ii()[0]], a - 1) * pow(eta[ii()[1]], b));
        const double expected_deta =
            (0 == b ? 0.0
                    : static_cast<double>(b * (n + 1)) * pow(xi[ii()[0]], a) *
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
  };
  helper(logical_partial_derivatives<GradientTags>(u, mesh));
  std::array<Variables<GradientTags>, 2> du{};
  logical_partial_derivatives(make_not_null(&du), u, mesh);
  helper(du);
  // We've checked that du is correct, now test that taking derivatives of
  // individual tensors gets the matching result.
  test_logical_partial_derivative_per_tensor(du, u, mesh);
}

template <typename VariableTags, typename GradientTags = VariableTags>
void test_logical_partial_derivatives_3d(const Mesh<3>& mesh) {
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
          static_cast<double>(n + 1) * pow(xi[ii()[0]], a) *
          pow(eta[ii()[1]], b) * pow(zeta[ii()[2]], c);
    }
  }

  const auto helper = [&](const auto& du) {
    for (size_t n = 0;
         n < Variables<GradientTags>::number_of_independent_components; ++n) {
      for (IndexIterator<3> ii(mesh.extents()); ii; ++ii) {
        const double expected_dxi =
            (0 == a
                 ? 0.0
                 : static_cast<double>(a * (n + 1)) * pow(xi[ii()[0]], a - 1) *
                       pow(eta[ii()[1]], b) * pow(zeta[ii()[2]], c));
        const double expected_deta =
            (0 == b ? 0.0
                    : static_cast<double>(b * (n + 1)) * pow(xi[ii()[0]], a) *
                          pow(eta[ii()[1]], b - 1) * pow(zeta[ii()[2]], c));
        const double expected_dzeta =
            (0 == c ? 0.0
                    : static_cast<double>(c * (n + 1)) * pow(xi[ii()[0]], a) *
                          pow(eta[ii()[1]], b) * pow(zeta[ii()[2]], c - 1));
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
  };
  helper(logical_partial_derivatives<GradientTags>(u, mesh));
  std::array<Variables<GradientTags>, 3> du{};
  logical_partial_derivatives(make_not_null(&du), u, mesh);
  helper(du);
  // We've checked that du is correct, now test that taking derivatives of
  // individual tensors gets the matching result.
  test_logical_partial_derivative_per_tensor(du, u, mesh);
}

template <typename VariableTags, typename GradientTags = VariableTags>
void test_partial_derivatives_1d(const Mesh<1>& mesh) {
  const size_t number_of_grid_points = mesh.number_of_grid_points();
  const Affine x_map{-1.0, 1.0, -0.3, 0.7};
  const auto map_1d =
      domain::make_coordinate_map<Frame::ElementLogical, Frame::Grid>(
          Affine{x_map});
  const auto x = map_1d(logical_coordinates(mesh));
  const InverseJacobian<DataVector, 1, Frame::ElementLogical, Frame::Grid>
      inverse_jacobian(number_of_grid_points, 2.0);

  Variables<VariableTags> u(number_of_grid_points);
  Variables<
      db::wrap_tags_in<Tags::deriv, GradientTags, tmpl::size_t<1>, Frame::Grid>>
      expected_du(number_of_grid_points);
  for (size_t a = 0; a < mesh.extents(0); ++a) {
    tmpl::for_each<VariableTags>([&a, &x, &u](auto tag) {
      using Tag = tmpl::type_from<decltype(tag)>;
      get<Tag>(u) = Tag::f({{a}}, x);
    });
    tmpl::for_each<GradientTags>([&a, &x, &expected_du](auto tag) {
      using Tag = typename decltype(tag)::type;
      using DerivativeTag = Tags::deriv<Tag, tmpl::size_t<1>, Frame::Grid>;
      get<DerivativeTag>(expected_du) = Tag::df({{a}}, x);
    });

    const auto helper = [&](const auto& du) {
      for (size_t n = 0; n < du.size(); ++n) {
        CHECK(du.data()[n] == approx(expected_du.data()[n]));  // NOLINT
      }
    };
    helper(partial_derivatives<GradientTags>(u, mesh, inverse_jacobian));
    using vars_type =
        decltype(partial_derivatives<GradientTags>(u, mesh, inverse_jacobian));
    vars_type du{};
    partial_derivatives(make_not_null(&du), u, mesh, inverse_jacobian);
    helper(du);

    vars_type du_with_logical{};
    partial_derivatives(make_not_null(&du_with_logical),
                        logical_partial_derivatives<GradientTags>(u, mesh),
                        inverse_jacobian);
    helper(du_with_logical);

    // We've checked that du is correct, now test that taking derivatives of
    // individual tensors gets the matching result.
    test_partial_derivative_per_tensor(du, u, mesh, inverse_jacobian);
  }
}

template <typename VariableTags, typename GradientTags = VariableTags>
void test_partial_derivatives_2d(const Mesh<2>& mesh) {
  const size_t number_of_grid_points = mesh.number_of_grid_points();
  const auto prod_map2d =
      domain::make_coordinate_map<Frame::ElementLogical, Frame::Grid>(
          Affine2D{Affine{-1.0, 1.0, -0.3, 0.7}, Affine{-1.0, 1.0, 0.3, 0.55}});
  const auto x = prod_map2d(logical_coordinates(mesh));
  InverseJacobian<DataVector, 2, Frame::ElementLogical, Frame::Grid>
      inverse_jacobian(number_of_grid_points, 0.0);
  inverse_jacobian.get(0, 0) = 2.0;
  inverse_jacobian.get(1, 1) = 8.0;

  Variables<VariableTags> u(number_of_grid_points);
  Variables<
      db::wrap_tags_in<Tags::deriv, GradientTags, tmpl::size_t<2>, Frame::Grid>>
      expected_du(number_of_grid_points);
  for (size_t a = 0; a < mesh.extents(0); ++a) {
    for (size_t b = 0; b < mesh.extents(1); ++b) {
      tmpl::for_each<VariableTags>([&a, &b, &x, &u](auto tag) {
        using Tag = typename decltype(tag)::type;
        get<Tag>(u) = Tag::f({{a, b}}, x);
      });
      tmpl::for_each<GradientTags>([&a, &b, &x, &expected_du](auto tag) {
        using Tag = typename decltype(tag)::type;
        using DerivativeTag = Tags::deriv<Tag, tmpl::size_t<2>, Frame::Grid>;
        get<DerivativeTag>(expected_du) = Tag::df({{a, b}}, x);
      });

      const auto helper = [&](const auto& du) {
        for (size_t n = 0; n < du.size(); ++n) {
          CHECK(du.data()[n] ==                                  // NOLINT
                approx(expected_du.data()[n]).epsilon(1.e-13));  // NOLINT
        }
      };
      helper(partial_derivatives<GradientTags>(u, mesh, inverse_jacobian));
      using vars_type = decltype(partial_derivatives<GradientTags>(
          u, mesh, inverse_jacobian));
      vars_type du{};
      partial_derivatives(make_not_null(&du), u, mesh, inverse_jacobian);
      helper(du);

      vars_type du_with_logical{};
      partial_derivatives(make_not_null(&du_with_logical),
                          logical_partial_derivatives<GradientTags>(u, mesh),
                          inverse_jacobian);
      helper(du_with_logical);

      // We've checked that du is correct, now test that taking derivatives of
      // individual tensors gets the matching result.
      test_partial_derivative_per_tensor(du, u, mesh, inverse_jacobian);
    }
  }
}

template <typename VariableTags, typename GradientTags = VariableTags>
void test_partial_derivatives_3d(const Mesh<3>& mesh) {
  const size_t number_of_grid_points = mesh.number_of_grid_points();
  const auto prod_map3d =
      domain::make_coordinate_map<Frame::ElementLogical, Frame::Grid>(
          Affine3D{Affine{-1.0, 1.0, -0.3, 0.7}, Affine{-1.0, 1.0, 0.3, 0.55},
                   Affine{-1.0, 1.0, 2.3, 2.8}});
  const auto x = prod_map3d(logical_coordinates(mesh));
  InverseJacobian<DataVector, 3, Frame::ElementLogical, Frame::Grid>
      inverse_jacobian(number_of_grid_points, 0.0);
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
        tmpl::for_each<VariableTags>([&a, &b, &c, &x, &u](auto tag) {
          using Tag = typename decltype(tag)::type;
          get<Tag>(u) = Tag::f({{a, b, c}}, x);
        });
        tmpl::for_each<GradientTags>([&a, &b, &c, &x, &expected_du](auto tag) {
          using Tag = typename decltype(tag)::type;
          using DerivativeTag = Tags::deriv<Tag, tmpl::size_t<3>, Frame::Grid>;
          get<DerivativeTag>(expected_du) = Tag::df({{a, b, c}}, x);
        });

        const auto helper = [&](const auto& du) {
          for (size_t n = 0; n < du.size(); ++n) {
            CHECK(du.data()[n] ==  // NOLINT
                  approx(expected_du.data()[n]).epsilon(1.e-11));
          }
        };
        helper(partial_derivatives<GradientTags>(u, mesh, inverse_jacobian));
        using vars_type = decltype(partial_derivatives<GradientTags>(
            u, mesh, inverse_jacobian));
        vars_type du{};
        partial_derivatives(make_not_null(&du), u, mesh, inverse_jacobian);
        helper(du);

        vars_type du_with_logical{};
        partial_derivatives(make_not_null(&du_with_logical),
                            logical_partial_derivatives<GradientTags>(u, mesh),
                            inverse_jacobian);
        helper(du_with_logical);

        // We've checked that du is correct, now test that taking derivatives of
        // individual tensors gets the matching result.
        test_partial_derivative_per_tensor(du, u, mesh, inverse_jacobian);
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
    const Mesh<1> mesh_1d{n0, Spectral::Basis::Legendre,
                          Spectral::Quadrature::GaussLobatto};
    test_logical_partial_derivatives_1d<two_vars<1>>(mesh_1d);
    test_logical_partial_derivatives_1d<two_vars<1>, one_var<1>>(mesh_1d);
    for (size_t n1 = min_points; n1 <= max_points; ++n1) {
      const Mesh<2> mesh_2d{{{n0, n1}},
                            Spectral::Basis::Legendre,
                            Spectral::Quadrature::GaussLobatto};
      test_logical_partial_derivatives_2d<two_vars<2>>(mesh_2d);
      test_logical_partial_derivatives_2d<two_vars<2>, one_var<2>>(mesh_2d);
      for (size_t n2 = min_points; n2 <= max_points; ++n2) {
        const Mesh<3> mesh_3d{{{n0, n1, n2}},
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
  const Mesh<1> mesh_1d{n0, Spectral::Basis::Legendre,
                        Spectral::Quadrature::GaussLobatto};
  test_partial_derivatives_1d<two_vars<1>>(mesh_1d);
  test_partial_derivatives_1d<two_vars<1>, one_var<1>>(mesh_1d);
  const Mesh<2> mesh_2d{{{n0, n1}},
                        Spectral::Basis::Legendre,
                        Spectral::Quadrature::GaussLobatto};
  test_partial_derivatives_2d<two_vars<2>>(mesh_2d);
  test_partial_derivatives_2d<two_vars<2>, one_var<2>>(mesh_2d);
  const Mesh<3> mesh_3d{{{n0, n1, n2}},
                        Spectral::Basis::Legendre,
                        Spectral::Quadrature::GaussLobatto};
  test_partial_derivatives_3d<two_vars<3>>(mesh_3d);
  test_partial_derivatives_3d<two_vars<3>, one_var<3>>(mesh_3d);

  TestHelpers::db::test_prefix_tag<
      Tags::deriv<Var1<3>, tmpl::size_t<3>, Frame::Grid>>("deriv(Var1)");
  TestHelpers::db::test_prefix_tag<
      Tags::spacetime_deriv<Var1<3>, tmpl::size_t<3>, Frame::Grid>>(
      "spacetime_deriv(Var1)");

  BENCHMARK_ADVANCED("Partial derivatives")
  (Catch::Benchmark::Chronometer meter) {
    const Mesh<3> mesh{4, Spectral::Basis::Legendre,
                       Spectral::Quadrature::GaussLobatto};
    const Affine map1d(-1.0, 1.0, -1.0, 1.0);
    const domain::CoordinateMap<Frame::ElementLogical, Frame::Grid, Affine3D>
        map(Affine3D{map1d, map1d, map1d});
    const auto inv_jacobian = map.inv_jacobian(logical_coordinates(mesh));
    const Variables<tmpl::list<Var1<3>, Var2>> u{mesh.number_of_grid_points(),
                                                 0.0};
    Variables<tmpl::list<::Tags::deriv<Var1<3>, tmpl::size_t<3>, Frame::Grid>,
                         ::Tags::deriv<Var2, tmpl::size_t<3>, Frame::Grid>>>
        du{mesh.number_of_grid_points()};
    meter.measure([&du, &u, &mesh, &inv_jacobian]() {
      partial_derivatives(make_not_null(&du), u, mesh, inv_jacobian);
    });
  };
}

namespace {
template <class MapType>
struct MapTag : db::SimpleTag {
  static constexpr size_t dim = MapType::dim;
  using target_frame = typename MapType::target_frame;
  using source_frame = typename MapType::source_frame;

  using type = MapType;
};

template <typename Tag>
struct SomePrefix : db::PrefixTag, db::SimpleTag {
  using type = typename Tag::type;
  using tag = Tag;
  static std::string name() {
    return "SomePrefix(" + db::tag_name<Tag>() + ")";
  }
};

template <size_t Dim, typename T>
void test_partial_derivatives_compute_item(
    const std::array<size_t, Dim> extents_array, const T& map) {
  using vars_tags = tmpl::list<Var1<Dim>, Var2>;
  using map_tag = MapTag<std::decay_t<decltype(map)>>;
  using inv_jac_tag = domain::Tags::InverseJacobianCompute<
      map_tag, domain::Tags::LogicalCoordinates<Dim>>;
  using deriv_tag = Tags::DerivCompute<Tags::Variables<vars_tags>,
                                       domain::Tags::Mesh<Dim>, inv_jac_tag>;
  using prefixed_variables_tag =
      db::add_tag_prefix<SomePrefix, Tags::Variables<vars_tags>>;
  using deriv_prefixed_tag =
      Tags::DerivCompute<prefixed_variables_tag, domain::Tags::Mesh<Dim>,
                         inv_jac_tag, tmpl::list<SomePrefix<Var1<Dim>>>>;

  TestHelpers::db::test_compute_tag<deriv_tag>(
      "Variables(deriv(Var1),deriv(Var2))");

  const std::array<size_t, Dim> array_to_functions{extents_array -
                                                   make_array<Dim>(size_t{1})};
  const Mesh<Dim> mesh{extents_array, Spectral::Basis::Legendre,
                       Spectral::Quadrature::GaussLobatto};
  const size_t num_grid_points = mesh.number_of_grid_points();
  Variables<vars_tags> u(num_grid_points);
  Variables<
      db::wrap_tags_in<Tags::deriv, vars_tags, tmpl::size_t<Dim>, Frame::Grid>>
      expected_du(num_grid_points);
  const auto x_logical = logical_coordinates(mesh);
  const auto x = map(logical_coordinates(mesh));

  tmpl::for_each<vars_tags>([&array_to_functions, &x, &u](auto tag) {
    using Tag = tmpl::type_from<decltype(tag)>;
    get<Tag>(u) = Tag::f(array_to_functions, x);
  });
  typename prefixed_variables_tag::type prefixed_vars(u);

  tmpl::for_each<vars_tags>([&array_to_functions, &x, &expected_du](auto tag) {
    using Tag = typename decltype(tag)::type;
    using DerivativeTag = Tags::deriv<Tag, tmpl::size_t<Dim>, Frame::Grid>;
    get<DerivativeTag>(expected_du) = Tag::df(array_to_functions, x);
  });

  auto box = db::create<
      db::AddSimpleTags<domain::Tags::Mesh<Dim>, Tags::Variables<vars_tags>,
                        prefixed_variables_tag, map_tag>,
      db::AddComputeTags<domain::Tags::LogicalCoordinates<Dim>, inv_jac_tag,
                         deriv_tag, deriv_prefixed_tag>>(mesh, u, prefixed_vars,
                                                         map);

  const auto& du = db::get<deriv_tag>(box);

  for (size_t n = 0; n < du.size(); ++n) {
    // clang-tidy: pointer arithmetic
    CHECK(du.data()[n] == approx(expected_du.data()[n]));  // NOLINT
  }

  // Test prefixes are handled correctly
  const auto& du_prefixed_vars = get<db::add_tag_prefix<
      Tags::deriv,
      db::add_tag_prefix<SomePrefix, Tags::Variables<tmpl::list<Var1<Dim>>>>,
      tmpl::size_t<Dim>, Frame::Grid>>(box);
  const auto& du_prefixed =
      get<Tags::deriv<SomePrefix<Var1<Dim>>, tmpl::size_t<Dim>, Frame::Grid>>(
          du_prefixed_vars);
  const auto& expected_du_prefixed =
      get<Tags::deriv<Var1<Dim>, tmpl::size_t<Dim>, Frame::Grid>>(expected_du);
  CHECK_ITERABLE_APPROX(du_prefixed, expected_du_prefixed);
}

template <size_t Dim, typename T>
void test_partial_derivatives_tensor_compute_item(
    const std::array<size_t, Dim> extents_array, const T& map) {
  using tensor_tag = Var1<Dim>;
  using map_tag = MapTag<std::decay_t<decltype(map)>>;
  using inv_jac_tag = domain::Tags::InverseJacobianCompute<
      map_tag, domain::Tags::LogicalCoordinates<Dim>>;
  using deriv_tensor_tag = Tags::DerivTensorCompute<tensor_tag, inv_jac_tag>;

  const std::array<size_t, Dim> array_to_functions{extents_array -
                                                   make_array<Dim>(size_t{1})};
  const Mesh<Dim> mesh{extents_array, Spectral::Basis::Legendre,
                       Spectral::Quadrature::GaussLobatto};
  const auto x_logical = logical_coordinates(mesh);
  const auto x = map(logical_coordinates(mesh));

  const auto u = tensor_tag::f(array_to_functions, x);
  const auto expected_du = tensor_tag::df(array_to_functions, x);

  auto box = db::create<
      db::AddSimpleTags<domain::Tags::Mesh<Dim>, tensor_tag, map_tag>,
      db::AddComputeTags<domain::Tags::LogicalCoordinates<Dim>, inv_jac_tag,
                         deriv_tensor_tag>>(mesh, u, map);

  const auto& du = db::get<deriv_tensor_tag>(box);

  // CHECK_ITERABLE_APPROX(du, expected_du.data());
  for (size_t n = 0; n < du.size(); ++n) {
    CHECK_ITERABLE_APPROX(du[n], expected_du[n]);
  }
}
}  // namespace

SPECTRE_TEST_CASE("Unit.Numerical.LinearOperators.PartialDerivs.ComputeItems",
                  "[NumericalAlgorithms][LinearOperators][Unit]") {
  Index<3> max_extents{10, 10, 5};

  for (size_t a = 1; a < max_extents[0]; ++a) {
    test_partial_derivatives_compute_item(
        std::array<size_t, 1>{{a + 1}},
        domain::make_coordinate_map<Frame::ElementLogical, Frame::Grid>(
            Affine{-1.0, 1.0, -0.3, 0.7}));
    for (size_t b = 1; b < max_extents[1]; ++b) {
      test_partial_derivatives_compute_item(
          std::array<size_t, 2>{{a + 1, b + 1}},
          domain::make_coordinate_map<Frame::ElementLogical, Frame::Grid>(
              Affine2D{Affine{-1.0, 1.0, -0.3, 0.7},
                       Affine{-1.0, 1.0, 0.3, 0.55}}));
      for (size_t c = 1; a < max_extents[0] / 2 and b < max_extents[1] / 2 and
                         c < max_extents[2];
           ++c) {
        test_partial_derivatives_compute_item(
            std::array<size_t, 3>{{a + 1, b + 1, c + 1}},
            domain::make_coordinate_map<Frame::ElementLogical, Frame::Grid>(
                Affine3D{Affine{-1.0, 1.0, -0.3, 0.7},
                         Affine{-1.0, 1.0, 0.3, 0.55},
                         Affine{-1.0, 1.0, 2.3, 2.8}}));
      }
    }
  }
  for (size_t a = 1; a < max_extents[0]; ++a) {
    test_partial_derivatives_tensor_compute_item(
        std::array<size_t, 1>{{a + 1}},
        domain::make_coordinate_map<Frame::ElementLogical, Frame::Grid>(
            Affine{-1.0, 1.0, -0.3, 0.7}));
    for (size_t b = 1; b < max_extents[1]; ++b) {
      test_partial_derivatives_tensor_compute_item(
          std::array<size_t, 2>{{a + 1, b + 1}},
          domain::make_coordinate_map<Frame::ElementLogical, Frame::Grid>(
              Affine2D{Affine{-1.0, 1.0, -0.3, 0.7},
                       Affine{-1.0, 1.0, 0.3, 0.55}}));
      for (size_t c = 1; a < max_extents[0] / 2 and b < max_extents[1] / 2 and
                         c < max_extents[2];
           ++c) {
        test_partial_derivatives_tensor_compute_item(
            std::array<size_t, 3>{{a + 1, b + 1, c + 1}},
            domain::make_coordinate_map<Frame::ElementLogical, Frame::Grid>(
                Affine3D{Affine{-1.0, 1.0, -0.3, 0.7},
                         Affine{-1.0, 1.0, 0.3, 0.55},
                         Affine{-1.0, 1.0, 2.3, 2.8}}));
      }
    }
  }
}
