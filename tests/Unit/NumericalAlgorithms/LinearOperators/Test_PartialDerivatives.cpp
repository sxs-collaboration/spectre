// Distributed under the MIT License.
// See LICENSE.txt for details.

#include <catch.hpp>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Index.hpp"
#include "DataStructures/IndexIterator.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "DataStructures/Tensor/TypeAliases.hpp"
#include "DataStructures/Variables.hpp"
#include "Domain/CoordinateMaps/Affine.hpp"
#include "Domain/CoordinateMaps/CoordinateMap.hpp"
#include "Domain/CoordinateMaps/ProductMaps.hpp"
#include "Domain/LogicalCoordinates.hpp"
#include "NumericalAlgorithms/LinearOperators/PartialDerivatives.hpp"
#include "NumericalAlgorithms/Spectral/LegendreGaussLobatto.hpp"
#include "Utilities/MakeArray.hpp"
#include "tests/Unit/TestHelpers.hpp"

#include "DataStructures/DataBox/DataBox.hpp"
#include "Domain/Tags.hpp"

#include "NumericalAlgorithms/LinearOperators/PartialDerivatives.cpp"

namespace {

template <size_t Dim, class Frame = ::Frame::Grid>
struct Var1 : db::DataBoxTag {
  using type = tnsr::i<DataVector, Dim, Frame>;
  static constexpr db::DataBoxString label = "Var1";
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

struct Var2 : db::DataBoxTag {
  using type = Scalar<DataVector>;
  static constexpr db::DataBoxString label = "Var2";
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
using two_vars = typelist<Var1<Dim>, Var2>;

template <size_t Dim>
using one_var = typelist<Var1<Dim>>;

template <typename VariableTags, typename GradientTags = VariableTags>
void test_logical_partial_derivatives_1d(const Index<1>& extents) {
  const size_t number_of_grid_points = extents.product();
  const DataVector& xi = Basis::lgl::collocation_points(extents[0]);
  Variables<VariableTags> u(number_of_grid_points);
  for (size_t a = 0; a < extents[0]; ++a) {
    for (size_t n = 0; n < u.number_of_independent_components; ++n) {
      for (size_t s = 0; s < number_of_grid_points; ++s) {
        u.data()[s + n * number_of_grid_points]  // NOLINT
            = (n + 1) * pow(xi[s], a);
      }
    }

    const auto du = logical_partial_derivatives<GradientTags>(u, extents);

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
void test_logical_partial_derivatives_2d(const Index<2>& extents) {
  const size_t number_of_grid_points = extents.product();
  const DataVector& xi = Basis::lgl::collocation_points(extents[0]);
  const DataVector& eta = Basis::lgl::collocation_points(extents[1]);
  Variables<VariableTags> u(number_of_grid_points);
  const size_t a = extents[0] - 1;
  const size_t b = extents[1] - 1;
  for (size_t n = 0; n < u.number_of_independent_components; ++n) {
    for (IndexIterator<2> ii(extents); ii; ++ii) {
      u.data()[ii.offset() + n * number_of_grid_points] =  // NOLINT
          (n + 1) * pow(xi[ii()[0]], a) * pow(eta[ii()[1]], b);
    }
  }

  const auto du = logical_partial_derivatives<GradientTags>(u, extents);

  for (size_t n = 0;
       n < Variables<GradientTags>::number_of_independent_components; ++n) {
    for (IndexIterator<2> ii(extents); ii; ++ii) {
      const double expected_dxi =
          (0 == a
               ? 0.0
               : a * (n + 1) * pow(xi[ii()[0]], a - 1) * pow(eta[ii()[1]], b));
      const double expected_deta = (0 == b ? 0.0
                                           : b * (n + 1) * pow(xi[ii()[0]], a) *
                                                 pow(eta[ii()[1]], b - 1));
      CHECK(du[0].data()[ii.offset() + n * number_of_grid_points] ==  // NOLINT
            approx(expected_dxi));
      CHECK(du[1].data()[ii.offset() + n * number_of_grid_points] ==  // NOLINT
            approx(expected_deta));
    }
  }
}

template <typename VariableTags, typename GradientTags = VariableTags>
void test_logical_partial_derivatives_3d(const Index<3>& extents) {
  const size_t number_of_grid_points = extents.product();
  const DataVector& xi = Basis::lgl::collocation_points(extents[0]);
  const DataVector& eta = Basis::lgl::collocation_points(extents[1]);
  const DataVector& zeta = Basis::lgl::collocation_points(extents[2]);
  Variables<VariableTags> u(number_of_grid_points);
  const size_t a = extents[0] - 1;
  const size_t b = extents[1] - 1;
  const size_t c = extents[2] - 1;
  for (size_t n = 0; n < u.number_of_independent_components; ++n) {
    for (IndexIterator<3> ii(extents); ii; ++ii) {
      u.data()[ii.offset() + n * number_of_grid_points] =  // NOLINT
          (n + 1) * pow(xi[ii()[0]], a) * pow(eta[ii()[1]], b) *
          pow(zeta[ii()[2]], c);
    }
  }

  const auto du = logical_partial_derivatives<GradientTags>(u, extents);

  for (size_t n = 0;
       n < Variables<GradientTags>::number_of_independent_components; ++n) {
    for (IndexIterator<3> ii(extents); ii; ++ii) {
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
      CHECK(du[0].data()[ii.offset() + n * number_of_grid_points] ==  // NOLINT
            approx(expected_dxi));
      CHECK(du[1].data()[ii.offset() + n * number_of_grid_points] ==  // NOLINT
            approx(expected_deta));
      CHECK(du[2].data()[ii.offset() + n * number_of_grid_points] ==  // NOLINT
            approx(expected_dzeta));
    }
  }
}

template <typename VariableTags, typename GradientTags = VariableTags>
void test_partial_derivatives_1d(const Index<1>& extents) {
  const size_t number_of_grid_points = extents.product();
  const CoordinateMaps::Affine x_map{-1.0, 1.0, -0.3, 0.7};
  const auto map_1d = make_coordinate_map<Frame::Logical, Frame::Grid>(
      CoordinateMaps::Affine{x_map});
  const auto x = map_1d(logical_coordinates(extents));
  const InverseJacobian<1, Frame::Logical, Frame::Grid> inverse_jacobian(
      extents.product(), 2.0);

  Variables<VariableTags> u(number_of_grid_points);
  Variables<
      db::wrap_tags_in<Tags::deriv, GradientTags, tmpl::size_t<1>, Frame::Grid>>
      expected_du(number_of_grid_points);
  for (size_t a = 0; a < extents[0]; ++a) {
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
        partial_derivatives<GradientTags>(u, extents, inverse_jacobian);

    for (size_t n = 0; n < du.size(); ++n) {
      CAPTURE_PRECISE(du.data()[n] - expected_du.data()[n]);  // NOLINT
      CHECK(du.data()[n] == approx(expected_du.data()[n]));   // NOLINT
    }
  }
}

template <typename VariableTags, typename GradientTags = VariableTags>
void test_partial_derivatives_2d(const Index<2>& extents) {
  using affine_map = CoordinateMaps::Affine;
  using affine_map_2d = CoordinateMaps::ProductOf2Maps<affine_map, affine_map>;
  const size_t number_of_grid_points = extents.product();
  const auto prod_map2d =
      make_coordinate_map<Frame::Logical, Frame::Grid>(affine_map_2d{
          affine_map{-1.0, 1.0, -0.3, 0.7}, affine_map{-1.0, 1.0, 0.3, 0.55}});
  const auto x = prod_map2d(logical_coordinates(extents));
  InverseJacobian<2, Frame::Logical, Frame::Grid> inverse_jacobian(
      number_of_grid_points, 0.0);
  inverse_jacobian.get(0, 0) = 2.0;
  inverse_jacobian.get(1, 1) = 8.0;

  Variables<VariableTags> u(number_of_grid_points);
  Variables<
      db::wrap_tags_in<Tags::deriv, GradientTags, tmpl::size_t<2>, Frame::Grid>>
      expected_du(number_of_grid_points);
  for (size_t a = 0; a < extents[0]; ++a) {
    for (size_t b = 0; b < extents[1]; ++b) {
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
          partial_derivatives<GradientTags>(u, extents, inverse_jacobian);

      for (size_t n = 0; n < du.size(); ++n) {
        CAPTURE_PRECISE(du.data()[n] - expected_du.data()[n]);  // NOLINT
        CHECK(du.data()[n] ==                                   // NOLINT
              approx(expected_du.data()[n]).epsilon(1.e-13));   // NOLINT
      }
    }
  }
}

template <typename VariableTags, typename GradientTags = VariableTags>
void test_partial_derivatives_3d(const Index<3>& extents) {
  using affine_map = CoordinateMaps::Affine;
  using affine_map_3d =
      CoordinateMaps::ProductOf3Maps<affine_map, affine_map, affine_map>;
  const size_t number_of_grid_points = extents.product();
  const auto prod_map3d =
      make_coordinate_map<Frame::Logical, Frame::Grid>(affine_map_3d{
          affine_map{-1.0, 1.0, -0.3, 0.7}, affine_map{-1.0, 1.0, 0.3, 0.55},
          affine_map{-1.0, 1.0, 2.3, 2.8}});
  const auto x = prod_map3d(logical_coordinates(extents));
  InverseJacobian<3, Frame::Logical, Frame::Grid> inverse_jacobian(
      extents.product(), 0.0);
  inverse_jacobian.get(0, 0) = 2.0;
  inverse_jacobian.get(1, 1) = 8.0;
  inverse_jacobian.get(2, 2) = 4.0;

  Variables<VariableTags> u(number_of_grid_points);
  Variables<
      db::wrap_tags_in<Tags::deriv, GradientTags, tmpl::size_t<3>, Frame::Grid>>
      expected_du(number_of_grid_points);
  for (size_t a = 0; a < extents[0] / 2; ++a) {
    for (size_t b = 0; b < extents[1] / 2; ++b) {
      for (size_t c = 0; c < extents[2] / 2; ++c) {
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
            partial_derivatives<GradientTags>(u, extents, inverse_jacobian);

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
  for (size_t n0 = 2; n0 <= Basis::lgl::maximum_number_of_pts / 2; ++n0) {
    const Index<1> extents_1d(n0);
    test_logical_partial_derivatives_1d<two_vars<1>>(extents_1d);
    test_logical_partial_derivatives_1d<two_vars<1>, one_var<1>>(extents_1d);
    for (size_t n1 = 2; n1 <= Basis::lgl::maximum_number_of_pts / 2; ++n1) {
      const Index<2> extents_2d(n0, n1);
      test_logical_partial_derivatives_2d<two_vars<2>>(extents_2d);
      test_logical_partial_derivatives_2d<two_vars<2>, one_var<2>>(extents_2d);
      for (size_t n2 = 2; n2 <= Basis::lgl::maximum_number_of_pts / 2; ++n2) {
        const Index<3> extents_3d(n0, n1, n2);
        test_logical_partial_derivatives_3d<two_vars<3>>(extents_3d);
        test_logical_partial_derivatives_3d<two_vars<3>, one_var<3>>(
            extents_3d);
      }
    }
  }
}

SPECTRE_TEST_CASE("Unit.Numerical.LinearOperators.PartialDerivs",
                  "[NumericalAlgorithms][LinearOperators][Unit]") {
  const size_t n0 = Basis::lgl::maximum_number_of_pts / 2;
  const size_t n1 = Basis::lgl::maximum_number_of_pts / 2 + 1;
  const size_t n2 = Basis::lgl::maximum_number_of_pts / 2 - 1;
  const Index<1> extents_1d(n0);
  test_partial_derivatives_1d<two_vars<1>>(extents_1d);
  test_partial_derivatives_1d<two_vars<1>, one_var<1>>(extents_1d);
  const Index<2> extents_2d(n0, n1);
  test_partial_derivatives_2d<two_vars<2>>(extents_2d);
  test_partial_derivatives_2d<two_vars<2>, one_var<2>>(extents_2d);
  const Index<3> extents_3d(n0, n1, n2);
  test_partial_derivatives_3d<two_vars<3>>(extents_3d);
  test_partial_derivatives_3d<two_vars<3>, one_var<3>>(extents_3d);
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
  const Index<Dim> extents{extents_array};
  Variables<vars_tags> u(extents.product());
  Variables<db::wrap_tags_in<Tags::deriv, vars_tags, tmpl::size_t<Dim>,
                             Frame::Logical>>
      expected_du(extents.product());
  const auto x = logical_coordinates(extents);

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
      db::AddTags<Tags::Extents<Dim>, Tags::Variables<vars_tags>>,
      db::AddComputeItemsTags<Tags::LogicalCoordinates<Dim>, deriv_tag>>(
      extents, u);

  const auto& du = db::get<deriv_tag>(box);

  tmpl::for_each<vars_tags>([&du, &expected_du, &extents ](auto tag) noexcept {
    using Tag = tmpl::type_from<decltype(tag)>;
    using DerivativeTag = Tags::deriv<Tag, tmpl::size_t<Dim>, Frame::Logical>;
    auto& expected_dvariable = get<DerivativeTag>(expected_du);
    for (auto it = expected_dvariable.begin(); it != expected_dvariable.end();
         ++it) {
      const auto deriv_indices = expected_dvariable.get_tensor_index(it);
      const size_t deriv_index = deriv_indices[0];
      const auto tensor_indices =
          all_but_specified_element_of<0>(deriv_indices);
      for (size_t n = 0; n < extents.product(); ++n) {
        CAPTURE_PRECISE(get<Tag>(du[deriv_index]).get(tensor_indices)[n] -
                        (*it)[n]);
        CHECK(get<Tag>(du[deriv_index]).get(tensor_indices)[n] ==
              approx((*it)[n]));
      }
    }
  });
}

template <class MapType>
struct MapTag : db::DataBoxTag {
  using type = MapType;
  static constexpr db::DataBoxString label = "MapTag";
};

template <size_t Dim, typename T>
void test_partial_derivatives_compute_item(
    const std::array<size_t, Dim> extents_array, const T& map) noexcept {
  using vars_tags = tmpl::list<Var1<Dim>, Var2>;
  using map_tag = MapTag<std::decay_t<decltype(map)>>;
  using inv_jac_tag =
      Tags::InverseJacobian<map_tag, Tags::LogicalCoordinates<Dim>>;
  using deriv_tag = Tags::deriv<vars_tags, vars_tags, inv_jac_tag>;

  const std::array<size_t, Dim> array_to_functions{extents_array -
                                                   make_array<Dim>(size_t{1})};
  const Index<Dim> extents{extents_array};
  Variables<vars_tags> u(extents.product());
  Variables<
      db::wrap_tags_in<Tags::deriv, vars_tags, tmpl::size_t<Dim>, Frame::Grid>>
      expected_du(extents.product());
  const auto x_logical = logical_coordinates(extents);
  const auto x = map(logical_coordinates(extents));

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

  auto box = db::create<
      db::AddTags<Tags::Extents<Dim>, Tags::Variables<vars_tags>, map_tag>,
      db::AddComputeItemsTags<Tags::LogicalCoordinates<Dim>, inv_jac_tag,
                              deriv_tag>>(extents, u, map);

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
  using Affine = CoordinateMaps::Affine;
  using Affine2d = CoordinateMaps::ProductOf2Maps<Affine, Affine>;
  using Affine3d = CoordinateMaps::ProductOf3Maps<Affine, Affine, Affine>;

  Index<3> max_extents{10, 10, 5};

  for (size_t a = 1; a < max_extents[0]; ++a) {
    test_partial_derivatives_compute_item(
        std::array<size_t, 1>{{a + 1}},
        make_coordinate_map<Frame::Logical, Frame::Grid>(
            CoordinateMaps::Affine{-1.0, 1.0, -0.3, 0.7}));
    for (size_t b = 1; b < max_extents[1]; ++b) {
      test_partial_derivatives_compute_item(
          std::array<size_t, 2>{{a + 1, b + 1}},
          make_coordinate_map<Frame::Logical, Frame::Grid>(Affine2d{
              Affine{-1.0, 1.0, -0.3, 0.7}, Affine{-1.0, 1.0, 0.3, 0.55}}));
      for (size_t c = 1; a < max_extents[0] / 2 and b < max_extents[1] / 2 and
                         c < max_extents[2];
           ++c) {
        test_partial_derivatives_compute_item(
            std::array<size_t, 3>{{a + 1, b + 1, c + 1}},
            make_coordinate_map<Frame::Logical, Frame::Grid>(Affine3d{
                Affine{-1.0, 1.0, -0.3, 0.7}, Affine{-1.0, 1.0, 0.3, 0.55},
                Affine{-1.0, 1.0, 2.3, 2.8}}));
      }
    }
  }
}
