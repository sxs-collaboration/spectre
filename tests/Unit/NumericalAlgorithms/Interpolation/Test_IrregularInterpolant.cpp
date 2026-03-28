// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <array>
#include <cstddef>
#include <limits>
#include <memory>
#include <numbers>
#include <optional>
#include <pup.h>
#include <random>
#include <string>

#include "DataStructures/ComplexDataVector.hpp"
#include "DataStructures/DataBox/Tag.hpp"
#include "DataStructures/DataVector.hpp"
#include "DataStructures/IndexIterator.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "DataStructures/Variables.hpp"
#include "DataStructures/VariablesTag.hpp"
#include "Domain/Block.hpp"
#include "Domain/CoordinateMaps/Affine.hpp"
#include "Domain/CoordinateMaps/CoordinateMap.hpp"
#include "Domain/CoordinateMaps/CoordinateMap.tpp"
#include "Domain/CoordinateMaps/ProductMaps.hpp"
#include "Domain/CoordinateMaps/ProductMaps.tpp"
#include "Domain/Creators/Rectilinear.hpp"
#include "Domain/Domain.hpp"
#include "Domain/ElementMap.hpp"
#include "Domain/Structure/ElementId.hpp"
#include "Framework/TestHelpers.hpp"
#include "Helpers/DataStructures/MakeWithRandomValues.hpp"
#include "Helpers/NumericalAlgorithms/Spectral/FourierTestFunctions.hpp"
#include "Helpers/NumericalAlgorithms/SphericalHarmonics/YlmTestFunctions.hpp"
#include "NumericalAlgorithms/Interpolation/IrregularInterpolant.hpp"
#include "NumericalAlgorithms/Spectral/Basis.hpp"
#include "NumericalAlgorithms/Spectral/LogicalCoordinates.hpp"
#include "NumericalAlgorithms/Spectral/Mesh.hpp"
#include "NumericalAlgorithms/Spectral/Quadrature.hpp"
#include "PointwiseFunctions/AnalyticSolutions/RelativisticEuler/TovStar.hpp"
#include "PointwiseFunctions/Hydro/Tags.hpp"
#include "PointwiseFunctions/MathFunctions/MathFunction.hpp"
#include "PointwiseFunctions/MathFunctions/PowX.hpp"
#include "PointwiseFunctions/MathFunctions/TensorProduct.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/MakeArray.hpp"
#include "Utilities/MakeWithValue.hpp"
#include "Utilities/TMPL.hpp"
#include "Utilities/TaggedTuple.hpp"

namespace {
// Polynomial of given degree with leading coefficinet \f$a_0\f$, and with
// \f$a_{n+1} = a_n / falloff
class Polynomial {
 public:
  Polynomial(const size_t degree, const double a_0, const double falloff)
      : coefficients_(degree + 1) {
    double n = falloff * a_0;
    std::generate(coefficients_.begin(), coefficients_.end(), [&falloff, &n]() {
      n /= falloff;
      return n;
    });
  }
  Polynomial() = default;
  DataVector operator()(const DataVector& x) const {
    return evaluate_polynomial(coefficients_, x);
  }

 private:
  std::vector<double> coefficients_;
};

using Affine = domain::CoordinateMaps::Affine;
using Affine2D = domain::CoordinateMaps::ProductOf2Maps<Affine, Affine>;
using Affine3D = domain::CoordinateMaps::ProductOf3Maps<Affine, Affine, Affine>;

const double inertial_coord_min = -0.3;
const double inertial_coord_max = 0.7;

template <size_t VolumeDim>
auto make_affine_map();

template <>
auto make_affine_map<1>() {
  return domain::make_coordinate_map<Frame::ElementLogical, Frame::Inertial>(
      Affine{-1.0, 1.0, inertial_coord_min, inertial_coord_max});
}

template <>
auto make_affine_map<2>() {
  return domain::make_coordinate_map<Frame::ElementLogical, Frame::Inertial>(
      Affine2D{Affine{-1.0, 1.0, inertial_coord_min, inertial_coord_max},
               Affine{-1.0, 1.0, inertial_coord_min, inertial_coord_max}});
}

template <>
auto make_affine_map<3>() {
  return domain::make_coordinate_map<Frame::ElementLogical, Frame::Inertial>(
      Affine3D{Affine{-1.0, 1.0, inertial_coord_min, inertial_coord_max},
               Affine{-1.0, 1.0, inertial_coord_min, inertial_coord_max},
               Affine{-1.0, 1.0, inertial_coord_min, inertial_coord_max}});
}

namespace TestTags {

template <typename DataType, size_t Dim>
struct Vector : db::SimpleTag {
  using type = tnsr::I<DataType, Dim>;
  static auto fill_values(const MathFunctions::TensorProduct<Dim>& f,
                          const tnsr::I<DataVector, Dim>& x) {
    auto result = make_with_value<tnsr::I<DataType, Dim>>(x, 0.);
    const auto f_of_x = f(x);
    for (size_t d = 0; d < Dim; ++d) {
      result.get(d) = (d + 0.5) * get(f_of_x);
    }
    return result;
  }
};

template <typename DataType, size_t Dim>
struct SymmetricTensor : db::SimpleTag {
  using type = tnsr::ii<DataType, Dim>;
  static auto fill_values(const MathFunctions::TensorProduct<Dim>& f,
                          const tnsr::I<DataVector, Dim>& x) {
    auto result = make_with_value<tnsr::ii<DataType, Dim>>(x, 0.);
    const auto f_of_x = f(x);
    for (size_t i = 0; i < Dim; ++i) {
      for (size_t j = i; j < Dim; ++j) {  // Symmetry
        result.get(i, j) = (static_cast<double>(i + j) + 0.33) * get(f_of_x);
      }
    }
    return result;
  }
};

}  // namespace TestTags

template <size_t Dim>
void test_interpolate_to_points(const Mesh<Dim>& mesh) {
  // Fill target interpolation coordinates with random values
  MAKE_GENERATOR(generator);
  std::uniform_real_distribution<> dist(inertial_coord_min, inertial_coord_max);

  const auto nn_generator = make_not_null(&generator);
  const auto nn_dist = make_not_null(&dist);

  const size_t number_of_points = 6;
  const auto target_x_inertial =
      make_with_random_values<tnsr::I<DataVector, Dim>>(
          nn_generator, nn_dist, DataVector(number_of_points));

  const auto coordinate_map = make_affine_map<Dim>();
  const auto target_x = [&target_x_inertial, &coordinate_map,
                         &number_of_points]() {
    tnsr::I<DataVector, Dim, Frame::ElementLogical> result(number_of_points);
    for (size_t s = 0; s < number_of_points; ++s) {
      tnsr::I<double, Dim> x_inertial_local{};
      for (size_t d = 0; d < Dim; ++d) {
        x_inertial_local.get(d) = target_x_inertial.get(d)[s];
      }
      const auto x_local = coordinate_map.inverse(x_inertial_local).value();
      for (size_t d = 0; d < Dim; ++d) {
        result.get(d)[s] = x_local.get(d);
      }
    }
    return result;
  }();

  // Set up interpolator. Need do this only once.
  const intrp::Irregular<Dim> irregular_interpolant(mesh, target_x);
  test_serialization(irregular_interpolant);

  // ... but we construct another interpolator to test operator!=
  {
    auto target_x_new = target_x;
    target_x_new.get(0)[0] *= 0.98;  // Change one point slightly.
    const intrp::Irregular<Dim> irregular_interpolant_new(mesh, target_x_new);
    CHECK(irregular_interpolant_new != irregular_interpolant);
  }

  // ... and another to test the constructor from a single point
  {
    tnsr::I<double, Dim, Frame::ElementLogical> target_x_single{};
    tnsr::I<DataVector, Dim, Frame::ElementLogical> target_x_single_dv(1_st);
    for (size_t d = 0; d < Dim; ++d) {
      target_x_single.get(d) = target_x.get(d)[0];
      target_x_single_dv.get(d)[0] = target_x_single.get(d);
    }
    CHECK(intrp::Irregular<Dim>(mesh, target_x_single) ==
          intrp::Irregular<Dim>(mesh, target_x_single_dv));
  }

  // Coordinates on the grid
  const auto src_x = coordinate_map(logical_coordinates(mesh));

  // Set up variables
  using tags = tmpl::list<TestTags::Vector<DataVector, Dim>,
                          TestTags::SymmetricTensor<DataVector, Dim>>;
  Variables<tags> src_vars(mesh.number_of_grid_points());
  Variables<tags> expected_dest_vars(number_of_points);

  // We will make polynomials of the form x^a y^b z^c ...
  // for all a,b,c, that result in exact interpolation.
  // IndexIterator loops over "a,b,c"
  for (IndexIterator<Dim> iter(mesh.extents()); iter; ++iter) {
    // Set up analytic solution.  We fill a Variables with this solution,
    // interpolate to arbitrary points, and then check that the
    // values at arbitrary points match this solution.
    // We choose polynomials so that interpolation is exact on an LGL grid.
    std::array<std::unique_ptr<MathFunction<1, Frame::Inertial>>, Dim>
        functions;
    for (size_t d = 0; d < Dim; ++d) {
      gsl::at(functions, d) =
          std::make_unique<MathFunctions::PowX<1, Frame::Inertial>>(iter()[d]);
    }
    MathFunctions::TensorProduct<Dim> f(1.0, std::move(functions));

    // Fill source and expected destination Variables with analytic solution.
    tmpl::for_each<tags>([&f, &src_x, &target_x_inertial, &src_vars,
                          &expected_dest_vars](auto tag) {
      using Tag = tmpl::type_from<decltype(tag)>;
      get<Tag>(src_vars) = Tag::fill_values(f, src_x);
      get<Tag>(expected_dest_vars) = Tag::fill_values(f, target_x_inertial);
    });

    // Interpolate
    // (g++ 7.2.0 does not allow `const auto dest_vars` here)
    const Variables<tags> dest_vars =
        irregular_interpolant.interpolate(src_vars);

    CHECK_VARIABLES_APPROX(dest_vars, expected_dest_vars);

    const DataVector result_dv = irregular_interpolant.interpolate(
        get<0>(get<TestTags::Vector<DataVector, Dim>>(src_vars)));
    CHECK_ITERABLE_APPROX(
        result_dv,
        get<0>(get<TestTags::Vector<DataVector, Dim>>(expected_dest_vars)));

    {
      INFO("Complex data");
      // Copy the real data above into a complex Variables
      Variables<tmpl::list<TestTags::Vector<ComplexDataVector, Dim>>>
          src_vars_complex(mesh.number_of_grid_points());
      Variables<tmpl::list<TestTags::Vector<ComplexDataVector, Dim>>>
          expected_complex(number_of_points);
      const auto& src_vector = get<TestTags::Vector<DataVector, Dim>>(src_vars);
      const auto& expected_vector =
          get<TestTags::Vector<DataVector, Dim>>(expected_dest_vars);
      for (size_t d = 0; d < Dim; ++d) {
        for (size_t j = 0; j < src_vars.number_of_grid_points(); ++j) {
          get<TestTags::Vector<ComplexDataVector, Dim>>(src_vars_complex)
              .get(d)[j] = std::complex<double>(src_vector.get(d)[j],
                                                2. * src_vector.get(d)[j]);
        }
        for (size_t j = 0; j < number_of_points; ++j) {
          get<TestTags::Vector<ComplexDataVector, Dim>>(expected_complex)
              .get(d)[j] = std::complex<double>(expected_vector.get(d)[j],
                                                2. * expected_vector.get(d)[j]);
        }
      }
      // Interpolate the complex data
      const auto result_complex =
          irregular_interpolant.interpolate(src_vars_complex);
      CHECK_VARIABLES_APPROX(result_complex, expected_complex);
      const auto result_cdv = irregular_interpolant.interpolate(get<0>(
          get<TestTags::Vector<ComplexDataVector, Dim>>(src_vars_complex)));
      CHECK_ITERABLE_APPROX(
          result_cdv, get<0>(get<TestTags::Vector<ComplexDataVector, Dim>>(
                          expected_complex)));
      std::vector<std::complex<double>> result_data(expected_complex.size());
      auto result_span = gsl::make_span(result_data);
      irregular_interpolant.interpolate(
          make_not_null(&result_span),
          gsl::make_span(src_vars_complex.data(), src_vars_complex.size()));
    }
    {
      INFO("Single precision data");
      // Copy the data above into single precision
      std::vector<float> src_vars_single(src_vars.size());
      std::vector<float> expected_dest_single(expected_dest_vars.size());
      std::copy_n(src_vars.data(), src_vars.size(), src_vars_single.data());
      std::copy_n(expected_dest_vars.data(), expected_dest_vars.size(),
                  expected_dest_single.data());
      // Interpolate the single precision data
      std::vector<float> result_single(expected_dest_single.size());
      auto result_single_span = gsl::make_span(result_single);
      irregular_interpolant.interpolate(make_not_null(&result_single_span),
                                        gsl::make_span(src_vars_single));
      const Approx custom_approx =
          Approx::custom()
              .epsilon(std::numeric_limits<float>::epsilon() * 10.)
              .scale(1.);
      CHECK_ITERABLE_CUSTOM_APPROX(result_single, expected_dest_single,
                                   custom_approx);
    }
  }
}

template <Spectral::Basis Basis, Spectral::Quadrature Quadrature>
void test_irregular_interpolant() {
  const size_t start_points = 4;
  const size_t end_points = 6;
  for (size_t n0 = start_points; n0 < end_points; ++n0) {
    test_interpolate_to_points<1>(Mesh<1>{n0, Basis, Quadrature});
    for (size_t n1 = start_points; n1 < end_points; ++n1) {
      test_interpolate_to_points<2>(Mesh<2>{{{n0, n1}}, Basis, Quadrature});
      for (size_t n2 = start_points; n2 < end_points; ++n2) {
        test_interpolate_to_points<3>(
            Mesh<3>{{{n0, n1, n2}}, Basis, Quadrature});
      }
    }
  }
}

void test_irregular_interpolant_mixed_quadrature() {
  const size_t start_points = 4;
  const size_t end_points = 6;
  for (size_t n0 = start_points; n0 < end_points; ++n0) {
    for (size_t n1 = start_points; n1 < end_points; ++n1) {
      test_interpolate_to_points<2>(Mesh<2>{
          {{n0, n1}},
          {{Spectral::Basis::Legendre, Spectral::Basis::Legendre}},
          {{Spectral::Quadrature::GaussLobatto, Spectral::Quadrature::Gauss}}});
      for (size_t n2 = start_points; n2 < end_points; ++n2) {
        test_interpolate_to_points<3>(Mesh<3>{
            {{n0, n1, n2}},
            {{Spectral::Basis::Legendre, Spectral::Basis::Legendre,
              Spectral::Basis::Legendre}},
            {{Spectral::Quadrature::GaussLobatto, Spectral::Quadrature::Gauss,
              Spectral::Quadrature::GaussLobatto}}});
      }
    }
  }
}

template <size_t Dim>
Domain<Dim> create_domain(double length,
                          const std::array<size_t, Dim>& extents);

template <>
Domain<3> create_domain<3>(const double length,
                           const std::array<size_t, 3>& extents) {
  const domain::creators::Brick creator{{{0.0, 0.0, 0.0}},
                                        {{length, length, length}},
                                        {{0, 0, 0}},
                                        extents,
                                        {{false, false, false}}};
  return creator.domain();
}

template <>
Domain<2> create_domain<2>(const double length,
                           const std::array<size_t, 2>& extents) {
  const domain::creators::Rectangle creator{
      {{0.0, 0.0}}, {{length, length}}, {{0, 0}}, extents, {{false, false}}};
  return creator.domain();
}

template <>
Domain<1> create_domain<1>(const double length,
                           const std::array<size_t, 1>& extents) {
  const domain::creators::Interval creator{{{0.0}}, {{length}}, {{0}}, extents};
  return creator.domain();
}

template <size_t Dim>
tnsr::I<DataVector, Dim, Frame::ElementLogical> create_target_points(
    size_t n_random_target_points) {
  tnsr::I<DataVector, Dim, Frame::ElementLogical> xi_target{
      n_random_target_points + 2, -1.0};
  for (size_t d = 0; d < Dim; ++d) {
    xi_target.get(d)[1] = 1.0;
  }
  return xi_target;
}

namespace Tags {
struct Scalar : ::db::SimpleTag {
  using type = ::Scalar<DataVector>;
};
}  // namespace Tags

using var_tags = tmpl::list<Tags::Scalar>;

template <size_t Dim>
Variables<var_tags> polynomial(
    const tnsr::I<DataVector, Dim, Frame::Inertial>& x, const size_t degree) {
  tnsr::I<DataVector, Dim, Frame::Inertial> v = x;
  for (size_t d = 0; d < Dim; ++d) {
    for (size_t n = degree; n > 1; --n) {
      v.get(d) *= (x.get(d) + 1.0);
    }
  }
  Variables<var_tags> result{get<0>(x).size()};
  get(get<Tags::Scalar>(result)) = get<0>(v);
  for (size_t d = 1; d < Dim; ++d) {
    get(get<Tags::Scalar>(result)) *= v.get(d);
  }
  return result;
}

template <size_t Dim>
void test_polynomial_interpolant(const std::array<size_t, Dim>& extents,
                                 const size_t max_degree) {
  const size_t n_random_target_points = 10;
  // use a small domain to avoid huge polynomial values for large max_degree
  // which results in large absolute errors
  const auto domain = create_domain<Dim>(1.3, extents);
  const auto& block = domain.blocks()[0];
  const ElementMap<Dim, Frame::Inertial> element_map{
      ElementId<Dim>{0}, block.stationary_map().get_clone()};
  Mesh<Dim> mesh(extents, Spectral::Basis::FiniteDifference,
                 Spectral::Quadrature::CellCentered);

  const auto source_xi = logical_coordinates(mesh);
  const auto source_x = element_map(source_xi);
  const auto target_xi = create_target_points<Dim>(n_random_target_points);
  const auto target_x = element_map(target_xi);
  const intrp::Irregular irregular_interp{mesh, target_xi,
                                    std::optional<size_t>{max_degree}};

  for (size_t degree = 0; degree <= max_degree; ++degree) {
    const auto source_vars = polynomial<Dim>(source_x, degree);
    const auto target_vars = irregular_interp.interpolate(source_vars);
    const auto expected_vars = polynomial<Dim>(target_x, degree);
    CAPTURE(Dim);
    CAPTURE(max_degree);
    CAPTURE(degree);
    CHECK_VARIABLES_APPROX(target_vars, expected_vars);
  }
}

void test_tov(const size_t max_degree, const bool specified_interp_order) {
  const std::array<size_t, 3> isotropic_extents{{15, 15, 15}};
  constexpr size_t n_resolutions = 4;
  auto errors =
      make_array<n_resolutions>(std::numeric_limits<double>::signaling_NaN());
  const double central_density = 1.28e-3;
  for (size_t i = 0; i < n_resolutions; ++i) {
    const Domain<3> domain = create_domain<3>(
        6.6666666666666666666 / two_to_the(i), isotropic_extents);
    const Block<3>& cube = domain.blocks()[0];
    Mesh<3> mesh(isotropic_extents, Spectral::Basis::FiniteDifference,
                 Spectral::Quadrature::CellCentered);
    const auto xi = logical_coordinates(mesh);
    const ElementMap<3, Frame::Inertial> element_map{
        ElementId<3>{0}, cube.stationary_map().get_clone()};
    const auto x = element_map(xi);

    RelativisticEuler::Solutions::TovStar tov_star(
        central_density,
        std::make_unique<EquationsOfState::PolytropicFluid<true>>(100.0, 2.0));

    using rho_tag = hydro::Tags::RestMassDensity<DataVector>;
    auto vars = variables_from_tagged_tuple(
        tov_star.variables(x, 0.0, tmpl::list<rho_tag>{}));

    const tnsr::I<DataVector, 3, Frame::ElementLogical> xi_target{1_st, -1.0};
    intrp::Irregular irregular_interp{mesh, xi_target};
    if (specified_interp_order) {
      irregular_interp =
          intrp::Irregular{mesh, xi_target, std::optional<size_t>{max_degree}};
    }

    const auto target_vars = irregular_interp.interpolate(vars);
    gsl::at(errors, i) =
        fabs(central_density - get(get<rho_tag>(target_vars))[0]);
  }

  std::reverse(std::begin(errors), std::end(errors));
  auto ratio_of_errors =
      make_array<n_resolutions>(std::numeric_limits<double>::signaling_NaN());
  std::adjacent_difference(std::begin(errors), std::end(errors),
                           std::begin(ratio_of_errors), std::divides<>{});

  Approx custom_approx = Approx::custom().epsilon(1.e-2).scale(1.);
  for (size_t i = 1; i < n_resolutions; ++i) {
    CAPTURE(max_degree);
    CAPTURE(i);
    // since \rho is a symmetric function across the center, quadratic
    // extrapolation has one order higher convergence rate at the center
    CHECK((specified_interp_order
               ? (max_degree == 2 ? 16.0 : two_to_the(max_degree + 1))
               : 4.0) == custom_approx(gsl::at(ratio_of_errors, i)));
  }
}

void test_2d_spherical(const gsl::not_null<std::mt19937*> generator) {
  std::uniform_real_distribution<> xi_distribution(-1.0, 1.0);
  std::uniform_real_distribution<> phi_distribution(0.0, 2.0 * M_PI);
  for (size_t n_target_points = 1; n_target_points < 13;
       n_target_points += 11) {
    tnsr::I<DataVector, 2, Frame::ElementLogical> xi_target{n_target_points};
    get<0>(xi_target) = acos(make_with_random_values<DataVector>(
        generator, make_not_null(&xi_distribution), xi_target));
    get<1>(xi_target) = make_with_random_values<DataVector>(
        generator, make_not_null(&phi_distribution), xi_target);
    for (size_t n_z = 0; n_z < 4; ++n_z) {
      for (size_t n_y = 0; n_y < 4; ++n_y) {
        for (size_t n_x = 0; n_x < 4; ++n_x) {
          const Mesh<2> source_mesh{
              std::array{n_x + n_y + n_z + 2, 2 * (n_x + n_y) + 3},
              std::array{Spectral::Basis::SphericalHarmonic,
                         Spectral::Basis::SphericalHarmonic},
              std::array{Spectral::Quadrature::Gauss,
                         Spectral::Quadrature::Equiangular}};
          const YlmTestFunctions::ProductOfPolynomials f(n_x, n_y, n_z);
          const auto xi_source = logical_coordinates(source_mesh);
          const DataVector f_source = f(xi_source);
          const DataVector f_expected = f(xi_target);
          const intrp::Irregular<2> interpolator(source_mesh, xi_target);
          const DataVector f_interpolated = interpolator.interpolate(f_source);
          CHECK_ITERABLE_APPROX(f_interpolated, f_expected);
        }
      }
    }
  }
}

void test_3d_spherical(const gsl::not_null<std::mt19937*> generator) {
  std::uniform_real_distribution<> xi_distribution(-1.0, 1.0);
  std::uniform_real_distribution<> phi_distribution(0.0, 2.0 * M_PI);
  for (size_t n_target_points = 1; n_target_points < 13;
       n_target_points += 11) {
    tnsr::I<DataVector, 3, Frame::ElementLogical> xi_target{n_target_points};
    get<0>(xi_target) = make_with_random_values<DataVector>(
        generator, make_not_null(&xi_distribution), xi_target);
    get<1>(xi_target) = acos(make_with_random_values<DataVector>(
        generator, make_not_null(&xi_distribution), xi_target));
    get<2>(xi_target) = make_with_random_values<DataVector>(
        generator, make_not_null(&phi_distribution), xi_target);
    for (size_t n_r = 2; n_r < 4; ++n_r) {
      for (size_t n_z = 0; n_z < 4; ++n_z) {
        for (size_t n_y = 0; n_y < 4; ++n_y) {
          for (size_t n_x = 0; n_x < 4; ++n_x) {
            const Mesh<3> source_mesh{
                std::array{n_r, n_x + n_y + n_z + 2, 2 * (n_x + n_y) + 3},
                std::array{Spectral::Basis::Legendre,
                           Spectral::Basis::SphericalHarmonic,
                           Spectral::Basis::SphericalHarmonic},
                std::array{Spectral::Quadrature::GaussLobatto,
                           Spectral::Quadrature::Gauss,
                           Spectral::Quadrature::Equiangular}};
            const Polynomial f_r{n_r - 1, 1.5, 2.0};
            const YlmTestFunctions::ProductOfPolynomials f_a(n_x, n_y, n_z);
            const auto xi_source = logical_coordinates(source_mesh);
            const DataVector f_source =
                f_r(get<0>(xi_source)) *
                f_a(get<1>(xi_source), get<2>(xi_source));
            const DataVector f_expected =
                f_r(get<0>(xi_target)) *
                f_a(get<1>(xi_target), get<2>(xi_target));
            const intrp::Irregular<3> interpolator(source_mesh, xi_target);
            const DataVector f_interpolated =
                interpolator.interpolate(f_source);
            CHECK_ITERABLE_APPROX(f_interpolated, f_expected);
          }
        }
      }
    }
  }
}

void test_2d_hollow_disk(const gsl::not_null<std::mt19937*> generator) {
  std::uniform_real_distribution<> xi_distribution(-1.0, 1.0);
  std::uniform_real_distribution<> phi_distribution(0.0,
                                                    2.0 * std::numbers::pi);
  const Approx custom_approx = Approx::custom().epsilon(1e-12).scale(1.);
  for (size_t n_target_points = 1; n_target_points < 13;
       n_target_points += 11) {
    tnsr::I<DataVector, 2, Frame::ElementLogical> xi_target{n_target_points};
    get<0>(xi_target) = make_with_random_values<DataVector>(
        generator, make_not_null(&xi_distribution), xi_target);
    get<1>(xi_target) = make_with_random_values<DataVector>(
        generator, make_not_null(&phi_distribution), xi_target);
    for (size_t n_r = 2; n_r < 4; ++n_r) {
      CAPTURE(n_r);
      for (size_t n_y = 0; n_y < 4; ++n_y) {
        CAPTURE(n_y);
        for (size_t n_x = 0; n_x < 4; ++n_x) {
          CAPTURE(n_x);
          const Mesh<2> source_mesh{
              std::array{n_r, 2 * (n_x + n_y) + 1},
              std::array{Spectral::Basis::Legendre, Spectral::Basis::Fourier},
              std::array{Spectral::Quadrature::GaussLobatto,
                         Spectral::Quadrature::Equiangular}};
          const Polynomial f_r{n_r - 1, 1.5, 2.0};
          const FourierTestFunctions::ProductOfPolynomials f_m{n_x, n_y};
          const auto xi_source = logical_coordinates(source_mesh);
          const DataVector f_source =
              f_r(get<0>(xi_source)) * f_m(get<1>(xi_source));
          const DataVector f_expected =
              f_r(get<0>(xi_target)) * f_m(get<1>(xi_target));
          const intrp::Irregular<2> interpolator(source_mesh, xi_target);
          const DataVector f_interpolated = interpolator.interpolate(f_source);
          CHECK_ITERABLE_CUSTOM_APPROX(f_interpolated, f_expected,
                                       custom_approx);
        }
      }
    }
  }
}

void test_3d_hollow_cylinder(const gsl::not_null<std::mt19937*> generator) {
  std::uniform_real_distribution<> xi_distribution(-1.0, 1.0);
  std::uniform_real_distribution<> phi_distribution(0.0,
                                                    2.0 * std::numbers::pi);
  const Approx custom_approx = Approx::custom().epsilon(1e-12).scale(1.);
  for (size_t n_target_points = 1; n_target_points < 13;
       n_target_points += 11) {
    tnsr::I<DataVector, 3, Frame::ElementLogical> xi_target{n_target_points};
    get<0>(xi_target) = make_with_random_values<DataVector>(
        generator, make_not_null(&xi_distribution), xi_target);
    get<1>(xi_target) = make_with_random_values<DataVector>(
        generator, make_not_null(&phi_distribution), xi_target);
    get<2>(xi_target) = make_with_random_values<DataVector>(
        generator, make_not_null(&xi_distribution), xi_target);
    for (size_t n_r = 2; n_r < 4; ++n_r) {
      CAPTURE(n_r);
      for (size_t n_y = 0; n_y < 4; ++n_y) {
        CAPTURE(n_y);
        for (size_t n_x = 0; n_x < 4; ++n_x) {
          CAPTURE(n_x);
          for (size_t n_z = 2; n_z < 4; ++n_z) {
            CAPTURE(n_z);
            const Mesh<3> source_mesh{
                std::array{n_r, 2 * (n_x + n_y) + 1, n_z},
                std::array{Spectral::Basis::Legendre, Spectral::Basis::Fourier,
                           Spectral::Basis::Legendre},
                std::array{Spectral::Quadrature::GaussLobatto,
                           Spectral::Quadrature::Equiangular,
                           Spectral::Quadrature::GaussLobatto}};
            const Polynomial f_r{n_r - 1, 1.5, 2.0};
            const Polynomial f_z{n_z - 1, 1.6, 2.0};
            const FourierTestFunctions::ProductOfPolynomials f_m{n_x, n_y};
            const auto xi_source = logical_coordinates(source_mesh);
            const DataVector f_source = f_r(get<0>(xi_source)) *
                                        f_m(get<1>(xi_source)) *
                                        f_z(get<2>(xi_source));
            const DataVector f_expected = f_r(get<0>(xi_target)) *
                                          f_m(get<1>(xi_target)) *
                                          f_z(get<2>(xi_target));
            const intrp::Irregular<3> interpolator(source_mesh, xi_target);
            const DataVector f_interpolated =
                interpolator.interpolate(f_source);
            CHECK_ITERABLE_CUSTOM_APPROX(f_interpolated, f_expected,
                                         custom_approx);
          }
        }
      }
    }
  }
}

void test_cartoon_spherical(const gsl::not_null<std::mt19937*> generator) {
  std::uniform_real_distribution<> xi_distribution(-1.0, 1.0);

  const size_t number_of_points = 6;
  tnsr::I<DataVector, 3, Frame::ElementLogical> xi_target{number_of_points};
  get<0>(xi_target) = make_with_random_values<DataVector>(
      generator, make_not_null(&xi_distribution), xi_target);
  get<1>(xi_target) = DataVector(number_of_points, 0.0);
  get<2>(xi_target) = DataVector(number_of_points, 0.0);
  for (size_t n_x = 4; n_x < 6; ++n_x) {
    const Mesh<3> source_mesh{
        {{n_x, 1, 1}},
        {{Spectral::Basis::Legendre, Spectral::Basis::Cartoon,
          Spectral::Basis::Cartoon}},
        {{Spectral::Quadrature::GaussLobatto,
          Spectral::Quadrature::SphericalSymmetry,
          Spectral::Quadrature::SphericalSymmetry}}};
    const Polynomial f_r{n_x - 1, 1.5, 2.0};
    const auto xi_source = logical_coordinates(source_mesh);
    const DataVector f_source = f_r(get<0>(xi_source));
    const DataVector f_expected = f_r(get<0>(xi_target));
    const intrp::Irregular<3> interpolator(source_mesh, xi_target);
    const DataVector f_interpolated = interpolator.interpolate(f_source);
    CHECK_ITERABLE_APPROX(f_interpolated, f_expected);
  }
}

void test_cartoon_axial(const gsl::not_null<std::mt19937*> generator) {
  std::uniform_real_distribution<> xi_distribution(-1.0, 1.0);

  const size_t number_of_points = 6;
  tnsr::I<DataVector, 3, Frame::ElementLogical> xi_target{number_of_points};
  get<0>(xi_target) = make_with_random_values<DataVector>(
      generator, make_not_null(&xi_distribution), xi_target);
  get<1>(xi_target) = make_with_random_values<DataVector>(
      generator, make_not_null(&xi_distribution), xi_target);
  get<2>(xi_target) = DataVector(number_of_points, 0.0);
  for (size_t n_x = 4; n_x < 6; ++n_x) {
    for (size_t n_y = 4; n_y < 6; ++n_y) {
      const Mesh<3> source_mesh{
          {{n_x, n_y, 1}},
          {{Spectral::Basis::Legendre, Spectral::Basis::Legendre,
            Spectral::Basis::Cartoon}},
          {{Spectral::Quadrature::GaussLobatto,
            Spectral::Quadrature::GaussLobatto,
            Spectral::Quadrature::AxialSymmetry}}};
      const Polynomial f_x{n_x - 1, 1.5, 2.0};
      const Polynomial f_y{n_y - 1, 2.5, 1.5};
      const auto xi_source = logical_coordinates(source_mesh);
      const DataVector f_source =
          f_x(get<0>(xi_source)) * f_y(get<1>(xi_source));
      const DataVector f_expected =
          f_x(get<0>(xi_target)) * f_y(get<1>(xi_target));
      const intrp::Irregular<3> interpolator(source_mesh, xi_target);
      const DataVector f_interpolated = interpolator.interpolate(f_source);
      CHECK_ITERABLE_APPROX(f_interpolated, f_expected);
    }
  }
}
}  // namespace

// [[TimeOut, 10]]
SPECTRE_TEST_CASE("Unit.Numerical.Interpolation.IrregularInterpolant",
                  "[Unit][NumericalAlgorithms]") {
  test_irregular_interpolant<Spectral::Basis::Legendre,
                             Spectral::Quadrature::GaussLobatto>();
  test_irregular_interpolant<Spectral::Basis::Legendre,
                             Spectral::Quadrature::Gauss>();
  test_irregular_interpolant_mixed_quadrature();
  for (size_t max_degree = 1; max_degree <= 3; ++max_degree) {
    test_polynomial_interpolant<1>({{11}}, max_degree);
    test_polynomial_interpolant<2>({{11, 11}}, max_degree);
    test_polynomial_interpolant<2>({{11, 9}}, max_degree);
    test_polynomial_interpolant<3>({{11, 11, 11}}, max_degree);
    test_polynomial_interpolant<3>({{11, 9, 11}}, max_degree);
    test_polynomial_interpolant<3>({{11, 11, 9}}, max_degree);
    test_polynomial_interpolant<3>({{11, 9, 9}}, max_degree);
    test_polynomial_interpolant<3>({{11, 9, 13}}, max_degree);
    for (const bool specified_interp_order : {false, true}) {
      test_tov(max_degree, specified_interp_order);
    }
  }
  MAKE_GENERATOR(generator);
  test_2d_spherical(make_not_null(&generator));
  test_3d_spherical(make_not_null(&generator));
  test_2d_hollow_disk(make_not_null(&generator));
  test_3d_hollow_cylinder(make_not_null(&generator));
  test_cartoon_spherical(make_not_null(&generator));
  test_cartoon_axial(make_not_null(&generator));
}
