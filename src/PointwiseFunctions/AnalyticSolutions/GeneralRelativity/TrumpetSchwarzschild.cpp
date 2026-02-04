// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "PointwiseFunctions/AnalyticSolutions/GeneralRelativity/TrumpetSchwarzschild.hpp"

#include <array>
#include <boost/math/quadrature/tanh_sinh.hpp>
#include <cmath>
#include <cstddef>
#include <pup.h>
#include <vector>

#include "DataStructures/DataBox/Prefixes.hpp"
#include "DataStructures/DataVector.hpp"
#include "DataStructures/Matrix.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "DataStructures/Tensor/TypeAliases.hpp"
#include "DataStructures/VectorImpl.hpp"
#include "Domain/CoordinateMaps/Affine.hpp"
#include "Domain/CoordinateMaps/CoordinateMap.hpp"
#include "Domain/CoordinateMaps/CoordinateMap.tpp"
#include "NumericalAlgorithms/Interpolation/IrregularInterpolant.hpp"
#include "NumericalAlgorithms/LinearOperators/IndefiniteIntegral.hpp"
#include "NumericalAlgorithms/RootFinding/TOMS748.hpp"
#include "NumericalAlgorithms/Spectral/Basis.hpp"
#include "NumericalAlgorithms/Spectral/LogicalCoordinates.hpp"
#include "NumericalAlgorithms/Spectral/Mesh.hpp"
#include "NumericalAlgorithms/Spectral/Quadrature.hpp"
#include "Options/ParseError.hpp"
#include "PointwiseFunctions/GeneralRelativity/Tags.hpp"
#include "Utilities/Algorithm.hpp"
#include "Utilities/ErrorHandling/Assert.hpp"
#include "Utilities/Functional.hpp"
#include "Utilities/GenerateInstantiations.hpp"
#include "Utilities/MakeWithValue.hpp"
#include "Utilities/TMPL.hpp"

namespace {
// All quantities are dimensionless, i.e. in unit of black hole mass M

constexpr size_t num_of_pts = 20;  // number of LGL pts per DG element
constexpr double lapse_threshold = 0.1;
// the max_isotropic_r works fine for larger value; one just need to add
// one or two more elements. Very close to the puncture (r<1.e-4)
// the error is relatively large, and one needs to add more elements.
constexpr double max_isotropic_r = 5000;  // in unit of the black hole mass M

// Element placement in isotropic coordinates for interpolation to user grid
// It is assumed that element_lower_bounds and element_upper_bounds are sorted
// in asending order, and element_lower_bounds[i] < element_upper_bounds[i]
constexpr std::array element_lower_bounds{
    0., 1.e-4, 1.e-3, 1.e-2, 1.e-1, 0.5,  1.,   2.,    4.,
    8., 16.,    32.,    64.,    128.,   256., 512., 1024., 2048.};
constexpr size_t num_of_elements = element_lower_bounds.size();
constexpr auto element_upper_bounds = [] {
  std::array<double, num_of_elements> upper_bounds{};
  for (size_t i = 0; i + 1 < num_of_elements; ++i) {
    upper_bounds.at(i) = element_lower_bounds.at(i + 1);
  }
  upper_bounds.at(num_of_elements - 1) = max_isotropic_r;
  return upper_bounds;
}();

const Mesh<1> source_mesh{num_of_pts, Spectral::Basis::Legendre,
                          Spectral::Quadrature::GaussLobatto};

using Affine = domain::CoordinateMaps::Affine;

// Get the critical lapse from eq. (42)
// for n=2 critical lapse ~0.16
double get_crit_lapse(const double n) {
  return sqrt((sqrt(4. + 9. * n * n) - 3. * n) /
              (sqrt(4. + 9. * n * n) + 3. * n));
}

// Get the critical Schwarzschild radius R_c from eq. (41)
double get_crit_schwarzschild_r(const double n) {
  return ((3. * n * n + sqrt(4. * pow<2>(n) + 9. * pow<4>(n))) / (4. * n * n));
}

// Get the value of C(n)^2 in eq. (43); note that eq. (43) is missing a factor
// of M^4
double get_c_n_squared(const double n) {
  return pow<3>(3. * n + sqrt(4. + 9. * n * n)) / (128. * pow<3>(n)) *
         exp(-2. * get_crit_lapse(n) / n);
}

// Get the function value of the transedental equation eq. (39)
double function_of_lapse_and_schwarzschild_r(const double lapse,
                                             const double schwarzschild_r,
                                             const double n) {
  ASSERT((lapse >= 0.) and (lapse < 1.),
         "invalid lapse range!"
         "Required: [0, 1) but given "
             << lapse);
  return (pow<4>(schwarzschild_r) - 2. * pow<3>(schwarzschild_r) +
          get_c_n_squared(n) * exp(2. * lapse / n) -
          lapse * lapse * pow<4>(schwarzschild_r));
}

// Get the Schwarzschild R corresponding to a lapse value by solving the
// transedental equation eq. (39)
double get_schwarzschild_r_from_lapse(const double n, const double lapse,
                                      const double crit_lapse,
                                      const double crit_schwarzschild_r) {
  ASSERT((lapse >= 0.) and (lapse < 1.),
         "invalid lapse range!"
         "Required: [0, 1) but given "
             << lapse);

  // if one really needs solution beyond max_isotropic_r, the following needs
  // to be modified.
  const double max_schwarzschild_r =
      std::max(max_isotropic_r, 4. / (1. - pow<2>(lapse)));
  // the min_schwarzschild_r is NOT the location of the black
  // hole throat. This is chosen purely for numerical root finding
  const double min_schwarzschild_r = 0.;

  return RootFinder::toms748(
      [n, lapse](double schwarzschild_r) -> double {
        return function_of_lapse_and_schwarzschild_r(lapse, schwarzschild_r, n);
      },
      (lapse < crit_lapse) ? min_schwarzschild_r : crit_schwarzschild_r,
      (lapse < crit_lapse) ? crit_schwarzschild_r : max_schwarzschild_r,
      1.e-15, 1.e-15);
}

// Get the derivative of the lapse with respect to the Schwarzschild R
// Note that at the critical Schwarzschild R, this quantity is undefined
// since both the numerator and the denominator are zero.
template <typename DataType>
DataType get_d_lapse_d_schwarzschild_r_from_lapse(
    const double n, const DataType lapse, const DataType schwarzschild_r) {
  return n *
         (2. * schwarzschild_r - 3. - 2. * schwarzschild_r * pow<2>(lapse)) /
         (schwarzschild_r *
          (schwarzschild_r - 2. + n * schwarzschild_r * lapse -
           schwarzschild_r * pow<2>(lapse)));
}

// Get the first integral in eq. (56) with upper bound lapse
double first_integral_above_threshold(const double n,
                                      const double integral_upper_bound,
                                      const double crit_lapse,
                                      const double crit_schwarzschild_r) {
  ASSERT((integral_upper_bound >= lapse_threshold) and
             (integral_upper_bound <= 1.),
         "invalid upper integration bound in eq. (56)."
         "Required: lapse_threshold <= integral_upper_bound"
         " <= 1. but given integral_upper_bound = "
             << integral_upper_bound
             << " and lapse_threshold = " << lapse_threshold);
  const auto integrand = [n, crit_lapse,
                          crit_schwarzschild_r](const double lapse) {
    const double schwarzschild_r = get_schwarzschild_r_from_lapse(
        n, lapse, crit_lapse, crit_schwarzschild_r);
    return log(schwarzschild_r) / pow<2>(lapse);
  };

  boost::math::quadrature::tanh_sinh<double> de_integrator{};
  return de_integrator.integrate(integrand, lapse_threshold,
                                 integral_upper_bound);
}

// Get C_0 from eq. (55)
double get_c_0(const double n, const double crit_lapse,
               const double crit_schwarzschild_r) {
  return first_integral_above_threshold(n, 1., crit_lapse,
                                        crit_schwarzschild_r);
}

// Get the first integral in eq. (54) with lower bound lapse
// note that this includes the minus sign already
double first_integral_below_threshold(const double n,
                                      const double integral_lower_bound,
                                      const double crit_lapse,
                                      const double crit_schwarzschild_r) {
  ASSERT(
      (integral_lower_bound < lapse_threshold) and (integral_lower_bound > 0.),
      "invalid lower integration bound in eq. (56)."
      "Required: 0. <= integral_lower_bound"
      " <= lapse_threshold but given integral_lower_bound = "
          << integral_lower_bound
          << " and lapse_threshold = " << lapse_threshold);
  const auto integrand = [n, crit_lapse,
                          crit_schwarzschild_r](const double lapse) {
    const double schwarzschild_r = get_schwarzschild_r_from_lapse(
        n, lapse, crit_lapse, crit_schwarzschild_r);
    const double d_lapse_d_schwarzschild_r =
        get_d_lapse_d_schwarzschild_r_from_lapse(n, lapse, schwarzschild_r);
    return (-1.) / (d_lapse_d_schwarzschild_r * lapse * schwarzschild_r);
  };

  boost::math::quadrature::tanh_sinh<double> de_integrator{};
  return de_integrator.integrate(integrand, integral_lower_bound,
                                 lapse_threshold);
}

// Get the isotropic r if the corresponding lapse is above
// lapse threshold
double get_isotropic_r_from_lapse(const double n, const double lapse,
                                  const double crit_lapse,
                                  const double crit_schwarzschild_r,
                                  const double c_0) {
  ASSERT((lapse >= 0.) and (lapse < 1.),
         "invalid lapse range!"
         "Required: [0, 1) but given "
             << lapse);
  if (lapse == 0.) {
    return 0.;
  } else if (lapse < lapse_threshold) {
    return pow(get_schwarzschild_r_from_lapse(n, lapse_threshold, crit_lapse,
                                              crit_schwarzschild_r),
               1. / lapse_threshold) *
           exp(first_integral_below_threshold(n, lapse, crit_lapse,
                                              crit_schwarzschild_r) -
               c_0);
  } else {
    return pow(get_schwarzschild_r_from_lapse(n, lapse, crit_lapse,
                                              crit_schwarzschild_r),
               1. / lapse) *
           exp(first_integral_above_threshold(n, lapse, crit_lapse,
                                              crit_schwarzschild_r) -
               c_0);
  }
}

// Get the lapse corresponding to a target isotropic_r by solving
// r as a function of lapse
double get_lapse_from_isotropic_r(const double n,
                                  const double target_isotropic_r,
                                  const double crit_lapse,
                                  const double crit_schwarzschild_r,
                                  const double c_0) {
  ASSERT(target_isotropic_r >= 0.,
         "isotropic r must be "
         "non-negative but given target_isotropic_r = "
             << target_isotropic_r);
  ASSERT(target_isotropic_r <= max_isotropic_r,
         "we do not support"
         " trumpet initial data with isotropic r greater than "
             << max_isotropic_r
             << "but given target_isotropic_r = " << target_isotropic_r);

  if (target_isotropic_r == 0.) {
    return 0.;
  } else {
    // define the upper bound of lapse for root finding
    const double max_lapse = sqrt(1. - 1. / max_isotropic_r);
    const auto isotropic_r_minus_target = [&](const double lapse) -> double {
      return (get_isotropic_r_from_lapse(n, lapse, crit_lapse,
                                         crit_schwarzschild_r, c_0) -
              target_isotropic_r);
    };
    return RootFinder::toms748(isotropic_r_minus_target, 0., max_lapse, 1.e-15,
                               1.e-15);
  }
}

// Set a DG grid of isotropic_r as a source grid
// for interpolation to user-specified grid
tnsr::I<DataVector, 1, Frame::Inertial> set_source_grid() {
  const DataVector used_for_size(num_of_elements * num_of_pts,
                                 std::numeric_limits<double>::signaling_NaN());
  auto source_grid = make_with_value<tnsr::I<DataVector, 1, Frame::Inertial>>(
      used_for_size, 0.);
  const auto logical_coords = logical_coordinates(source_mesh);

  for (size_t i = 0; i < num_of_elements; ++i) {
    const auto coord_map =
        domain::make_coordinate_map<Frame::ElementLogical, Frame::Inertial>(
            Affine{-1., 1., element_lower_bounds.at(i),
                   element_upper_bounds.at(i)});
    const auto inertial_coords = coord_map(logical_coords);

    for (size_t j = 0; j < num_of_pts; ++j) {
      (get<0>(source_grid))[j + i * num_of_pts] = (get<0>(inertial_coords))[j];
    }
  }
  return source_grid;
}

// Get the lapse corresponding to a source grid of isotropic_r for
// interpolation to user-specified isotropic_r
DataVector get_lapse_on_grid(
    const double n, const tnsr::I<DataVector, 1, Frame::Inertial>& grid,
    const double crit_lapse, const double crit_schwarzschild_r,
    const double c_0) {
  DataVector lapse_on_grid = get<0>(grid);
  for (size_t i = 0; i < get<0>(grid).size(); ++i) {
    lapse_on_grid[i] = get_lapse_from_isotropic_r(
        n, (get<0>(grid))[i], crit_lapse, crit_schwarzschild_r, c_0);
    ASSERT(abs(lapse_on_grid[i] - crit_lapse) > 1.e-6,
           "lapse on a source grid point is too close to the critical "
           "point where the"
           "radial derivative of lapse cannot be calculated directly."
           "The violating lapse has value "
               << lapse_on_grid[i]
               << " which is too close to the critical lapse " << crit_lapse);
  }
  return lapse_on_grid;
}

// Get the Schwarzschild R corresponding to a source gird of
// isotropic_r for interpolation to user-specified isotropic_r
DataVector get_schwarzschild_r_on_grid(const double n,
                                       const DataVector& lapse_on_grid,
                                       const double crit_lapse,
                                       const double crit_schwarzschild_r) {
  DataVector schwarzschild_r_on_grid = lapse_on_grid;
  for (size_t i = 0; i < lapse_on_grid.size(); ++i) {
    schwarzschild_r_on_grid[i] = get_schwarzschild_r_from_lapse(
        n, lapse_on_grid[i], crit_lapse, crit_schwarzschild_r);
    ASSERT(abs(schwarzschild_r_on_grid[i] - crit_schwarzschild_r) > 1.e-6,
           "Schwarzschild R on a source grid point is too close to the "
           "critical point where the"
           "radial derivative of lapse cannot be calculated directly."
           "The violating Schwarzschild R has value "
               << schwarzschild_r_on_grid[i]
               << " which is too close to the critical Schwarzschild R "
               << crit_schwarzschild_r);
  }
  return schwarzschild_r_on_grid;
}

// Return the first indices in user-defined grid of isotropic r
// where isotropic r > element_lower_bounds[i] for i=0, ..., num_of_elements.
// We assume the user_grid is sorted from small to large.
std::array<size_t, num_of_elements> get_element_partition_index(
    const tnsr::I<DataVector, 1, Frame::Inertial>& user_grid) {
  std::array<size_t, num_of_elements> lower_bound_indices{};
  for (size_t i = 0; i < num_of_elements; ++i) {
    const auto iter =
        std::lower_bound(get<0>(user_grid).begin(), get<0>(user_grid).end(),
                         element_lower_bounds.at(i));
    lower_bound_indices.at(i) =
        static_cast<size_t>(std::distance(get<0>(user_grid).begin(), iter));
  }
  return lower_bound_indices;
}

// Interpolate a tensor on source_grid to user_grid
template <size_t num_of_data_vectors>
std::array<DataVector, num_of_data_vectors> interpolate_to_user_grid(
    const std::array<DataVector, num_of_data_vectors>& data_on_source_grid,
    const tnsr::I<DataVector, 1, Frame::Inertial>& user_grid) {
  std::array<DataVector, num_of_data_vectors> data_on_user_grid;

  // sort the user grid to put user-grid pts into correct elements
  // for interpolation. Memorize the original index locations
  // to transform back after interpolation
  std::vector<size_t> sort_index(get<0>(user_grid).size());
  std::iota(sort_index.begin(), sort_index.end(), 0);
  alg::sort(sort_index, [&user_grid](size_t i, size_t j) {
    return get<0>(user_grid)[i] < get<0>(user_grid)[j];
  });
  auto sorted_user_grid = user_grid;
  for (size_t i = 0; i < get<0>(user_grid).size(); ++i) {
    get<0>(sorted_user_grid)[i] = get<0>(user_grid)[sort_index[i]];
  }

  const double requested_max_isotropic_r =
      get<0>(sorted_user_grid)[get<0>(user_grid).size() - 1];
  if (requested_max_isotropic_r > max_isotropic_r) {
    ERROR("The max isotropic radius supported is "
          << max_isotropic_r << "M but the max requested is "
          << requested_max_isotropic_r << "M");
  }

  const std::array<size_t, num_of_elements> lower_bound_indices =
      get_element_partition_index(sorted_user_grid);

  for (size_t k = 0; k < num_of_data_vectors; ++k) {
    data_on_user_grid.at(k) = get<0>(sorted_user_grid);
  }

  for (size_t i = 0; i < num_of_elements; ++i) {
    const tnsr::I<DataVector, 1, Frame::Inertial> user_subgrid_view;
    const size_t num_of_user_pts_in_element =
        (i == (num_of_elements - 1))
            ? (get<0>(sorted_user_grid)).size() - lower_bound_indices.at(i)
            : lower_bound_indices.at(i + 1) - lower_bound_indices.at(i);

    if (num_of_user_pts_in_element > 0) {
      make_const_view(make_not_null(&get<0>(user_subgrid_view)),
                      get<0>(sorted_user_grid), lower_bound_indices.at(i),
                      num_of_user_pts_in_element);
      // transform user_subgrid_view to logical coordinates for interpolation
      const auto coord_map =
          domain::make_coordinate_map<Frame::Inertial, Frame::ElementLogical>(
              Affine{element_lower_bounds.at(i), element_upper_bounds.at(i),
                     -1., 1.});
      const intrp::Irregular interpolant(source_mesh,
                                         coord_map(user_subgrid_view));

      for (size_t k = 0; k < num_of_data_vectors; ++k) {
        const DataVector data_on_source_subgrid;
        make_const_view(make_not_null(&data_on_source_subgrid),
                        data_on_source_grid.at(k), i * num_of_pts, num_of_pts);
        DataVector data_on_user_subgrid =
            interpolant.interpolate(data_on_source_subgrid);

        for (size_t j = 0; j < num_of_user_pts_in_element; ++j) {
          (data_on_user_grid.at(k))[sort_index[lower_bound_indices.at(i) + j]] =
              data_on_user_subgrid[j];
        }
      }
    }
  }

  return data_on_user_grid;
}

template <size_t num_of_data_vectors>
std::array<double, num_of_data_vectors> interpolate_to_user_grid(
    const std::array<DataVector, num_of_data_vectors>& data_on_source_grid,
    const tnsr::I<double, 1, Frame::Inertial>& user_grid) {
  std::array<double, num_of_data_vectors> result{};
  auto temp_user_grid =
      make_with_value<tnsr::I<DataVector, 1, Frame::Inertial>>(
          DataVector(1, std::numeric_limits<double>::signaling_NaN()),
          std::numeric_limits<double>::signaling_NaN());
  get<0>(temp_user_grid)[0] = get<0>(user_grid);
  const auto data_on_user_grid =
      interpolate_to_user_grid(data_on_source_grid, temp_user_grid);

  for (size_t i = 0; i < num_of_data_vectors; ++i) {
    result.at(i) = (data_on_user_grid.at(i))[0];
  }
  return result;
}

// Get the isotropic r on user grid
template <typename DataType>
constexpr tnsr::I<DataType, 1, Frame::Inertial> get_isotropic_r_on_grid(
    const tnsr::I<DataType, gr::Solutions::TrumpetSchwarzschild::volume_dim,
                  Frame::Inertial>& x,
    const double mass) {
  tnsr::I<DataType, 1, Frame::Inertial> isotropic_r_on_grid{
      sqrt(pow<2>(get<0>(x)) + pow<2>(get<1>(x)) + pow<2>(get<2>(x))) / mass};
  return isotropic_r_on_grid;
}
}  // namespace

namespace gr::Solutions {
const tnsr::I<DataVector, 1, Frame::Inertial>
    TrumpetSchwarzschild::source_grid_ = set_source_grid();

TrumpetSchwarzschild::TrumpetSchwarzschild(const double mass, const double n,
                                           const Options::Context& context)
    : mass_(mass), n_(n) {
  if (mass <= 0.) {
    PARSE_ERROR(context,
                "Black hole mass must be positive, but given " << mass_);
  }
  if (n < 0.) {
    PARSE_ERROR(context, "Parameter n must be non-negative, but given " << n);
  }

  const double crit_lapse = get_crit_lapse(n);
  const double crit_schwarzschild_r = get_crit_schwarzschild_r(n);
  const double c_0 = get_c_0(n, crit_lapse, crit_schwarzschild_r);

  data_on_source_grid_.at(0) =
      get_lapse_on_grid(n, source_grid_, crit_lapse, crit_schwarzschild_r, c_0);
  data_on_source_grid_.at(1) = get_schwarzschild_r_on_grid(
      n, data_on_source_grid_.at(0), crit_lapse, crit_schwarzschild_r);
}

void TrumpetSchwarzschild::pup(PUP::er& p) {
  p | mass_;
  p | n_;
  p | data_on_source_grid_;
}

// We first compute lapse, schwarzchid_r, d_lapse_d_schwarzschild_r
// on a source isotropic grid before interpolating to the user-
// specified isotropic grid corresponding to x
template <typename DataType>
TrumpetSchwarzschild::IntermediateVars<DataType>::IntermediateVars(
    const double mass, const double n,
    const tnsr::I<DataType, volume_dim, Frame::Inertial>& x, const double /*t*/,
    const std::array<DataVector, 2>& data_on_source_grid)
    : one_over_mass(1. / mass), one_over_n(1. / n) {
  auto target_grid = get_isotropic_r_on_grid(x, mass);
  const auto data_on_user_grid =
      interpolate_to_user_grid(data_on_source_grid, target_grid);

  d_lapse_d_schwarzschild_r_on_user_grid =
      get_d_lapse_d_schwarzschild_r_from_lapse(n, data_on_user_grid.at(0),
                                               data_on_user_grid.at(1));
  lapse_on_user_grid = std::move(data_on_user_grid.at(0));
  one_over_schwarzschild_r_on_user_grid = 1. / data_on_user_grid.at(1);
  schwarzschild_r_on_user_grid = std::move(data_on_user_grid.at(1));
  one_over_isotropic_r_on_user_grid = 1. / get<0>(target_grid);
  isotropic_r_on_user_grid = std::move(get<0>(target_grid));
}

template <typename DataType>
auto TrumpetSchwarzschild::variables(
    const tnsr::I<DataType, volume_dim, Frame::Inertial>& /*x*/,
    const double /*t*/, const IntermediateVars<DataType>& vars,
    tmpl::list<gr::Tags::Lapse<DataType>> /*meta*/) const
    -> tuples::TaggedTuple<gr::Tags::Lapse<DataType>> {
  return {Scalar<DataType>{vars.lapse_on_user_grid}};
}

template <typename DataType>
auto TrumpetSchwarzschild::variables(
    const tnsr::I<DataType, volume_dim, Frame::Inertial>& /*x*/,
    const double /*t*/, const IntermediateVars<DataType>& vars,
    tmpl::list<::Tags::dt<gr::Tags::Lapse<DataType>>> /*meta*/) const
    -> tuples::TaggedTuple<::Tags::dt<gr::Tags::Lapse<DataType>>> {
  return {make_with_value<Scalar<DataType>>(vars.lapse_on_user_grid, 0.)};
}

template <typename DataType>
auto TrumpetSchwarzschild::variables(
    const tnsr::I<DataType, volume_dim, Frame::Inertial>& x, const double /*t*/,
    const IntermediateVars<DataType>& vars,
    tmpl::list<DerivLapse<DataType>> /*meta*/) const
    -> tuples::TaggedTuple<DerivLapse<DataType>> {
  tnsr::i<DataType, volume_dim, Frame::Inertial> d_lapse;
  for (size_t i = 0; i < volume_dim; ++i) {
    // note that we divide by mass_ to get the correct dimension
    d_lapse.get(i) = vars.d_lapse_d_schwarzschild_r_on_user_grid *
                     vars.lapse_on_user_grid *
                     vars.schwarzschild_r_on_user_grid * x.get(i) *
                     pow<2>(vars.one_over_isotropic_r_on_user_grid) *
                     pow<2>(vars.one_over_mass);
  }
  return d_lapse;
}

template <typename DataType>
auto TrumpetSchwarzschild::variables(
    const tnsr::I<DataType, volume_dim, Frame::Inertial>& x, const double /*t*/,
    const IntermediateVars<DataType>& vars,
    tmpl::list<gr::Tags::Shift<DataType, volume_dim>> /*meta*/) const
    -> tuples::TaggedTuple<gr::Tags::Shift<DataType, volume_dim>> {
  tnsr::I<DataType, volume_dim, Frame::Inertial> shift;
  for (size_t i = 0; i < volume_dim; ++i) {
    shift.get(i) = x.get(i) * vars.one_over_mass * sqrt(get_c_n_squared(n_)) *
                   exp(vars.lapse_on_user_grid * vars.one_over_n) *
                   pow<3>(vars.one_over_schwarzschild_r_on_user_grid);
  }
  return shift;
}

template <typename DataType>
auto TrumpetSchwarzschild::variables(
    const tnsr::I<DataType, volume_dim, Frame::Inertial>& x, const double /*t*/,
    const IntermediateVars<DataType>& /*vars*/,
    tmpl::list<::Tags::dt<gr::Tags::Shift<DataType, volume_dim>>> /*meta*/)
    const
    -> tuples::TaggedTuple<::Tags::dt<gr::Tags::Shift<DataType, volume_dim>>> {
  return {
      make_with_value<tnsr::I<DataType, volume_dim, Frame::Inertial>>(x, 0.)};
}

template <typename DataType>
auto TrumpetSchwarzschild::variables(
    const tnsr::I<DataType, volume_dim, Frame::Inertial>& x, const double /*t*/,
    const IntermediateVars<DataType>& vars,
    tmpl::list<DerivShift<DataType>> /*meta*/) const
    -> tuples::TaggedTuple<DerivShift<DataType>> {
  tnsr::iJ<DataType, volume_dim, Frame::Inertial> d_shift;
  DataType sqrt_of_lapse_squared_minus_f =
      sqrt(get_c_n_squared(n_)) *
      exp(vars.lapse_on_user_grid * vars.one_over_n) *
      pow<2>(vars.one_over_schwarzschild_r_on_user_grid);

  for (size_t i = 0; i < volume_dim; ++i) {
    for (size_t j = 0; j < volume_dim; ++j) {
      DataType lapse_times_xi_xj_over_isotropic_r_squared =
          vars.lapse_on_user_grid * x.get(i) * x.get(j) *
          pow<2>(vars.one_over_mass) *
          pow<2>(vars.one_over_isotropic_r_on_user_grid);

      d_shift.get(i, j) =
          lapse_times_xi_xj_over_isotropic_r_squared *
          (1. / sqrt_of_lapse_squared_minus_f *
               (vars.lapse_on_user_grid *
                    vars.d_lapse_d_schwarzschild_r_on_user_grid -
                1. * pow<2>(vars.one_over_schwarzschild_r_on_user_grid)) -
           sqrt_of_lapse_squared_minus_f *
               vars.one_over_schwarzschild_r_on_user_grid);

      if (i == j) {
        d_shift.get(i, j) += sqrt_of_lapse_squared_minus_f *
                             vars.one_over_schwarzschild_r_on_user_grid;
      }
      // divide by mass to restore unit
      d_shift.get(i, j) *= vars.one_over_mass;
    }
  }
  return d_shift;
}

template <typename DataType>
auto TrumpetSchwarzschild::variables(
    const tnsr::I<DataType, volume_dim, Frame::Inertial>& x, const double /*t*/,
    const IntermediateVars<DataType>& vars,
    tmpl::list<gr::Tags::SpatialMetric<DataType, volume_dim>> /*meta*/) const
    -> tuples::TaggedTuple<gr::Tags::SpatialMetric<DataType, volume_dim>> {
  auto spatial_metric =
      make_with_value<tnsr::ii<DataType, volume_dim, Frame::Inertial>>(x, 0.);
  for (size_t i = 0; i < volume_dim; ++i) {
    spatial_metric.get(i, i) = pow<2>(vars.schwarzschild_r_on_user_grid *
                                      vars.one_over_isotropic_r_on_user_grid);
  }
  return spatial_metric;
}

template <typename DataType>
auto TrumpetSchwarzschild::variables(
    const tnsr::I<DataType, volume_dim, Frame::Inertial>& x, const double /*t*/,
    const IntermediateVars<DataType>& /*vars*/,
    tmpl::list<
        ::Tags::dt<gr::Tags::SpatialMetric<DataType, volume_dim>>> /*meta*/)
    const -> tuples::TaggedTuple<
              ::Tags::dt<gr::Tags::SpatialMetric<DataType, volume_dim>>> {
  auto dt_spatial_metric =
      make_with_value<tnsr::ii<DataType, volume_dim, Frame::Inertial>>(x, 0.);
  return dt_spatial_metric;
}

template <typename DataType>
auto TrumpetSchwarzschild::variables(
    const tnsr::I<DataType, volume_dim, Frame::Inertial>& x, const double /*t*/,
    const IntermediateVars<DataType>& vars,
    tmpl::list<DerivSpatialMetric<DataType>> /*meta*/) const
    -> tuples::TaggedTuple<DerivSpatialMetric<DataType>> {
  auto d_spatial_metric =
      make_with_value<tnsr::ijj<DataType, volume_dim, Frame::Inertial>>(x, 0.);

  for (size_t i = 0; i < volume_dim; ++i) {
    for (size_t j = 0; j < volume_dim; ++j) {
      d_spatial_metric.get(i, j, j) =
          2. * pow<2>(vars.schwarzschild_r_on_user_grid) * x.get(i) *
          pow<4>(vars.one_over_isotropic_r_on_user_grid) *
          (vars.lapse_on_user_grid - 1.) * pow<2>(vars.one_over_mass);
    }
  }
  return d_spatial_metric;
}

template <typename DataType>
auto TrumpetSchwarzschild::variables(
    const tnsr::I<DataType, volume_dim, Frame::Inertial>& /*x*/,
    const double /*t*/, const IntermediateVars<DataType>& vars,
    tmpl::list<gr::Tags::SqrtDetSpatialMetric<DataType>> /*meta*/) const
    -> tuples::TaggedTuple<gr::Tags::SqrtDetSpatialMetric<DataType>> {
  return {Scalar<DataType>{pow<3>(vars.schwarzschild_r_on_user_grid *
                                  vars.one_over_isotropic_r_on_user_grid)}};
}

template <typename DataType>
auto TrumpetSchwarzschild::variables(
    const tnsr::I<DataType, volume_dim, Frame::Inertial>& x, const double /*t*/,
    const IntermediateVars<DataType>& vars,
    tmpl::list<gr::Tags::ExtrinsicCurvature<DataType, volume_dim>> /*meta*/)
    const
    -> tuples::TaggedTuple<gr::Tags::ExtrinsicCurvature<DataType, volume_dim>> {
  tnsr::ii<DataType, volume_dim, Frame::Inertial> extrinsic_curvature;
  DataType sqrt_lapse_squared_minus_f =
      sqrt(get_c_n_squared(n_)) *
      exp(vars.lapse_on_user_grid * vars.one_over_n) *
      pow<2>(vars.one_over_schwarzschild_r_on_user_grid);
  DataType diagonal_common_factor =
      vars.schwarzschild_r_on_user_grid * sqrt_lapse_squared_minus_f *
      pow<2>(vars.one_over_isotropic_r_on_user_grid);
  DataType off_diagonal_indep_term =
      (pow<2>(vars.schwarzschild_r_on_user_grid) * vars.lapse_on_user_grid *
           vars.d_lapse_d_schwarzschild_r_on_user_grid -
       1.) *
      pow<4>(vars.one_over_isotropic_r_on_user_grid) /
      (sqrt_lapse_squared_minus_f);

  for (size_t i = 0; i < volume_dim; ++i) {
    for (size_t j = 0; j <= i; ++j) {
      extrinsic_curvature.get(i, j) =
          x.get(i) * x.get(j) * pow<2>(vars.one_over_mass) *
              off_diagonal_indep_term +
          diagonal_common_factor *
              (((i == j) ? 1. : 0.) -
               x.get(i) * x.get(j) * pow<2>(vars.one_over_mass) *
                   pow<2>(vars.one_over_isotropic_r_on_user_grid));

      // divide by mass to get the correct dimension
      extrinsic_curvature.get(i, j) *= vars.one_over_mass;
    }
  }
  return extrinsic_curvature;
}

template <typename DataType>
auto TrumpetSchwarzschild::variables(
    const tnsr::I<DataType, volume_dim, Frame::Inertial>& x, const double /*t*/,
    const IntermediateVars<DataType>& vars,
    tmpl::list<gr::Tags::InverseSpatialMetric<DataType, volume_dim>> /*meta*/)
    const -> tuples::TaggedTuple<
              gr::Tags::InverseSpatialMetric<DataType, volume_dim>> {
  auto inverse_spatial_metric =
      make_with_value<tnsr::II<DataType, volume_dim, Frame::Inertial>>(x, 0.);
  for (size_t i = 0; i < volume_dim; ++i) {
    inverse_spatial_metric.get(i, i) =
        pow<2>(vars.isotropic_r_on_user_grid *
               vars.one_over_schwarzschild_r_on_user_grid);
  }
  return inverse_spatial_metric;
}

bool operator==(const TrumpetSchwarzschild& lhs,
                const TrumpetSchwarzschild& rhs) {
  return lhs.mass() == rhs.mass() and lhs.n() == rhs.n();
}

bool operator!=(const TrumpetSchwarzschild& lhs,
                const TrumpetSchwarzschild& rhs) {
  return not(lhs == rhs);
}

#define DIM(data) BOOST_PP_TUPLE_ELEM(0, data)
#define DTYPE(data) BOOST_PP_TUPLE_ELEM(1, data)

#define INSTANTIATE(_, data)                                                   \
  template TrumpetSchwarzschild::IntermediateVars<DTYPE(data)>::               \
      IntermediateVars(                                                        \
          const double mass, const double n,                                   \
          const tnsr::I<DTYPE(data), DIM(data), Frame::Inertial>& x,           \
          const double /*t*/,                                                  \
          const std::array<DataVector, 2>& data_on_source_grid);               \
  template tuples::TaggedTuple<gr::Tags::Lapse<DTYPE(data)>>                   \
  TrumpetSchwarzschild::variables(                                             \
      const tnsr::I<DTYPE(data), DIM(data), Frame::Inertial>& /*x*/,           \
      const double /*t*/,                                                      \
      const TrumpetSchwarzschild::IntermediateVars<DTYPE(data)>& vars,         \
      tmpl::list<gr::Tags::Lapse<DTYPE(data)>> /*meta*/) const;                \
  template tuples::TaggedTuple<::Tags::dt<gr::Tags::Lapse<DTYPE(data)>>>       \
  TrumpetSchwarzschild::variables(                                             \
      const tnsr::I<DTYPE(data), DIM(data)>& /*x*/, const double /*t*/,        \
      const TrumpetSchwarzschild::IntermediateVars<DTYPE(data)>& vars,         \
      tmpl::list<::Tags::dt<gr::Tags::Lapse<DTYPE(data)>>> /*meta*/) const;    \
  template tuples::TaggedTuple<::Tags::deriv<                                  \
      gr::Tags::Lapse<DTYPE(data)>, tmpl::size_t<DIM(data)>, Frame::Inertial>> \
  TrumpetSchwarzschild::variables(                                             \
      const tnsr::I<DTYPE(data), DIM(data)>& x, const double /*t*/,            \
      const TrumpetSchwarzschild::IntermediateVars<DTYPE(data)>& vars,         \
      tmpl::list<                                                              \
          ::Tags::deriv<gr::Tags::Lapse<DTYPE(data)>, tmpl::size_t<DIM(data)>, \
                        Frame::Inertial>> /*meta*/) const;                     \
  template tuples::TaggedTuple<gr::Tags::Shift<DTYPE(data), DIM(data)>>        \
  TrumpetSchwarzschild::variables(                                             \
      const tnsr::I<DTYPE(data), DIM(data)>& x, const double /*t*/,            \
      const TrumpetSchwarzschild::IntermediateVars<DTYPE(data)>& vars,         \
      tmpl::list<gr::Tags::Shift<DTYPE(data), DIM(data)>> /*meta*/) const;     \
  template tuples::TaggedTuple<                                                \
      ::Tags::dt<gr::Tags::Shift<DTYPE(data), DIM(data)>>>                     \
  TrumpetSchwarzschild::variables(                                             \
      const tnsr::I<DTYPE(data), DIM(data)>& x, const double /*t*/,            \
      const TrumpetSchwarzschild::IntermediateVars<DTYPE(data)>& vars,         \
      tmpl::list<                                                              \
          ::Tags::dt<gr::Tags::Shift<DTYPE(data), DIM(data)>>> /*meta*/)       \
      const;                                                                   \
  template tuples::TaggedTuple<                                                \
      ::Tags::deriv<gr::Tags::Shift<DTYPE(data), DIM(data)>,                   \
                    tmpl::size_t<DIM(data)>, Frame::Inertial>>                 \
  TrumpetSchwarzschild::variables(                                             \
      const tnsr::I<DTYPE(data), DIM(data)>& x, const double /*t*/,            \
      const TrumpetSchwarzschild::IntermediateVars<DTYPE(data)>& vars,         \
      tmpl::list<                                                              \
          ::Tags::deriv<gr::Tags::Shift<DTYPE(data), DIM(data)>,               \
                        tmpl::size_t<DIM(data)>, Frame::Inertial>> /*meta*/)   \
      const;                                                                   \
  template tuples::TaggedTuple<                                                \
      gr::Tags::SpatialMetric<DTYPE(data), DIM(data)>>                         \
  TrumpetSchwarzschild::variables(                                             \
      const tnsr::I<DTYPE(data), DIM(data)>& x, const double /*t*/,            \
      const TrumpetSchwarzschild::IntermediateVars<DTYPE(data)>& vars,         \
      tmpl::list<gr::Tags::SpatialMetric<DTYPE(data), DIM(data)>> /*meta*/)    \
      const;                                                                   \
  template tuples::TaggedTuple<                                                \
      ::Tags::dt<gr::Tags::SpatialMetric<DTYPE(data), DIM(data)>>>             \
  TrumpetSchwarzschild::variables(                                             \
      const tnsr::I<DTYPE(data), DIM(data)>& x, const double /*t*/,            \
      const TrumpetSchwarzschild::IntermediateVars<DTYPE(data)>& vars,         \
      tmpl::list<::Tags::dt<                                                   \
          gr::Tags::SpatialMetric<DTYPE(data), DIM(data)>>> /*meta*/) const;   \
  template tuples::TaggedTuple<                                                \
      ::Tags::deriv<gr::Tags::SpatialMetric<DTYPE(data), DIM(data)>,           \
                    tmpl::size_t<DIM(data)>, Frame::Inertial>>                 \
  TrumpetSchwarzschild::variables(                                             \
      const tnsr::I<DTYPE(data), DIM(data)>& x, const double /*t*/,            \
      const TrumpetSchwarzschild::IntermediateVars<DTYPE(data)>& vars,         \
      tmpl::list<                                                              \
          ::Tags::deriv<gr::Tags::SpatialMetric<DTYPE(data), DIM(data)>,       \
                        tmpl::size_t<DIM(data)>, Frame::Inertial>> /*meta*/)   \
      const;                                                                   \
  template tuples::TaggedTuple<                                                \
      gr::Tags::InverseSpatialMetric<DTYPE(data), DIM(data)>>                  \
  TrumpetSchwarzschild::variables(                                             \
      const tnsr::I<DTYPE(data), DIM(data)>& x, const double /*t*/,            \
      const TrumpetSchwarzschild::IntermediateVars<DTYPE(data)>& vars,         \
      tmpl::list<                                                              \
          gr::Tags::InverseSpatialMetric<DTYPE(data), DIM(data)>> /*meta*/)    \
      const;                                                                   \
  template tuples::TaggedTuple<                                                \
      gr::Tags::ExtrinsicCurvature<DTYPE(data), DIM(data)>>                    \
  TrumpetSchwarzschild::variables(                                             \
      const tnsr::I<DTYPE(data), DIM(data)>& x, const double /*t*/,            \
      const TrumpetSchwarzschild::IntermediateVars<DTYPE(data)>& vars,         \
      tmpl::list<                                                              \
          gr::Tags::ExtrinsicCurvature<DTYPE(data), DIM(data)>> /*meta*/)      \
      const;                                                                   \
  template tuples::TaggedTuple<gr::Tags::SqrtDetSpatialMetric<DTYPE(data)>>    \
  TrumpetSchwarzschild::variables(                                             \
      const tnsr::I<DTYPE(data), DIM(data)>& /*x*/, const double /*t*/,        \
      const TrumpetSchwarzschild::IntermediateVars<DTYPE(data)>& vars,         \
      tmpl::list<gr::Tags::SqrtDetSpatialMetric<DTYPE(data)>> /*meta*/) const;

GENERATE_INSTANTIATIONS(INSTANTIATE, (3), (double, DataVector))

#undef INSTANTIATE
#undef DTYPE
#undef DIM
}  // namespace gr::Solutions
