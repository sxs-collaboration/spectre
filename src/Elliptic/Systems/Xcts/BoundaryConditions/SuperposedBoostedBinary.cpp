// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Elliptic/Systems/Xcts/BoundaryConditions/SuperposedBoostedBinary.hpp"

#include <brigand/brigand.hpp>

#include <algorithm>
#include <array>
#include <cstddef>
#include <memory>
#include <optional>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/EagerMath/DeterminantAndInverse.hpp"
#include "DataStructures/Tensor/IndexType.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "DataStructures/Tensor/TypeAliases.hpp"
#include "Elliptic/BoundaryConditions/BoundaryConditionType.hpp"
#include "Elliptic/Systems/Xcts/Tags.hpp"
#include "PointwiseFunctions/AnalyticSolutions/Xcts/Factory.hpp"
#include "PointwiseFunctions/AnalyticSolutions/Xcts/Schwarzschild.hpp"
#include "PointwiseFunctions/GeneralRelativity/Lapse.hpp"
#include "PointwiseFunctions/GeneralRelativity/Shift.hpp"
#include "PointwiseFunctions/GeneralRelativity/SpacetimeMetric.hpp"
#include "PointwiseFunctions/GeneralRelativity/SpatialMetric.hpp"
#include "PointwiseFunctions/InitialDataUtilities/AnalyticSolution.hpp"
#include "PointwiseFunctions/SpecialRelativity/LorentzBoostMatrix.hpp"
#include "Utilities/CallWithDynamicType.hpp"
#include "Utilities/ConstantExpressions.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/TaggedTuple.hpp"

namespace Xcts::BoundaryConditions {

namespace {

template <typename IsolatedObjectBase, typename IsolatedObjectClasses>
void implement_apply_dirichlet(
    const gsl::not_null<Scalar<DataVector>*> conformal_factor_minus_one,
    const gsl::not_null<Scalar<DataVector>*>
        lapse_times_conformal_factor_minus_one,
    const gsl::not_null<tnsr::I<DataVector, 3>*> shift_excess,
    const std::array<std::optional<std::unique_ptr<IsolatedObjectBase>>, 2>&
        superposed_objects,
    const std::array<double, 2>& xcoords, const std::array<double, 2>& masses,
    const std::array<double, 3>& momentum_left,
    const std::array<double, 3>& momentum_right, const double y_offset,
    const double z_offset, const tnsr::I<DataVector, 3>& x) {
  using analytic_tags =
      tmpl::list<Xcts::Tags::ConformalFactorMinusOne<DataVector>,
                 Xcts::Tags::LapseTimesConformalFactorMinusOne<DataVector>,
                 Xcts::Tags::ShiftExcess<DataVector, 3, Frame::Inertial>>;

  std::array<tnsr::I<DataVector, 3>, 2> x_isolated{{x, x}};
  const std::array<std::array<double, 3>, 2> coords_isolated{
      {{{xcoords[0], y_offset, z_offset}}, {{xcoords[1], y_offset, z_offset}}}};
  for (size_t i = 0; i < 2; ++i) {
    for (size_t dim = 0; dim < 3; dim++) {
      gsl::at(x_isolated, i).get(dim) -=
          gsl::at(gsl::at(coords_isolated, i), dim);
    }
  }
  std::array<tnsr::I<DataVector, 3>, 2> x_unboosted{{x, x}};
  sr::lorentz_boost(make_not_null(&(gsl::at(x_unboosted, 0))),
                    gsl::at(x_isolated, 0), 0., momentum_left / masses[0]);
  sr::lorentz_boost(make_not_null(&(gsl::at(x_unboosted, 1))),
                    gsl::at(x_isolated, 1), 0., momentum_right / masses[1]);

  const auto solution_left =
      call_with_dynamic_type<tuples::tagged_tuple_from_typelist<analytic_tags>,
                             IsolatedObjectClasses>(
          (*(superposed_objects[0])).get(),
          [&x_unboosted](const auto* const local_solution) {
            return local_solution->variables(gsl::at(x_unboosted, 0),
                                             analytic_tags{});
          });
  const auto solution_right =
      call_with_dynamic_type<tuples::tagged_tuple_from_typelist<analytic_tags>,
                             IsolatedObjectClasses>(
          (*(superposed_objects[1])).get(),
          [&x_unboosted](const auto* const local_solution) {
            return local_solution->variables(gsl::at(x_unboosted, 1),
                                             analytic_tags{});
          });
  Scalar<DataVector> conformal_factor_minus_one_left =
      get<Xcts::Tags::ConformalFactorMinusOne<DataVector>>(solution_left);
  Scalar<DataVector> conformal_factor_minus_one_right =
      get<Xcts::Tags::ConformalFactorMinusOne<DataVector>>(solution_right);
  Scalar<DataVector> lapse_times_conformal_factor_minus_one_left =
      get<Xcts::Tags::LapseTimesConformalFactorMinusOne<DataVector>>(
          solution_left);
  Scalar<DataVector> lapse_times_conformal_factor_minus_one_right =
      get<Xcts::Tags::LapseTimesConformalFactorMinusOne<DataVector>>(
          solution_right);
  tnsr::I<DataVector, 3> shift_excess_left =
      get<Xcts::Tags::ShiftExcess<DataVector, 3, Frame::Inertial>>(
          solution_left);
  tnsr::I<DataVector, 3> shift_excess_right =
      get<Xcts::Tags::ShiftExcess<DataVector, 3, Frame::Inertial>>(
          solution_right);

  // Boosted spacetime left
  Scalar<DataVector> conformal_factor_left{get<0>(shift_excess_left).size()};
  get(conformal_factor_left) = 1. + get(conformal_factor_minus_one_left);
  Scalar<DataVector> lapse_left{get<0>(shift_excess_left).size()};
  get(lapse_left) = (1. + get(lapse_times_conformal_factor_minus_one_left)) /
                    get(conformal_factor_left);

  tnsr::ii<DataVector, 3> spatial_metric_left{get<0>(shift_excess_left).size()};
  std::fill(spatial_metric_left.begin(), spatial_metric_left.end(), 0.);
  for (size_t i = 0; i < 3; ++i) {
    spatial_metric_left.get(i, i) = square(square(get(conformal_factor_left)));
  }

  tnsr::aa<DataVector, 3> spacetime_metric_left{
      get<0>(shift_excess_left).size()};
  gr::spacetime_metric(make_not_null(&spacetime_metric_left), lapse_left,
                       shift_excess_left, spatial_metric_left);

  const tnsr::Ab<double, 3, Frame::NoFrame> lorentz_boost_matrix_double_left =
      sr::lorentz_boost_matrix(-(momentum_left / masses[0]));
  tnsr::Ab<DataVector, 3> lorentz_boost_matrix_left{
      get<0>(shift_excess_left).size()};
  for (size_t i = 0; i < 4; ++i) {
    for (size_t j = 0; j < 4; ++j) {
      lorentz_boost_matrix_left.get(i, j) =
          lorentz_boost_matrix_double_left.get(i, j);
    }
  }

  tnsr::aa<DataVector, 3> spacetime_metric_boosted_left{
      get<0>(shift_excess_left).size()};
  for (size_t i = 0; i < 4; ++i) {
    for (size_t j = 0; j <= i; ++j) {
      spacetime_metric_boosted_left.get(i, j) = 0.;
      for (size_t k = 0; k < 4; ++k) {
        for (size_t l = 0; l < 4; ++l) {
          spacetime_metric_boosted_left.get(i, j) +=
              lorentz_boost_matrix_left.get(k, i) *
              lorentz_boost_matrix_left.get(l, j) *
              spacetime_metric_left.get(k, l);
        }
      }
    }
  }

  // Boosted spacetime right
  Scalar<DataVector> conformal_factor_right{get<0>(shift_excess_right).size()};
  get(conformal_factor_right) = 1. + get(conformal_factor_minus_one_right);
  Scalar<DataVector> lapse_right{get<0>(shift_excess_right).size()};
  get(lapse_right) = (1. + get(lapse_times_conformal_factor_minus_one_right)) /
                     get(conformal_factor_right);

  tnsr::ii<DataVector, 3> spatial_metric_right{
      get<0>(shift_excess_right).size()};
  std::fill(spatial_metric_right.begin(), spatial_metric_right.end(), 0.);
  for (size_t i = 0; i < 3; ++i) {
    spatial_metric_right.get(i, i) =
        square(square(get(conformal_factor_right)));
  }

  tnsr::aa<DataVector, 3> spacetime_metric_right{
      get<0>(shift_excess_right).size()};
  gr::spacetime_metric(make_not_null(&spacetime_metric_right), lapse_right,
                       shift_excess_right, spatial_metric_right);

  const tnsr::Ab<double, 3, Frame::NoFrame> lorentz_boost_matrix_double_right =
      sr::lorentz_boost_matrix(-(momentum_right / masses[1]));
  tnsr::Ab<DataVector, 3> lorentz_boost_matrix_right{
      get<0>(shift_excess_right).size()};
  for (size_t i = 0; i < 4; ++i) {
    for (size_t j = 0; j < 4; ++j) {
      lorentz_boost_matrix_right.get(i, j) =
          lorentz_boost_matrix_double_right.get(i, j);
    }
  }

  tnsr::aa<DataVector, 3> spacetime_metric_boosted_right{
      get<0>(shift_excess_right).size()};
  for (size_t i = 0; i < 4; ++i) {
    for (size_t j = 0; j <= i; ++j) {
      spacetime_metric_boosted_right.get(i, j) = 0.;
      for (size_t k = 0; k < 4; ++k) {
        for (size_t l = 0; l < 4; ++l) {
          spacetime_metric_boosted_right.get(i, j) +=
              lorentz_boost_matrix_right.get(k, i) *
              lorentz_boost_matrix_right.get(l, j) *
              spacetime_metric_right.get(k, l);
        }
      }
    }
  }

  // Superposed lapse and shift (no longer conformally flat
  //  so conformal factor is 1. and "boosted spatial metric" should be on
  //  background)
  get(*conformal_factor_minus_one) = 0.;

  const auto conformal_metric_left =
      gr::spatial_metric(spacetime_metric_boosted_left);
  const auto inv_conformal_metric_left =
      determinant_and_inverse(conformal_metric_left).second;
  const auto boosted_shift_left =
      gr::shift(spacetime_metric_boosted_left, inv_conformal_metric_left);
  const Scalar<DataVector> boosted_lapse_left =
      gr::lapse(boosted_shift_left, spacetime_metric_boosted_left);

  const auto conformal_metric_right =
      gr::spatial_metric(spacetime_metric_boosted_right);
  const auto inv_conformal_metric_right =
      determinant_and_inverse(conformal_metric_right).second;
  const auto boosted_shift_right =
      gr::shift(spacetime_metric_boosted_right, inv_conformal_metric_right);
  const Scalar<DataVector> boosted_lapse_right =
      gr::lapse(boosted_shift_right, spacetime_metric_boosted_right);

  get(*lapse_times_conformal_factor_minus_one) =
      (get(boosted_lapse_left) * get(boosted_lapse_right) - 1.);

  for (size_t i = 0; i < 3; ++i) {
    shift_excess->get(i) =
        boosted_shift_left.get(i) + boosted_shift_right.get(i);
  }
}

}  // namespace

template <typename IsolatedObjectBase, typename IsolatedObjectClasses>
void SuperposedBoostedBinary<IsolatedObjectBase, IsolatedObjectClasses>::apply(
    const gsl::not_null<Scalar<DataVector>*> conformal_factor_minus_one,
    const gsl::not_null<Scalar<DataVector>*>
        lapse_times_conformal_factor_minus_one,
    const gsl::not_null<tnsr::I<DataVector, 3>*> shift_excess,
    const gsl::not_null<
        Scalar<DataVector>*> /*n_dot_conformal_factor_gradient*/,
    const gsl::not_null<Scalar<DataVector>*>
    /*n_dot_lapse_times_conformal_factor_gradient*/,
    const gsl::not_null<tnsr::I<DataVector, 3>*>
    /*n_dot_longitudinal_shift_excess*/,
    const tnsr::i<DataVector, 3>& /*deriv_conformal_factor_correction1*/,
    const tnsr::i<DataVector,
                  3>& /*deriv_lapse_times_conformal_factor_correction1*/,
    const tnsr::iJ<DataVector, 3>& /*deriv_shift_excess_correction1*/,
    const tnsr::I<DataVector, 3>& x) const {
  implement_apply_dirichlet<IsolatedObjectBase, IsolatedObjectClasses>(
      conformal_factor_minus_one, lapse_times_conformal_factor_minus_one,
      shift_excess, superposed_objects_, xcoords_, masses_, momentum_left_,
      momentum_right_, y_offset_, z_offset_, x);
}

template <typename IsolatedObjectBase, typename IsolatedObjectClasses>
void SuperposedBoostedBinary<IsolatedObjectBase, IsolatedObjectClasses>::
    apply_linearized(
        const gsl::not_null<Scalar<DataVector>*> conformal_factor_correction,
        const gsl::not_null<Scalar<DataVector>*>
            lapse_times_conformal_factor_correction,
        const gsl::not_null<tnsr::I<DataVector, 3>*> shift_excess_correction,
        const gsl::not_null<Scalar<DataVector>*>
        /*n_dot_conformal_factor_gradient_correction*/,
        const gsl::not_null<Scalar<DataVector>*>
        /*n_dot_lapse_times_conformal_factor_gradient_correction*/,
        const gsl::not_null<tnsr::I<DataVector, 3>*>
        /*n_dot_longitudinal_shift_excess_correction*/,
        const tnsr::i<DataVector, 3>& /*deriv_conformal_factor_correction*/,
        const tnsr::i<DataVector, 3>&
        /*deriv_lapse_times_conformal_factor_correction*/,
        const tnsr::iJ<DataVector, 3>& /*deriv_shift_excess_correction*/)
        const {
  get(*conformal_factor_correction) = 0.;
  get(*lapse_times_conformal_factor_correction) = 0.;
  std::fill(shift_excess_correction->begin(), shift_excess_correction->end(),
            0.);
}

template <typename IsolatedObjectBase, typename IsolatedObjectClasses>
bool operator==(const SuperposedBoostedBinary<IsolatedObjectBase,
                                              IsolatedObjectClasses>& /*lhs*/,
                const SuperposedBoostedBinary<IsolatedObjectBase,
                                              IsolatedObjectClasses>& /*rhs*/) {
  return true;
}

template <typename IsolatedObjectBase, typename IsolatedObjectClasses>
bool operator!=(const SuperposedBoostedBinary<IsolatedObjectBase,
                                              IsolatedObjectClasses>& lhs,
                const SuperposedBoostedBinary<IsolatedObjectBase,
                                              IsolatedObjectClasses>& rhs) {
  return not(lhs == rhs);
}

template class SuperposedBoostedBinary<
    elliptic::analytic_data::AnalyticSolution,
    Xcts::Solutions::all_analytic_solutions>;

template class SuperposedBoostedBinary<
    elliptic::analytic_data::AnalyticSolution,
    tmpl::list<Xcts::Solutions::Schwarzschild>>;

}  // namespace Xcts::BoundaryConditions
