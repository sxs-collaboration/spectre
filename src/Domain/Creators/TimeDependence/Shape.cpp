// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Domain/Creators/TimeDependence/Shape.hpp"

#include <array>
#include <cstddef>
#include <limits>
#include <memory>
#include <optional>
#include <ostream>
#include <string>
#include <unordered_map>
#include <vector>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/IndexType.hpp"
#include "Domain/CoordinateMaps/CoordinateMap.tpp"
#include "Domain/CoordinateMaps/MapInstantiationMacros.hpp"
#include "Domain/CoordinateMaps/TimeDependent/ShapeMapTransitionFunctions/ShapeMapTransitionFunction.hpp"
#include "Domain/CoordinateMaps/TimeDependent/ShapeMapTransitionFunctions/SphereTransition.hpp"
#include "Domain/Creators/TimeDependence/TimeDependence.hpp"
#include "Domain/FunctionsOfTime/FunctionOfTime.hpp"
#include "Domain/FunctionsOfTime/PiecewisePolynomial.hpp"
#include "Domain/Structure/ObjectLabel.hpp"
#include "Options/ParseError.hpp"
#include "PointwiseFunctions/AnalyticSolutions/GeneralRelativity/KerrHorizon.hpp"
#include "Utilities/ErrorHandling/Assert.hpp"
#include "Utilities/GenerateInstantiations.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/StdArrayHelpers.hpp"
#include "Utilities/TMPL.hpp"

namespace domain {
namespace creators::time_dependence {
using SphereTransition =
    domain::CoordinateMaps::ShapeMapTransitionFunctions::SphereTransition;

template <domain::ObjectLabel Label>
Shape<Label>::Shape(const double initial_time, const size_t l_max,
                    const double mass, const std::array<double, 3> spin,
                    const std::array<double, 3> center,
                    const double inner_radius, const double outer_radius,
                    const Options::Context& context)
    : initial_time_(initial_time),
      l_max_(l_max),
      mass_(mass),
      spin_(spin),
      center_(center),
      inner_radius_(inner_radius),
      outer_radius_(outer_radius),
      transition_func_(std::make_unique<SphereTransition>(
          SphereTransition{inner_radius, outer_radius})) {
  using ::operator<<;
  if (mass <= 0.0) {
    PARSE_ERROR(context,
                "Tried to create a Shape TimeDependence, but "
                "the mass ("
                    << mass << ") must be strictly positive.");
  }
  if (magnitude(spin) >= 1.0) {
    PARSE_ERROR(context,
                "Tried to create a Shape TimeDependence, but "
                "the magnitude of the spin ("
                    << spin << ") is greater than one.");
  }
  // There is no PARSE_ERROR for the `inner_radius` < `outer_radius` condition
  // because the SphereTransition already checks for this condition.
}

template <domain::ObjectLabel Label>
std::unique_ptr<TimeDependence<Shape<Label>::mesh_dim>>
Shape<Label>::get_clone() const {
  return std::make_unique<Shape<Label>>(initial_time_, l_max_, mass_, spin_,
                                        center_, inner_radius_, outer_radius_);
}

template <domain::ObjectLabel Label>
std::vector<std::unique_ptr<domain::CoordinateMapBase<
    Frame::Grid, Frame::Inertial, Shape<Label>::mesh_dim>>>
Shape<Label>::block_maps_grid_to_inertial(const size_t number_of_blocks) const {
  ASSERT(number_of_blocks > 0,
         "Must have at least one block on which to create a map.");

  std::vector<std::unique_ptr<
      domain::CoordinateMapBase<Frame::Grid, Frame::Inertial, 3>>>
      result{number_of_blocks};
  result[0] = std::make_unique<GridToInertialMap>(grid_to_inertial_map());
  for (size_t i = 1; i < number_of_blocks; ++i) {
    result[i] = result[0]->get_clone();
  }
  return result;
}

template <domain::ObjectLabel Label>
std::vector<std::unique_ptr<domain::CoordinateMapBase<
    Frame::Grid, Frame::Distorted, Shape<Label>::mesh_dim>>>
Shape<Label>::block_maps_grid_to_distorted(
    const size_t number_of_blocks) const {
  ASSERT(number_of_blocks > 0,
         "Must have at least one block on which to create a map.");

  std::vector<std::unique_ptr<
      domain::CoordinateMapBase<Frame::Grid, Frame::Distorted, 3>>>
      result{number_of_blocks};
  result[0] = std::make_unique<GridToDistortedMap>(grid_to_distorted_map());
  for (size_t i = 1; i < number_of_blocks; ++i) {
    result[i] = result[0]->get_clone();
  }
  return result;
}

template <domain::ObjectLabel Label>
std::vector<std::unique_ptr<domain::CoordinateMapBase<
    Frame::Distorted, Frame::Inertial, Shape<Label>::mesh_dim>>>
Shape<Label>::block_maps_distorted_to_inertial(
    const size_t number_of_blocks) const {
  ASSERT(number_of_blocks > 0,
         "Must have at least one block on which to create a map.");

  std::vector<std::unique_ptr<
      domain::CoordinateMapBase<Frame::Distorted, Frame::Inertial, 3>>>
      result{number_of_blocks};
  result[0] =
      std::make_unique<DistortedToInertialMap>(distorted_to_inertial_map());
  for (size_t i = 1; i < number_of_blocks; ++i) {
    result[i] = result[0]->get_clone();
  }
  return result;
}

template <domain::ObjectLabel Label>
std::unordered_map<std::string,
                   std::unique_ptr<domain::FunctionsOfTime::FunctionOfTime>>
Shape<Label>::functions_of_time(const std::unordered_map<std::string, double>&
                                    initial_expiration_times) const {
  std::unordered_map<std::string,
                     std::unique_ptr<domain::FunctionsOfTime::FunctionOfTime>>
      result{};

  // Functions of time don't expire by default
  double expiration_time = std::numeric_limits<double>::infinity();

  // If we have control systems, overwrite the expiration time with the one
  // supplied by the control system
  if (initial_expiration_times.count(function_of_time_name_) == 1) {
    expiration_time = initial_expiration_times.at(function_of_time_name_);
  }

  const ylm::Spherepack ylm{l_max_, l_max_};
  const DataVector radial_distortion =
      1.0 - get(gr::Solutions::kerr_schild_radius_from_boyer_lindquist(
                inner_radius_, ylm.theta_phi_points(), mass_, spin_)) /
                inner_radius_;
  const auto radial_distortion_coefs = ylm.phys_to_spec(radial_distortion);
  const DataVector zeros =
      make_with_value<DataVector>(radial_distortion_coefs, 0.0);
  result[function_of_time_name_] =
      std::make_unique<FunctionsOfTime::PiecewisePolynomial<3>>(
          initial_time_,
          std::array<DataVector, 4>{
              {radial_distortion_coefs, zeros, zeros, zeros}},
          expiration_time);
  // This adds a size function of time that is the l=0 component of the radial
  // distortion coefficients. We do this so that this time dependence can be
  // used with the Shape control system. The control error for the shape control
  // system requires there to be two functions of time in the cache: one for
  // shape and one for size. This size function of time isn't controlling any
  // maps so it never expires and is constant the entire time. If we want this
  // time dependence to work with size control as well, we'll have to add in the
  // size map as well.
  const DataVector zeros_size{1, 0.0};
  const std::string size_name = "Size"s + get_output(Label);
  result[size_name] = std::make_unique<FunctionsOfTime::PiecewisePolynomial<3>>(
      initial_time_,
      std::array<DataVector, 4>{
          {{radial_distortion_coefs[0]}, zeros_size, zeros_size, zeros_size}},
      std::numeric_limits<double>::infinity());
  return result;
}

template <domain::ObjectLabel Label>
auto Shape<Label>::grid_to_inertial_map() const -> GridToInertialMap {
  return GridToInertialMap{ShapeMap{center_, l_max_, l_max_,
                                    transition_func_->get_clone(),
                                    function_of_time_name_}};
}

template <domain::ObjectLabel Label>
auto Shape<Label>::grid_to_distorted_map() const -> GridToDistortedMap {
  return GridToDistortedMap{ShapeMap{center_, l_max_, l_max_,
                                     transition_func_->get_clone(),
                                     function_of_time_name_}};
}

template <domain::ObjectLabel Label>
auto Shape<Label>::distorted_to_inertial_map() -> DistortedToInertialMap {
  return DistortedToInertialMap{Identity{}};
}

template <domain::ObjectLabel Label>
bool operator==(const Shape<Label>& lhs, const Shape<Label>& rhs) {
  return lhs.initial_time_ == rhs.initial_time_ and lhs.l_max_ == rhs.l_max_ and
         lhs.mass_ == rhs.mass_ and lhs.spin_ == rhs.spin_ and
         lhs.center_ == rhs.center_ and
         lhs.inner_radius_ == rhs.inner_radius_ and
         lhs.outer_radius_ == rhs.outer_radius_;
}

template <domain::ObjectLabel Label>
bool operator!=(const Shape<Label>& lhs, const Shape<Label>& rhs) {
  return not(lhs == rhs);
}

#define LABEL(data) BOOST_PP_TUPLE_ELEM(0, data)

#define INSTANTIATION(_, data)                                             \
  template class Shape<LABEL(data)>;                                       \
  template bool operator==                                                 \
      <LABEL(data)>(const Shape<LABEL(data)>&, const Shape<LABEL(data)>&); \
  template bool operator!=                                                 \
      <LABEL(data)>(const Shape<LABEL(data)>&, const Shape<LABEL(data)>&);

GENERATE_INSTANTIATIONS(INSTANTIATION,
                        (domain::ObjectLabel::A, domain::ObjectLabel::B,
                         domain::ObjectLabel::None))

#undef LABEL
}  // namespace creators::time_dependence

using ShapeMap3d = CoordinateMaps::TimeDependent::Shape;

INSTANTIATE_MAPS_FUNCTIONS(((ShapeMap3d)), (Frame::Grid),
                           (Frame::Distorted, Frame::Inertial),
                           (double, DataVector))
INSTANTIATE_MAPS_FUNCTIONS(((ShapeMap3d)), (Frame::Distorted),
                           (Frame::Inertial), (double, DataVector))

}  // namespace domain
