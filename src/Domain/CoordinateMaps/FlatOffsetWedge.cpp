// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Domain/CoordinateMaps/FlatOffsetWedge.hpp"

#include <cmath>
#include <limits>
#include <optional>
#include <pup.h>
#include <sstream>
#include <utility>

#include "DataStructures/Tensor/EagerMath/DeterminantAndInverse.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Utilities/ConstantExpressions.hpp"
#include "Utilities/DereferenceWrapper.hpp"
#include "Utilities/EqualWithinRoundoff.hpp"
#include "Utilities/GenerateInstantiations.hpp"
#include "Utilities/Serialization/PupStlCpp11.hpp"

namespace domain::CoordinateMaps {

FlatOffsetWedge::FlatOffsetWedge(double lower_face_y_half_width,
                                 double lower_face_x_width, double outer_radius)
    : lower_face_y_half_width_(lower_face_y_half_width),
      lower_face_x_width_(lower_face_x_width),
      outer_radius_(outer_radius) {
  // The equal_within_roundoffs below have an implicit scale of 1,
  // so the ASSERTs may trigger in the case where we really
  // want an entire domain that is very small.
  ASSERT(not equal_within_roundoff(lower_face_y_half_width, 0.0),
         "Cannot have zero lower_face_y_half_width");
  ASSERT(lower_face_y_half_width > 0.0,
         "Cannot have negative lower_face_y_half_width");
  ASSERT(not equal_within_roundoff(lower_face_x_width, 0.0),
         "Cannot have zero lower_face_x_width");
  ASSERT(lower_face_x_width > 0.0, "Cannot have negative lower_face_x_width");
  ASSERT(not equal_within_roundoff(outer_radius, 0.0),
         "Cannot have zero outer_radius");
  ASSERT(outer_radius > 0.0, "Cannot have negative outer_radius");

  // The following ASSERT (and the ones above) are the strict
  // requirements for the map to be nonsingular.
  // Below we will have further restrictions that prevent the
  // map from going nearly singular and losing accuracy.
  ASSERT(square(outer_radius) - square(lower_face_x_width) >
             2.0 * square(lower_face_y_half_width),
         "Must have R^2-D^2 > 2 L^2 or else map is singular.  "
         "Here R = "
             << outer_radius << ", D = " << lower_face_x_width
             << ", L = " << lower_face_y_half_width);

  // Here we arbitrarily restrict the parameters of the map to make
  // our lives easier. The idea is that we don't want a map that is
  // epsilon away from being singular, since then the map will have
  // very large Jacobians (even though it is technically nonsingular)
  // and this may cause numerical problems.  In the unit tests, we
  // will stick to maps that have parameters that obey the
  // restrictions below.
  //
  // The magic number epsilon here is chosen arbitrarily,
  // but based on what we think a sensible user would want.
  const double epsilon = 0.1;
  // However, there is a restriction on epsilon.
  // max(lower_face_y_half_width) - min(lower_face_y_half_with) must
  // be positive, or else there are no possible values of
  // lower_face_y_half_width.
  // With the choices below, the smallest possible value of
  // max(lower_face_y_half_width) - min(lower_face_y_half_with) turns out
  // to be outer_radius*epsilon*(2-7*epsilon+O(epsilon)^2).  So we should
  // choose epsilon < 2/7.
  //
  // Note that it is possible to choose multiple magic numbers,
  // e.g. one value of epsilon for restrictions on lower_face_x_width
  // and another value of epsilon for restrictions on
  // lower_face_y_half_width, or even different magic numbers for the
  // minimum and maximum values of each parameter. If we need to do
  // such a thing later we can do so, but for now keep only one value
  // of epsilon for simplicity.
  ASSERT(lower_face_x_width >= epsilon * outer_radius and
             lower_face_x_width <= (1 - epsilon) * outer_radius,
         "The map is not tested if lower_face_x_width < epsilon*outer_radius "
         "or if lower_face_x_width > (1-epsilon)*outer_radius. Here epsilon="
             << epsilon << ",lower_face_x_width=" << lower_face_x_width
             << ", outer_radius=" << outer_radius);
  ASSERT(lower_face_y_half_width >= epsilon * outer_radius,
         "The map is not tested if lower_face_y_half_width < "
         "epsilon*outer_radius. Here epsilon = "
             << epsilon << ",lower_face_y_half_width="
             << lower_face_y_half_width << ", outer_radius=" << outer_radius);
  ASSERT((square(outer_radius) - square(lower_face_x_width)) *
                 square(1 - epsilon) >=
             2.0 * square(lower_face_y_half_width),
         "The map is not tested if 2L^2 > (1-epsilon)^2(R^2-D^2). Here R = "
             << outer_radius << ", D = " << lower_face_x_width << ", L = "
             << lower_face_y_half_width << ",epsilon = " << epsilon);
}

template <typename T>
std::array<tt::remove_cvref_wrap_t<T>, 3> FlatOffsetWedge::operator()(
    const std::array<T, 3>& source_coords) const {
  using ReturnType = tt::remove_cvref_wrap_t<T>;
  const ReturnType& xi = source_coords[0];
  const ReturnType& eta = source_coords[1];
  const ReturnType& zeta = source_coords[2];
  std::array<ReturnType, 3> target_coords{};
  ReturnType& x = target_coords[0];
  ReturnType& y = target_coords[1];
  ReturnType& z = target_coords[2];

  const double q = 0.5 * lower_face_x_width_ / outer_radius_;

  // Use x as temporary storage so we avoid memory allocations.
  // x is set to P here (where P is the quantity in the dox).
  x = outer_radius_ *
      sqrt((1.0 - square(q * (xi - 1.0))) / (1.0 + square(eta)));

  // Now fill the coordinates using x as temporary.
  z = 0.5 *
      (x + lower_face_y_half_width_ + zeta * (x - lower_face_y_half_width_));
  y = eta * z;
  x = (0.5 * lower_face_x_width_) * (xi + 1.0);

  return target_coords;
}

std::optional<std::array<double, 3>> FlatOffsetWedge::inverse(
    const std::array<double, 3>& target_coords) const {
  const double& x = target_coords[0];
  const double& y = target_coords[1];
  const double& z = target_coords[2];

  const double xi = 2.0 * x / lower_face_x_width_ - 1.0;

  // Check for point out of range.
  // Allow out of range by roundoff.
  const double abs_xi = std::abs(xi);
  if (abs_xi > 1.0 and not equal_within_roundoff(abs_xi, 1.0)) {
    return std::nullopt;
  }

  // If z is zero, we are out of range (even if y is zero).
  if (z == 0.0) {
    return std::nullopt;
  }
  const double eta = y / z;

  // Check for point out of range.
  // Allow out of range by roundoff.
  const double abs_eta = std::abs(eta);
  if (abs_eta > 1.0 and not equal_within_roundoff(abs_eta, 1.0)) {
    return std::nullopt;
  }

  // Since we know that xi and eta are in range, the following sqrts
  // will always have positive arguments.
  const double P =
      outer_radius_ *
      sqrt((1.0 - square((x - lower_face_x_width_) / outer_radius_)) /
           (1.0 + square(eta)));
  const double zeta =
      (2.0 * z - P - lower_face_y_half_width_) / (P - lower_face_y_half_width_);

  // Check for point out of range.
  // Allow out of range by roundoff.
  const double abs_zeta = std::abs(zeta);
  if (abs_zeta > 1.0 and not equal_within_roundoff(abs_zeta, 1.0)) {
    return std::nullopt;
  }

  return {{xi, eta, zeta}};
}

template <typename T>
tnsr::Ij<tt::remove_cvref_wrap_t<T>, 3, Frame::NoFrame>
FlatOffsetWedge::jacobian(const std::array<T, 3>& source_coords) const {
  using ReturnType = tt::remove_cvref_wrap_t<T>;
  const ReturnType& xi = source_coords[0];
  const ReturnType& eta = source_coords[1];
  const ReturnType& zeta = source_coords[2];

  auto jac =
      make_with_value<tnsr::Ij<tt::remove_cvref_wrap_t<T>, 3, Frame::NoFrame>>(
          dereference_wrapper(source_coords[0]), 0.0);

  const double q = 0.5 * lower_face_x_width_ / outer_radius_;

  // Use Jacobian components as temporary storage to avoid extra
  // memory allocations.

  // temporarily jac(0,0) = P (where P is the quantity in the dox)
  get<0, 0>(jac) = outer_radius_ *
                   sqrt((1.0 - square(q * (xi - 1.0))) / (1.0 + square(eta)));

  // temporarily jac(1,1) = z
  get<1, 1>(jac) = 0.5 * (get<0, 0>(jac) + lower_face_y_half_width_ +
                          zeta * (get<0, 0>(jac) - lower_face_y_half_width_));

  // Fill in correct jac(2,1)
  get<2, 1>(jac) =
      -get<0, 0>(jac) * 0.5 * eta * (1.0 + zeta) / (1.0 + square(eta));
  // Now use that to get correct jac(1,1)
  get<1, 1>(jac) += eta * get<2, 1>(jac);

  // Fill in correct jac(2,0) and jac(1,0)
  get<2, 0>(jac) = 0.5 * get<0, 0>(jac) * square(q) * (1.0 + zeta) *
                   (1.0 - xi) / (1.0 - square(q * (xi - 1.0)));
  get<1, 0>(jac) = eta * get<2, 0>(jac);

  // Fill in correct jac(2,2) and jac(1,2)
  get<2, 2>(jac) = 0.5 * (get<0, 0>(jac) - lower_face_y_half_width_);
  get<1, 2>(jac) = eta * get<2, 2>(jac);

  // Now set jac(0,0) to its real value instead of the temporary.
  get<0, 0>(jac) = 0.5 * lower_face_x_width_;

  return jac;
}

template <typename T>
tnsr::Ij<tt::remove_cvref_wrap_t<T>, 3, Frame::NoFrame>
FlatOffsetWedge::inv_jacobian(const std::array<T, 3>& source_coords) const {
  return determinant_and_inverse(jacobian(source_coords)).second;
}

void FlatOffsetWedge::pup(PUP::er& p) {
  size_t version = 0;
  p | version;
  // Remember to increment the version number when making changes to this
  // function. Retain support for unpacking data written by previous versions
  // whenever possible. See `Domain` docs for details.
  if (version >= 0) {
    p | lower_face_y_half_width_;
    p | lower_face_x_width_;
    p | outer_radius_;
  }
}

bool operator==(const FlatOffsetWedge& lhs, const FlatOffsetWedge& rhs) {
  return lhs.lower_face_y_half_width_ == rhs.lower_face_y_half_width_ and
         lhs.lower_face_x_width_ == rhs.lower_face_x_width_ and
         lhs.outer_radius_ == rhs.outer_radius_;
}

bool operator!=(const FlatOffsetWedge& lhs, const FlatOffsetWedge& rhs) {
  return not(lhs == rhs);
}

// Explicit instantiations
#define DTYPE(data) BOOST_PP_TUPLE_ELEM(0, data)

#define INSTANTIATE(_, data)                                                   \
  template std::array<tt::remove_cvref_wrap_t<DTYPE(data)>, 3>                 \
  FlatOffsetWedge::operator()(const std::array<DTYPE(data), 3>& source_coords) \
      const;                                                                   \
  template tnsr::Ij<tt::remove_cvref_wrap_t<DTYPE(data)>, 3, Frame::NoFrame>   \
  FlatOffsetWedge::jacobian(const std::array<DTYPE(data), 3>& source_coords)   \
      const;                                                                   \
  template tnsr::Ij<tt::remove_cvref_wrap_t<DTYPE(data)>, 3, Frame::NoFrame>   \
  FlatOffsetWedge::inv_jacobian(                                               \
      const std::array<DTYPE(data), 3>& source_coords) const;

GENERATE_INSTANTIATIONS(INSTANTIATE, (double, DataVector,
                                      std::reference_wrapper<const double>,
                                      std::reference_wrapper<const DataVector>))

#undef DTYPE
#undef INSTANTIATE

}  // namespace domain::CoordinateMaps
