// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "ParallelAlgorithms/ApparentHorizonFinder/Criteria/Shape.hpp"

#include <cstddef>
#include <pup.h>

#include "Options/Context.hpp"
#include "Options/ParseError.hpp"

namespace ah::Criteria {
Shape::Shape(double min_truncation_error, double max_truncation_error,
             size_t max_pile_up_modes, size_t min_resolution_l,
             const Options::Context& context)
    : min_truncation_error_(min_truncation_error),
      max_truncation_error_(max_truncation_error),
      max_pile_up_modes_(max_pile_up_modes),
      min_resolution_l_(min_resolution_l) {
  if (min_truncation_error_ >= max_truncation_error_) {
    PARSE_ERROR(context,
                "MinTruncationError must be less than MaxTruncationError");
  }
  if (min_resolution_l_ < MinResolutionL::lower_bound()) {
    PARSE_ERROR(context, "MinResolutionL must be at least "
                             << MinResolutionL::lower_bound());
  }
}

void Shape::pup(PUP::er& p) {
  p | min_truncation_error_;
  p | max_truncation_error_;
  p | max_pile_up_modes_;
  p | min_resolution_l_;
}

bool Shape::is_equal(const Criterion& other) const {
  const auto* other_shape = dynamic_cast<const Shape*>(&other);
  if (other_shape == nullptr) {
    return false;
  }
  return min_truncation_error_ == other_shape->min_truncation_error_ and
         max_truncation_error_ == other_shape->max_truncation_error_ and
         max_pile_up_modes_ == other_shape->max_pile_up_modes_ and
         min_resolution_l_ == other_shape->min_resolution_l_;
}

Shape::Shape(CkMigrateMessage* msg) : Criterion(msg) {}

PUP::able::PUP_ID ah::Criteria::Shape::my_PUP_ID = 0;  // NOLINT
}  // namespace ah::Criteria
