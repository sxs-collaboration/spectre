// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "ParallelAlgorithms/ApparentHorizonFinder/Criteria/Residual.hpp"

#include <cstddef>
#include <pup.h>

#include "NumericalAlgorithms/SphericalHarmonics/Strahlkorper.hpp"
#include "Options/Context.hpp"
#include "Options/ParseError.hpp"
#include "ParallelAlgorithms/ApparentHorizonFinder/FastFlow.hpp"
#include "Utilities/GenerateInstantiations.hpp"

namespace ah::Criteria {
Residual::Residual(double min_residual, double max_residual,
                   size_t min_resolution_l, size_t max_resolution_l,
                   const Options::Context& context)
    : min_residual_(min_residual),
      max_residual_(max_residual),
      min_resolution_l_(min_resolution_l),
      max_resolution_l_(max_resolution_l) {
  if (min_residual_ >= max_residual_) {
    PARSE_ERROR(context, "MinResidual must be less than MaxResidual");
  }
  if (min_resolution_l_ < 2) {
    PARSE_ERROR(context, "MinResolutionL must not be less than 2");
  }
  if (max_resolution_l_ < 2) {
    PARSE_ERROR(context, "MaxResolutionL must not be less than 2");
  }
  if (min_resolution_l_ > max_resolution_l_) {
    PARSE_ERROR(context,
                "MinResolutionL must not be greater than MaxResolutionL");
  }
}

void Residual::pup(PUP::er& p) {
  p | min_residual_;
  p | max_residual_;
  p | min_resolution_l_;
  p | max_resolution_l_;
}
Residual::Residual(CkMigrateMessage* msg) : Criterion(msg) {}

#ifndef __CUDA_ARCH__
PUP::able::PUP_ID ah::Criteria::Residual::my_PUP_ID = 0;  // NOLINT
#endif                                                    // __CUDA_ARCH__
}  // namespace ah::Criteria
