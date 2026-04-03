// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "ParallelAlgorithms/RayTracer/BackgroundSpacetimes/NumericData.hpp"

#include "DataStructures/Tensor/Tensor.hpp"
#include "IO/Exporter/PointwiseInterpolator.hpp"
#include "IO/Exporter/SpacetimeInterpolator.hpp"
#include "Parallel/Printf/Printf.hpp"
#include "Utilities/TaggedTuple.hpp"

namespace ray_tracing {

NumericData::NumericData(std::string file_glob, std::string subfile_name,
                         std::optional<int> observation_step,
                         ::Verbosity verbosity)
    : file_glob_(std::move(file_glob)),
      subfile_name_(std::move(subfile_name)),
      observation_step_(observation_step),
      verbosity_(verbosity) {
  if (not observation_step_.has_value()) {
    // Construct the spacetime interpolator but don't load any data yet
    interpolator_ = spectre::Exporter::SpacetimeInterpolator<Dim, Frame>{
        file_glob_, subfile_name_,
        spectre::Exporter::get_tensor_components<tags>()};
  }
}

NumericData::NumericData(const NumericData& rhs)
    : NumericData(rhs.file_glob_, rhs.subfile_name_, rhs.observation_step_,
                  rhs.verbosity_) {}

NumericData& NumericData::operator=(const NumericData& rhs) {
  if (this != &rhs) {
    *this = NumericData(rhs);
  }
  return *this;
}

void NumericData::initialize(std::array<double, 2> new_time_bounds) {
  if (verbosity_ >= ::Verbosity::Verbose) {
    Parallel::printf("Loading numeric data...\n");
  }
  if (observation_step_.has_value()) {
    interpolator_ = PointwiseInterpolator{
        file_glob_, subfile_name_,
        spectre::Exporter::ObservationStep{observation_step_.value()},
        spectre::Exporter::get_tensor_components<tags>()};
  } else {
    std::get<SpacetimeInterpolator>(interpolator_)
        .load_time_bounds(new_time_bounds);
  }
  if (verbosity_ >= ::Verbosity::Verbose) {
    Parallel::printf("Numeric data loaded.\n");
  }
}

std::array<double, 2> NumericData::time_bounds() const {
  if (std::holds_alternative<SpacetimeInterpolator>(interpolator_)) {
    return std::get<SpacetimeInterpolator>(interpolator_).time_bounds();
  } else {
    return {{-std::numeric_limits<double>::infinity(),
             std::numeric_limits<double>::infinity()}};
  }
}

tuples::tagged_tuple_from_typelist<typename NumericData::tags>
NumericData::variables(const tnsr::I<DataType, Dim, Frame>& x, const double t,
                       const std::optional<gsl::not_null<std::vector<size_t>*>>
                           block_order) const {
  std::vector<double> result{};
  if (std::holds_alternative<PointwiseInterpolator>(interpolator_)) {
    std::get<PointwiseInterpolator>(interpolator_)
        .interpolate_to_point(make_not_null(&result), x, block_order);
  } else {
    std::get<SpacetimeInterpolator>(interpolator_)
        .interpolate_to_point(make_not_null(&result), x, t, block_order);
  }
  return spectre::Exporter::make_tagged_tuple<tags>(std::move(result));
}

void NumericData::pup(PUP::er& p) {
  BackgroundSpacetime::pup(p);
  p | file_glob_;
  p | subfile_name_;
  p | observation_step_;
  p | verbosity_;
  // Don't copy interpolator, it must be reinitialized
}

bool operator==(const NumericData& lhs, const NumericData& rhs) {
  return lhs.file_glob_ == rhs.file_glob_ and
         lhs.subfile_name_ == rhs.subfile_name_ and
         lhs.observation_step_ == rhs.observation_step_ and
         lhs.verbosity_ == rhs.verbosity_;
}

bool operator!=(const NumericData& lhs, const NumericData& rhs) {
  return not(lhs == rhs);
}

#if defined(SPECTRE_USE_CHARM)
PUP::able::PUP_ID NumericData::my_PUP_ID = 0;  // NOLINT
#endif                                         // SPECTRE_USE_CHARM

}  // namespace ray_tracing
