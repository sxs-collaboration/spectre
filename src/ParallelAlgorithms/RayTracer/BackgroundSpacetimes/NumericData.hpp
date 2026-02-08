// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <array>
#include <cstddef>
#include <memory>
#include <optional>
#include <pup.h>
#include <string>
#include <variant>
#include <vector>

#include "DataStructures/Tensor/TypeAliases.hpp"
#include "IO/Exporter/PointwiseInterpolator.hpp"
#include "IO/Exporter/SpacetimeInterpolator.hpp"
#include "IO/Logging/Verbosity.hpp"
#include "Options/Auto.hpp"
#include "Options/String.hpp"
#include "ParallelAlgorithms/RayTracer/BackgroundSpacetimes/BackgroundSpacetime.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/Serialization/CharmPupable.hpp"
#include "Utilities/TMPL.hpp"
#include "Utilities/TaggedTuple.hpp"

namespace ray_tracing {

/// Numeric data from volume data files.
class NumericData : public BackgroundSpacetime SPECTRE_FINDUS_DERIVED(
                        NumericData, BackgroundSpacetime) {
 public:
  NumericData() = default;
  NumericData(const NumericData& /*rhs*/);
  NumericData& operator=(const NumericData& /*rhs*/);
  NumericData(NumericData&& /*rhs*/) = default;
  NumericData& operator=(NumericData&& /*rhs*/) = default;
  ~NumericData() override = default;

  static constexpr Options::String help = "Numeric data from volume data files";

  struct FileGlob {
    using type = std::string;
    static constexpr Options::String help =
        "Volume data files. Can be a glob pattern.";
  };

  struct SubfileName {
    using type = std::string;
    static constexpr Options::String help = "Subfile name in the volume files";
  };

  struct ObservationStep {
    using type = Options::Auto<int>;
    static constexpr Options::String help =
        "Either a single observation step to load, or 'Auto' to load the full "
        "spacetime. "
        "When a single observation is loaded, then rays are traced "
        "only through that single time slice (fast light approximation). "
        "When 'Auto' is selected, then all time slices are loaded that cover "
        "the required time range and the data is interpolated in both space "
        "and time (slow light).";
  };

  struct Verbosity {
    using type = ::Verbosity;
    static constexpr Options::String help = "Verbosity of output.";
  };

  using options = tmpl::list<FileGlob, SubfileName, ObservationStep, Verbosity>;

  NumericData(std::string file_glob, std::string subfile_name,
              std::optional<int> observation_step,
              ::Verbosity verbosity = ::Verbosity::Silent);

  auto get_clone() const -> std::unique_ptr<BackgroundSpacetime> override {
    return std::make_unique<NumericData>(*this);
  }

  /// \cond
  WRAPPED_PUPable_decl_template(NumericData);
  /// \endcond

  void initialize(std::array<double, 2> new_time_bounds) override;

  std::array<double, 2> time_bounds() const override;

  tuples::tagged_tuple_from_typelist<tags> variables(
      const tnsr::I<DataType, Dim, Frame>& x, double t,
      std::optional<gsl::not_null<std::vector<size_t>*>> block_order =
          std::nullopt) const override;

  // NOLINTNEXTLINE(google-runtime-references)
  void pup(PUP::er& p) override;

  friend bool operator==(const NumericData& lhs, const NumericData& rhs);

 private:
  using PointwiseInterpolator =
      spectre::Exporter::PointwiseInterpolator<Dim, Frame>;
  using SpacetimeInterpolator =
      spectre::Exporter::SpacetimeInterpolator<Dim, Frame>;

  std::string file_glob_;
  std::string subfile_name_;
  std::optional<int> observation_step_;
  ::Verbosity verbosity_ = ::Verbosity::Silent;
  // Cache that holds tensor data in memory. This isn't copied and must be
  // reinitialized after a copy.
  std::variant<PointwiseInterpolator, SpacetimeInterpolator> interpolator_;
};

bool operator!=(const NumericData& lhs, const NumericData& rhs);

}  // namespace ray_tracing
