// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include "DataStructures/DataBox/DataBox.hpp"
#include "DataStructures/DataBox/Tag.hpp"
#include "DataStructures/LinkedMessageId.hpp"
#include "DataStructures/Tensor/IndexType.hpp"
#include "Domain/FunctionsOfTime/FunctionOfTime.hpp"
#include "Domain/FunctionsOfTime/Tags.hpp"
#include "IO/Logging/Verbosity.hpp"
#include "Parallel/GlobalCache.hpp"
#include "ParallelAlgorithms/ApparentHorizonFinder/Destination.hpp"
#include "ParallelAlgorithms/ApparentHorizonFinder/FastFlow.hpp"
#include "ParallelAlgorithms/ApparentHorizonFinder/Protocols/Callback.hpp"
#include "ParallelAlgorithms/ApparentHorizonFinder/Protocols/HorizonMetavars.hpp"
#include "ParallelAlgorithms/ApparentHorizonFinder/Tags.hpp"
#include "Utilities/ProtocolHelpers.hpp"

namespace TestHelpers::ah {
struct ExampleTimeTag : db::SimpleTag {
  using type = LinkedMessageId<double>;
};

struct ExampleTimeTagCompute : ExampleTimeTag, db::ComputeTag {
  using base = ExampleTimeTag;
  using return_type = LinkedMessageId<double>;
  using argument_tags = tmpl::list<>;
  static void function(const gsl::not_null<LinkedMessageId<double>*> result) {
    *result = LinkedMessageId<double>{1.0, std::nullopt};
  }
};

/// [HorizonFindCallback]
struct ExampleHorizonFindCallback : tt::ConformsTo<::ah::protocols::Callback> {
  template <typename DbTags, typename Metavariables>
  static void apply(db::DataBox<DbTags>& box,
                    const Parallel::GlobalCache<Metavariables>& cache,
                    const FastFlow::Status status) {
    const ::Verbosity verbosity = db::get<::ah::Tags::Verbosity>(box);
    const auto& functions_of_time =
        Parallel::get<domain::Tags::FunctionsOfTime>(cache);

    // Use these to run control system callbacks, observe quantities, or error
    // if the horizon find failed.
    (void)status;
    (void)verbosity;
    (void)functions_of_time;
  }
};
/// [HorizonFindCallback]

/// [HorizonMetavars]
struct ExampleHorizonMetavars
    : tt::ConformsTo<::ah::protocols::HorizonMetavars> {
  using time_tag = ExampleTimeTag;

  using frame = ::Frame::Distorted;

  using horizon_find_callbacks = tmpl::list<ExampleHorizonFindCallback>;
  using horizon_find_failure_callbacks = tmpl::list<>;

  using compute_tags_on_element = tmpl::list<ExampleTimeTagCompute>;

  static constexpr ::ah::Destination destination =
      ::ah::Destination::ControlSystem;

  static std::string name() { return "ExampleHorizon"; }
};
/// [HorizonMetavars]
}  // namespace TestHelpers::ah
