// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>
#include <optional>
#include <pup.h>
#include <string>

#include "ControlSystem/Protocols/ControlError.hpp"
#include "ControlSystem/Protocols/ControlSystem.hpp"
#include "ControlSystem/Protocols/Measurement.hpp"
#include "DataStructures/DataVector.hpp"
#include "Domain/Structure/ObjectLabel.hpp"
#include "Options/String.hpp"
#include "Parallel/GlobalCache.hpp"
#include "Utilities/GetOutput.hpp"
#include "Utilities/PrettyType.hpp"
#include "Utilities/ProtocolHelpers.hpp"
#include "Utilities/TMPL.hpp"
#include "Utilities/TaggedTuple.hpp"

namespace control_system::TestHelpers {
namespace TestStructs_detail {
struct LabelA {};
}  // namespace TestStructs_detail

template <typename Label>
struct Measurement : tt::ConformsTo<control_system::protocols::Measurement> {
  using submeasurements = tmpl::list<>;
};

template <size_t NumExcisions>
struct ControlError : tt::ConformsTo<control_system::protocols::ControlError> {
  static constexpr size_t expected_number_of_excisions = NumExcisions;
  using object_centers = domain::object_list<>;
  void pup(PUP::er& /*p*/) {}

  using options = tmpl::list<>;
  static constexpr Options::String help{"Example control error."};

  template <typename Metavariables, typename... QueueTags>
  DataVector operator()(
      const Parallel::GlobalCache<Metavariables>& /*cache*/,
      const double /*time*/, const std::string& /*function_of_time_name*/,
      const tuples::TaggedTuple<QueueTags...>& /*measurements*/) {
    return DataVector{};
  }
};

static_assert(tt::assert_conforms_to_v<Measurement<TestStructs_detail::LabelA>,
                                       control_system::protocols::Measurement>);

template <size_t DerivOrder, typename Label, typename Measurement,
          size_t NumExcisions = 0>
struct System : tt::ConformsTo<control_system::protocols::ControlSystem> {
  static std::string name() { return pretty_type::short_name<Label>(); }
  static std::optional<std::string> component_name(
      const size_t i, const size_t /*num_components*/) {
    return get_output(i);
  }
  using measurement = Measurement;
  using simple_tags = tmpl::list<>;
  using control_error = ControlError<NumExcisions>;
  static constexpr size_t deriv_order = DerivOrder;
};

static_assert(
    tt::assert_conforms_to_v<System<2, TestStructs_detail::LabelA,
                                    Measurement<TestStructs_detail::LabelA>>,
                             control_system::protocols::ControlSystem>);
}  // namespace control_system::TestHelpers
