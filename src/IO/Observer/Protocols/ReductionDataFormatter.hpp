// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

/// Protocols related to observers
namespace observers::protocols {
/*!
 * \brief Conforming classes can create an informative message from reduction
 * data. The message will be printed to screen when the formatter is passed to a
 * reduction action such as `observers::Actions::ContributeReductionData`.
 *
 * Conforming classes must provide the following type aliases:
 * - `reduction_data`: The `Parallel::ReductionData` that the formatter is
 *   applicable to.
 *
 * Conforming classes must implement the following member functions:
 * - `operator()`: A call operator that takes the values of the reduction data
 *   and returns a `std::string`. The string should be an informative message
 *   such as "Global minimum grid spacing at time 5.4 is: 0.1".
 * - `pup`: A PUP function for serialization.
 *
 * Here's an example for a formatter:
 *
 * \snippet Test_ReductionObserver.cpp formatter_example
 */
struct ReductionDataFormatter {
  template <typename ConformingType>
  struct test {
    using reduction_data = typename ConformingType::reduction_data;
  };
};
}  // namespace observers::protocols
