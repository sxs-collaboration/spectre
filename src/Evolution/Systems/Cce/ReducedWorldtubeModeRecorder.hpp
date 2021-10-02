// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cmath>
#include <cstddef>
#include <string>

#include "Evolution/Systems/Cce/Tags.hpp"
#include "IO/H5/File.hpp"
#include "Utilities/ForceInline.hpp"

/// \cond
class ComplexModalVector;
/// \endcond

namespace Cce {

/// The dataset string associated with each scalar that will be output in the
/// reduced set of SpEC-like worldtube boundary data.
template <typename Tag>
std::string dataset_label_for_tag();

template <>
SPECTRE_ALWAYS_INLINE std::string
dataset_label_for_tag<Cce::Tags::BoundaryValue<Cce::Tags::BondiBeta>>() {
  return "Beta";
}

template <>
SPECTRE_ALWAYS_INLINE std::string
dataset_label_for_tag<Cce::Tags::BoundaryValue<Cce::Tags::BondiU>>() {
  return "U";
}

template <>
SPECTRE_ALWAYS_INLINE std::string
dataset_label_for_tag<Cce::Tags::BoundaryValue<Cce::Tags::BondiQ>>() {
  return "Q";
}

template <>
SPECTRE_ALWAYS_INLINE std::string
dataset_label_for_tag<Cce::Tags::BoundaryValue<Cce::Tags::BondiW>>() {
  return "W";
}

template <>
SPECTRE_ALWAYS_INLINE std::string
dataset_label_for_tag<Cce::Tags::BoundaryValue<Cce::Tags::BondiJ>>() {
  return "J";
}

template <>
SPECTRE_ALWAYS_INLINE std::string dataset_label_for_tag<
    Cce::Tags::BoundaryValue<Cce::Tags::Dr<Cce::Tags::BondiJ>>>() {
  return "DrJ";
}

template <>
SPECTRE_ALWAYS_INLINE std::string dataset_label_for_tag<
    Cce::Tags::BoundaryValue<Cce::Tags::Du<Cce::Tags::BondiJ>>>() {
  return "H";
}

template <>
SPECTRE_ALWAYS_INLINE std::string
dataset_label_for_tag<Cce::Tags::BoundaryValue<Cce::Tags::BondiR>>() {
  return "R";
}

template <>
SPECTRE_ALWAYS_INLINE std::string dataset_label_for_tag<
    Cce::Tags::BoundaryValue<Cce::Tags::Du<Cce::Tags::BondiR>>>() {
  return "DuR";
}

/// Records a compressed representation of SpEC-like worldtube data associated
/// with just the spin-weighted scalars required to perform the CCE algorithm.
struct ReducedWorldtubeModeRecorder {
 public:
  /// The constructor takes the filename used to create the H5File object for
  /// writing the data
  explicit ReducedWorldtubeModeRecorder(const std::string& filename)
      : output_file_{filename} {}

  /// append to `dataset_path` the vector created by `time` followed by the
  /// `modes` rearranged in ascending m-varies-fastest format.
  ///
  /// For real quantities, negative m and the imaginary part of m=0 are omitted.
  void append_worldtube_mode_data(const std::string& dataset_path, double time,
                                  const ComplexModalVector& modes, size_t l_max,
                                  bool is_real = false);

 private:
  h5::H5File<h5::AccessType::ReadWrite> output_file_;
};
}  // namespace Cce
