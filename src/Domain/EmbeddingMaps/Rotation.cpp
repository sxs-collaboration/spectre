// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Domain/EmbeddingMaps/Rotation.hpp"

namespace EmbeddingMaps {

Rotation<2>::Rotation(const double rotation_angle)
    : rotation_angle_(rotation_angle),
      rotation_matrix_(std::numeric_limits<double>::signaling_NaN()) {
  const double cos_alpha = cos(rotation_angle_);
  const double sin_alpha = sin(rotation_angle_);
  rotation_matrix_.get(0, 0) = cos_alpha;
  rotation_matrix_.get(0, 1) = -sin_alpha;
  rotation_matrix_.get(1, 0) = sin_alpha;
  rotation_matrix_.get(1, 1) = cos_alpha;
}

std::unique_ptr<EmbeddingMap<2, 2>> Rotation<2>::get_clone() const {
  return std::make_unique<Rotation<2>>(rotation_angle_);
}

Point<2, Frame::Grid> Rotation<2>::operator()(
    const Point<2, Frame::Logical>& xi) const {
  return Point<2, Frame::Grid>{
      {{xi[0] * rotation_matrix_.template get<0, 0>() +
            xi[1] * rotation_matrix_.template get<0, 1>(),
        xi[0] * rotation_matrix_.template get<1, 0>() +
            xi[1] * rotation_matrix_.template get<1, 1>()}}};
}

Point<2, Frame::Logical> Rotation<2>::inverse(
    const Point<2, Frame::Grid>& x) const {
  // Inverse rotation matrix is the same as the transpose.
  return Point<2, Frame::Logical>{
      {{x[0] * rotation_matrix_.template get<0, 0>() +
            x[1] * rotation_matrix_.template get<1, 0>(),
        x[0] * rotation_matrix_.template get<0, 1>() +
            x[1] * rotation_matrix_.template get<1, 1>()}}};
}

double Rotation<2>::jacobian(const Point<2, Frame::Logical>& /*xi*/, size_t ud,
                             size_t ld) const {
  ASSERT(0 == ld || 1 == ld,
         "ld = " << ld << " in jacobian. ld must be 0 or 1");
  ASSERT(0 == ud || 1 == ud,
         "ud = " << ud << " in jacobian. ud must be 0 or 1");
  return rotation_matrix_.get(ud, ld);
}

double Rotation<2>::inv_jacobian(const Point<2, Frame::Logical>& /*xi*/,
                                 size_t ud, size_t ld) const {
  ASSERT(0 == ld || 1 == ld,
         "ld = " << ld << " in inverse jacobian. ld must be 0 or 1");
  ASSERT(0 == ud || 1 == ud,
         "ud = " << ud << " in inverse jacobian. us must be 0 or 1");
  // Recall inverse(rotation_matrix_) = transpose(rotation_matrix_)
  return rotation_matrix_.get(ld, ud);
}

Rotation<2>::Rotation(CkMigrateMessage* /* m */)
    : rotation_angle_(std::numeric_limits<double>::signaling_NaN()),
      rotation_matrix_() {}

void Rotation<2>::pup(PUP::er& p) {
  EmbeddingMap<2, 2>::pup(p);
  p | rotation_angle_;
  p | rotation_matrix_;
}

Rotation<3>::Rotation(const double rotation_about_z,
                      const double rotation_about_rotated_y,
                      const double rotation_about_rotated_z)
    : rotation_about_z_(rotation_about_z),
      rotation_about_rotated_y_(rotation_about_rotated_y),
      rotation_about_rotated_z_(rotation_about_rotated_z),
      rotation_matrix_(std::numeric_limits<double>::signaling_NaN()) {
  const double cos_alpha = cos(rotation_about_z_);
  const double sin_alpha = sin(rotation_about_z_);
  const double cos_beta = cos(rotation_about_rotated_y_);
  const double sin_beta = sin(rotation_about_rotated_y_);
  const double cos_gamma = cos(rotation_about_rotated_z_);
  const double sin_gamma = sin(rotation_about_rotated_z_);
  rotation_matrix_.get(0, 0) =
      cos_gamma * cos_beta * cos_alpha - sin_gamma * sin_alpha;
  rotation_matrix_.get(0, 1) =
      -sin_gamma * cos_beta * cos_alpha - cos_gamma * sin_alpha;
  rotation_matrix_.get(0, 2) = sin_beta * cos_alpha;
  rotation_matrix_.get(1, 0) =
      cos_gamma * cos_beta * sin_alpha + sin_gamma * cos_alpha;
  rotation_matrix_.get(1, 1) =
      -sin_gamma * cos_beta * sin_alpha + cos_gamma * cos_alpha;
  rotation_matrix_.get(1, 2) = sin_beta * sin_alpha;
  rotation_matrix_.get(2, 0) = -cos_gamma * sin_beta;
  rotation_matrix_.get(2, 1) = sin_gamma * sin_beta;
  rotation_matrix_.get(2, 2) = cos_beta;
}

std::unique_ptr<EmbeddingMap<3, 3>> Rotation<3>::get_clone() const {
  return std::make_unique<Rotation<3>>(
      rotation_about_z_, rotation_about_rotated_y_, rotation_about_rotated_z_);
}

Point<3, Frame::Grid> Rotation<3>::operator()(
    const Point<3, Frame::Logical>& xi) const {
  return Point<3, Frame::Grid>{
      {{xi[0] * rotation_matrix_.template get<0, 0>() +
            xi[1] * rotation_matrix_.template get<0, 1>() +
            xi[2] * rotation_matrix_.template get<0, 2>(),
        xi[0] * rotation_matrix_.template get<1, 0>() +
            xi[1] * rotation_matrix_.template get<1, 1>() +
            xi[2] * rotation_matrix_.template get<1, 2>(),
        xi[0] * rotation_matrix_.template get<2, 0>() +
            xi[1] * rotation_matrix_.template get<2, 1>() +
            xi[2] * rotation_matrix_.template get<2, 2>()}}};
}

Point<3, Frame::Logical> Rotation<3>::inverse(
    const Point<3, Frame::Grid>& x) const {
  // Inverse rotation matrix is the same as the transpose.
  return Point<3, Frame::Logical>{
      {{x[0] * rotation_matrix_.template get<0, 0>() +
            x[1] * rotation_matrix_.template get<1, 0>() +
            x[2] * rotation_matrix_.template get<2, 0>(),
        x[0] * rotation_matrix_.template get<0, 1>() +
            x[1] * rotation_matrix_.template get<1, 1>() +
            x[2] * rotation_matrix_.template get<2, 1>(),
        x[0] * rotation_matrix_.template get<0, 2>() +
            x[1] * rotation_matrix_.template get<1, 2>() +
            x[2] * rotation_matrix_.template get<2, 2>()}}};
}

double Rotation<3>::jacobian(const Point<3, Frame::Logical>& /*xi*/, size_t ud,
                             size_t ld) const {
  ASSERT(0 == ld || 1 == ld || 2 == ld,
         "ld = " << ld << "in jacobian. ld must be 0, 1, or 2");
  ASSERT(0 == ud || 1 == ud || 2 == ud,
         "ud = " << ud << "in jacobian. ud must be 0, 1, or 2");
  return rotation_matrix_.get(ud, ld);
}

double Rotation<3>::inv_jacobian(const Point<3, Frame::Logical>& /*xi*/,
                                 size_t ud, size_t ld) const {
  ASSERT(0 == ld || 1 == ld || 2 == ld,
         "ld = " << ld << "in inv_jacobian. ld must be 0, 1, or 2");
  ASSERT(0 == ud || 1 == ud || 2 == ud,
         "ud = " << ud << "in inv_jacobian. ud must be 0, 1, or 2");
  // Recall inverse(rotation_matrix_) = transpose(rotation_matrix_)
  return rotation_matrix_.get(ld, ud);
}

Rotation<3>::Rotation(CkMigrateMessage* /* m */)
    : rotation_about_z_(std::numeric_limits<double>::signaling_NaN()),
      rotation_about_rotated_y_(std::numeric_limits<double>::signaling_NaN()),
      rotation_about_rotated_z_(std::numeric_limits<double>::signaling_NaN()),
      rotation_matrix_() {}

void Rotation<3>::pup(PUP::er& p) {  // NOLINT
  EmbeddingMap<3, 3>::pup(p);
  p | rotation_about_z_;
  p | rotation_about_rotated_y_;
  p | rotation_about_rotated_z_;
  p | rotation_matrix_;
}
}  // namespace EmbeddingMaps

/// \cond HIDDEN_SYMBOLS
PUP::able::PUP_ID EmbeddingMaps::Rotation<2>::my_PUP_ID = 0;  // NOLINT

PUP::able::PUP_ID EmbeddingMaps::Rotation<3>::my_PUP_ID = 0;  // NOLINT
/// \endcond
