// ObjectSfM - Object Based Structure-from-Motion.
// Copyright (C) 2018  Ohio State University, CEGE, GDA group
//
// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
//
// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public License for more details.
//
// You should have received a copy of the GNU General Public License
// along with this program.  If not, see <http://www.gnu.org/licenses/>.

#ifndef OBJECTSFM_CAMERA_ABSOLUTE_POSE_P4PF_HELPER_H_
#define OBJECTSFM_CAMERA_ABSOLUTE_POSE_P4PF_HELPER_H_

#include <Eigen/Core>
#include <vector>

namespace objectsfm {

// Helper method to the P4Pf algorithm that computes the grobner basis that
// allows for a solution to the problem.
bool FourPointFocalLengthHelper(
    const double glab, const double glac, const double glad, const double glbc,
    const double glbd, const double glcd,
    const Eigen::Matrix<double, 2, 4>& features_normalized,
    std::vector<double>* f, std::vector<Eigen::Vector3d>* depths);

}  // namespace objectsfm

#endif  // OBJECTSFM_CAMERA_ABSOLUTE_POSE_P4PF_HELPER_H_
