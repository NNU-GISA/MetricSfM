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

#ifndef OBJECTSFM_CAMERA_ABSOLUTE_POSE_P4PF_H_
#define OBJECTSFM_CAMERA_ABSOLUTE_POSE_P4PF_H_

#include <vector>
#include <opencv/cv.h>
#include <Eigen/Core>
#include "basic_structs.h"

namespace objectsfm {

	class AbsolutePoseP4PF
	{
	public:
		AbsolutePoseP4PF(void) {};
		~AbsolutePoseP4PF() {};

		// return the best pose by verifying via 2d-3d correspondences
		static double P4PFRANSAC(std::vector<Eigen::Vector3d> &pts_w, std::vector<Eigen::Vector2d> &pts_2d, double &f, RTPose &pose);

		// Computes the absolute pose and focal length of a camera with the P4Pf
		// algorithm from the paper "A general solution to the P4P problem for camera
		// with unknown focal length" by Bujnak et al. The solution involves computing a
		// grobner basis based on a unique constraint of the focal length and pose
		// reprojection.
		// pts_2d: centralized, x = u-u0, y = v-v0
		// return all four pose hypotheses
		static bool P4PF(std::vector<Eigen::Vector3d> &pts_w, std::vector<Eigen::Vector2d> &pts_2d, std::vector<double> &f_hyps, std::vector<RTPose> &pose_hyps);

	private:
		static void GetRigidTransform(const Eigen::Matrix<double, 3, 4>& points1,
			const Eigen::Matrix<double, 3, 4>& points2,
			const bool left_handed_coordinates,
			Eigen::Matrix3d* rotation,
			Eigen::Vector3d* translation);

		static double Error(std::vector<Eigen::Vector3d> &pts_w, std::vector<Eigen::Vector2d> &pts_2d, double &f, RTPose &pose);
	};

}
#endif //OBJECTSFM_CAMERA_ABSOLUTE_POSE_P4PF_H_