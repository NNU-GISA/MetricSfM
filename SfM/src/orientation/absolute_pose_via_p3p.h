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

#ifndef OBJECTSFM_CAMERA_ABSOLUTE_POSE_P3P_H_
#define OBJECTSFM_CAMERA_ABSOLUTE_POSE_P3P_H_

#include <vector>
#include <opencv/cv.h>
#include <Eigen/Core>
#include "basic_structs.h"

namespace objectsfm {

	class AbsolutePoseP3P
	{
	public:
		AbsolutePoseP3P(void) {};
		~AbsolutePoseP3P() {};

		// return the best pose by verifying via 2d-3d correspondences
		static bool P3PRANSAC(std::vector<Eigen::Vector3d> &pts_w, std::vector<Eigen::Vector2d> &pts_2d, double f, RTPose &pose);

		// Computes camera pose using the three point algorithm and returns all possible
		// solutions (up to 4). Follows steps from the paper "A Novel Parameterization
		// of the Perspective-Three-Point Problem for a direct computation of Absolute
		// Camera position and Orientation" by Kneip et. al.
		// pts_2d: centralized, x = u-u0, y = v-v0
		// return all four pose hypotheses
		static bool P3P(std::vector<Eigen::Vector3d> &pts_w, std::vector<Eigen::Vector2d> &pts_2d, double f, std::vector<RTPose> &pose_hyps);

		static double Error(const std::vector<Eigen::Vector3d>& pts_w, const std::vector<Eigen::Vector2d>& pts_2d, double f, RTPose pose);

	private:
		static int SolvePlaneRotation(const Eigen::Vector3d normalized_image_points[3],
			const Eigen::Vector3d& intermediate_image_point,
			const Eigen::Vector3d& intermediate_world_point,
			const double d_12,
			double cos_theta[4],
			double cot_alphas[4],
			double* b);

		static void Backsubstitute(const Eigen::Matrix3d& intermediate_world_frame,
			const Eigen::Matrix3d& intermediate_camera_frame,
			const Eigen::Vector3d& world_point_0,
			const double cos_theta,
			const double cot_alpha,
			const double d_12,
			const double b,
			Eigen::Vector3d* translation,
			Eigen::Matrix3d* rotation);
	};

}
#endif //OBJECTSFM_CAMERA_ABSOLUTE_POSE_P3P_H_