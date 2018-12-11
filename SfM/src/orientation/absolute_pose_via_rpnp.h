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

#ifndef OBJECTSFM_CAMERA_ABSOLUTE_POSE_RPNP_H_
#define OBJECTSFM_CAMERA_ABSOLUTE_POSE_RPNP_H_

#include <vector>
#include <opencv/cv.h>
#include <Eigen/Core>
#include "basic_structs.h"

using namespace cv;
using namespace std;

namespace objectsfm {

	class AbsolutePoseRPNP
	{
	public:
		AbsolutePoseRPNP(void) {};
		~AbsolutePoseRPNP() {};


		// Computes camera pose using the three point algorithm and returns all possible
		// solutions (up to 4). Follows steps from the paper "A Novel Parameterization
		// of the Perspective-Three-Point Problem for a direct computation of Absolute
		// Camera position and Orientation" by Kneip et. al.
		// pts_2d: centralized, x = u-u0, y = v-v0
		// return all four pose hypotheses
		static bool RPnP(std::vector<Eigen::Vector3d> &pts_w, std::vector<Eigen::Vector2d> &pts_2d, double f, RTPose &pose);

		static void CalculateCamPose(Mat& XXc, Mat&XXw, Mat& foundR, Mat& foundT);

		static double Error(const std::vector<Eigen::Vector3d>& pts_w, const std::vector<Eigen::Vector2d>& pts_2d, double f, RTPose pose);

	};

}
#endif //OBJECTSFM_CAMERA_ABSOLUTE_POSE_P3P_H_