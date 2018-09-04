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

#ifndef OBJECTSFM_CAMERA_ABSOLUTE_POSE_DLT_ORIENTED_H_
#define OBJECTSFM_CAMERA_ABSOLUTE_POSE_DLT_ORIENTED_H_

#include <vector>
#include <opencv/cv.h>
#include <Eigen/Core>
#include "basic_structs.h"

namespace objectsfm {

	class AbsolutePoseDLTOriented
	{
	public:
		AbsolutePoseDLTOriented(void) {};
		~AbsolutePoseDLTOriented() {};

		// Computes the absolute pose and focal length of a camera given the R of 
		// the camera and the initial focal length
		// (r1*Xw+tx)*f = u*(r3*Xw+tz)
		// (r2*Xw+ty)*f = v*(r3*Xw+tz)
		// pts_2d: centralized, x = u-u0, y = v-v0
		// return all four pose hypotheses
		static bool DLTOriented(std::vector<Eigen::Vector3d> &pts_w, std::vector<Eigen::Vector2d> &pts_2d, Eigen::Matrix3d R, double &f, RTPose &pose);

	private:

		static double Error(std::vector<Eigen::Vector3d> &pts_w, std::vector<Eigen::Vector2d> &pts_2d, double &f, RTPose &pose);
	};

}
#endif //OBJECTSFM_CAMERA_ABSOLUTE_POSE_DLT_ORIENTED_H_