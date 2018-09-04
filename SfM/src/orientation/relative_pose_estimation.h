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

#ifndef OBJECTSFM_CAMERA_RELATIVE_POSE_ESTIMATION_H_
#define OBJECTSFM_CAMERA_RELATIVE_POSE_ESTIMATION_H_

#include <vector>
#include <opencv/cv.h>
#include <Eigen/Core>
#include "basic_structs.h"

namespace objectsfm {

	class RelativePoseEstimation
	{
	public:
		RelativePoseEstimation(void) {};
		~RelativePoseEstimation() {};

		// estimate relative pose without focal length via 8-point Fundamental Matrix decomposion
		static bool RelativePoseWithoutFocalLength(std::vector<Eigen::Vector2d> &pts_ref, std::vector<Eigen::Vector2d> &pts_cur,
			double& f_ref, double& f_cur, RTPoseRelative &pose_relative);

		// estimate relative pose with same focal lengths via 2-view focal length estimation 
		static bool RelativePoseWithSameFocalLength(std::vector<Eigen::Vector2d> &pts_ref, std::vector<Eigen::Vector2d> &pts_cur,
			double &f, RTPoseRelative &pose_relative);

		// estimate relative pose with focal length via Essential Matrix decomposion
		static bool RelativePoseWithFocalLength(std::vector<Eigen::Vector2d> &pts_ref, std::vector<Eigen::Vector2d> &pts_cur,
			double f_ref, double f_cur, RTPoseRelative &pose_relative);
	};

}
#endif //OBJECTSFM_CAMERA_RELATIVE_POSE_ESTIMATION_H_