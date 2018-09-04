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

#ifndef OBJECTSFM_CAMERA_ABSOLUTE_POSE_ESTIMATION_H_
#define OBJECTSFM_CAMERA_ABSOLUTE_POSE_ESTIMATION_H_

#include <vector>
#include <opencv/cv.h>
#include <Eigen/Core>
#include "basic_structs.h"

namespace objectsfm {

	class AbsolutePoseEstimation
	{
	public:
		AbsolutePoseEstimation(void) {};
		~AbsolutePoseEstimation() {};

		// estimate absolute pose without focal length via UP3P algorithm
		static bool AbsolutePoseWithoutFocalLength(std::vector<Eigen::Vector3d> &pts_w, std::vector<Eigen::Vector2d> &pts_2d,
			double &f, RTPose &pose_absolute, std::vector<double> &errors, double &avg_error);

		// estimate absolute pose with known focal length via EP3P algorithm
		static bool AbsolutePoseWithFocalLength(std::vector<Eigen::Vector3d> &pts_w, std::vector<Eigen::Vector2d> &pts_2d,
			double f, RTPose &pose_absolute, std::vector<double> &errors, double &avg_error);

		// estimate absolute pose with known R via DLT
		static bool AbsolutePoseWithRotation(std::vector<Eigen::Vector3d> &pts_w, std::vector<Eigen::Vector2d> &pts_2d,
			Eigen::Matrix3d R, double &f, RTPose &pose_absolute, std::vector<double> &errors, double &avg_error);

	private:
		static void Error(const std::vector<Eigen::Vector3d>& pts_w, const std::vector<Eigen::Vector2d>& pts_2d,
			double f, RTPose pose, std::vector<double> &errors, double &mse);
	};

}
#endif //OBJECTSFM_CAMERA_ABSOLUTE_POSE_ESTIMATION_H_