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

// EPNP is more accurate than P3P, the P3P may suffers from error matching
// while EPnP uses more correspondences to get a robuster result
#ifndef OBJECTSFM_CAMERA_ABSOLUTE_POSE_EPNPF_H_
#define OBJECTSFM_CAMERA_ABSOLUTE_POSE_EPNPF_H_

#include <vector>
#include <opencv/cv.h>
#include "basic_structs.h"

namespace objectsfm {

	class AbsolutePoseEPNPF 
	{
	public:
		AbsolutePoseEPNPF(void);

		~AbsolutePoseEPNPF();

		// sampling the focal length, and then use epnp to estimate the pose
		// then choose the best focal length as estimated
		static void EPNPF(std::vector<Eigen::Vector3d> &pts_w, std::vector<Eigen::Vector2d> &pts_2d, 
			double &f, double f_ratio_min, double f_ratio_max,
			RTPose &pose_absolute, double &mse);

	private:
		static double Error(const std::vector<Eigen::Vector3d>& pts_w, const std::vector<Eigen::Vector2d>& pts_2d, double f, RTPose pose);
	};

}
#endif //OBJECTSFM_CAMERA_ABSOLUTE_POSE_EPNPF_H_