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

#ifndef OBJECTSFM_OPTIMIZER_REPROJECTION_ERROR_POSE_CAM_H_
#define OBJECTSFM_OPTIMIZER_REPROJECTION_ERROR_POSE_CAM_H_

#include "ceres/ceres.h"
#include "ceres/rotation.h"

namespace objectsfm
{
	// pose: [0,1,2] are the angle-axis, [3,4,5] are the translation t.
	// cam: 0 is focal length, [1,2] are the distrotion parameters
	struct ReprojectionErrorPoseCam
	{
		ReprojectionErrorPoseCam(const double observed_x, const double observed_y, const double* point, const double weight)
			: observed_x(observed_x), observed_y(observed_y), point(point), weight(weight){}

		template <typename T>
		bool operator()(const T* const pose,
			const T* const cam,
			T* residuals) const
		{
			// pose[0,1,2] are the angle-axis rotation.
			T xyz_temp[3];
			xyz_temp[0] = T(point[0]) - pose[3];
			xyz_temp[1] = T(point[1]) - pose[4];
			xyz_temp[2] = T(point[2]) - pose[5];

			T p[3];
			ceres::AngleAxisRotatePoint(pose, xyz_temp, p);

			// Compute the center of distortion
			const T xp = p[0] / p[2];
			const T yp = p[1] / p[2];

			// Apply second and fourth order radial distortion.
			const T& focal = cam[0];
			const T& l1 = cam[1];
			const T& l2 = cam[2];
			const T r2 = xp * xp + yp * yp;
			const T distortion = 1.0 + r2 * (l1 + l2 * r2);

			// Compute final projected point position.
			const T predicted_x = focal * distortion * xp;
			const T predicted_y = focal * distortion * yp;

			// The error is the difference between the predicted and observed position.
			residuals[0] = weight*(predicted_x - observed_x);
			residuals[1] = weight*(predicted_y - observed_y);

			return true;
		}

		// Factory to hide the construction of the CostFunction object from
		// the client code.
		static ceres::CostFunction* Create(const double observed_x, const double observed_y, const double* point, const double weight)
		{
			return (new ceres::AutoDiffCostFunction<ReprojectionErrorPoseCam, 2, 6, 3>(
				new ReprojectionErrorPoseCam(observed_x, observed_y, point, weight)));
		}

		const double observed_x;
		const double observed_y;
		const double *point;
		const double weight;
	};

}  // namespace objectsfm

#endif  // OBJECTSFM_OPTIMIZER_SNAVELY_REPROJECTION_ERROR_H_
