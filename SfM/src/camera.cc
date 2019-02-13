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

#include "camera.h"

#include "utils/basic_funcs.h"


namespace objectsfm {

Camera::Camera()
{
	is_mutable_ = true;
}

Camera::~Camera()
{
}

void Camera::AssociateImage(int id_img)
{
	id_img_ = id_img;
}

void Camera::AssociateCamereModel(CameraModel * cam_model)
{
	cam_model_ = cam_model;
}

void Camera::SetRTPose(Eigen::Matrix3d R, Eigen::Vector3d t)
{
	pos_rt_.R = R;
	pos_rt_.t = t;

	// convert into AC pose
	rotation::RotationMatrixToAngleAxis(pos_rt_.R, pos_ac_.a);
	pos_ac_.c = -pos_rt_.R.inverse() * pos_rt_.t;

	UpdateDataFromPose();
}

void Camera::SetRTPose(RTPose pos1_abs, RTPoseRelative pos2to1)
{
	pos_rt_.R = pos1_abs.R * pos2to1.R;
	pos_rt_.t = pos1_abs.t + pos1_abs.R * pos2to1.t;

	// convert into AC pose
	rotation::RotationMatrixToAngleAxis(pos_rt_.R, pos_ac_.a);
	pos_ac_.c = -pos_rt_.R.inverse() * pos_rt_.t;

	UpdateDataFromPose();
}

void Camera::SetACPose(Eigen::Vector3d a, Eigen::Vector3d c)
{
	pos_ac_.a = a;
	pos_ac_.c = c;

	// // convert into Rt pose
	rotation::AngleAxisToRotationMatrix(pos_ac_.a, pos_rt_.R);
	pos_rt_.t = -pos_rt_.R * pos_ac_.c;

	UpdateDataFromPose();
}

void Camera::Transformation(Eigen::Matrix3d R, Eigen::Vector3d t, double scale)
{
	pos_rt_.R = pos_rt_.R * R.inverse();
	pos_ac_.c = scale * R * pos_ac_.c + t;
	pos_rt_.t = -pos_rt_.R * pos_ac_.c;
	rotation::RotationMatrixToAngleAxis(pos_rt_.R, pos_ac_.a);
	
	UpdateDataFromPose();
}

void Camera::UpdateDataFromPose()
{
	// pose angle aixs
	data[0] = pos_ac_.a[0];
	data[1] = pos_ac_.a[1];
	data[2] = pos_ac_.a[2];

	// camera c
	data[3] = pos_ac_.c[0];
	data[4] = pos_ac_.c[1];
	data[5] = pos_ac_.c[2];

	// the intrinsic matrix
	if (cam_model_ != NULL)
	{
		K = Eigen::DiagonalMatrix<double, 3>(cam_model_->f_, cam_model_->f_, 1.0);
	}
	

	// the convertion matrix, from Xw to Xc
	M << pos_rt_.R(0, 0), pos_rt_.R(0, 1), pos_rt_.R(0, 2), pos_rt_.t(0),
		pos_rt_.R(1, 0), pos_rt_.R(1, 1), pos_rt_.R(1, 2), pos_rt_.t(1),
		pos_rt_.R(2, 0), pos_rt_.R(2, 1), pos_rt_.R(2, 2), pos_rt_.t(2);

	// the projection matrix, from Xw to (u-u0, v-v0)
	P = K * M;
}

void Camera::UpdatePoseFromData()
{
	// pose angle aixs
	pos_ac_.a[0] = data[0];
	pos_ac_.a[1] = data[1];
	pos_ac_.a[2] = data[2];
	rotation::AngleAxisToRotationMatrix(pos_ac_.a, pos_rt_.R);

	// camera t
	pos_ac_.c[0] = data[3];
	pos_ac_.c[1] = data[4];
	pos_ac_.c[2] = data[5];
	pos_rt_.t = -pos_rt_.R * pos_ac_.c;

	// the intrinsic matrix
	K = Eigen::DiagonalMatrix<double, 3>(cam_model_->f_, cam_model_->f_, 1.0);

	// the convertion matrix, from Xw to Xc
	M << pos_rt_.R(0, 0), pos_rt_.R(0, 1), pos_rt_.R(0, 2), pos_rt_.t(0),
		pos_rt_.R(1, 0), pos_rt_.R(1, 1), pos_rt_.R(1, 2), pos_rt_.t(1),
		pos_rt_.R(2, 0), pos_rt_.R(2, 1), pos_rt_.R(2, 2), pos_rt_.t(2);

	// the projection matrix, from Xw to (u-u0, v-v0)
	P = K*M;
}

void Camera::SetFocalLength(double f)
{
	cam_model_->SetFocalLength(f);
}

bool Camera::Observe3DPoint(Eigen::Vector3d pt_w, Eigen::Vector2d & pt2d)
{
	if (!cam_model_->f_)
	{
		return false;
	}

	Eigen::Vector4d pt_3d(pt_w(0), pt_w(1), pt_w(2), 1.0);
	Eigen::Vector3d pt_c = P * pt_3d;
	pt2d(0) = pt_c(0) / pt_c(2);
	pt2d(1) = pt_c(1) / pt_c(2);
}

void Camera::AddPoints(Point3D* pt, int idx)
{
	pts_.insert(std::pair<int, Point3D*>(idx, pt));
}

void Camera::AddVisibleCamera(int id_visible_cam)
{
	visible_cams_.push_back(id_visible_cam);
}

void Camera::SetMutable(bool is_mutable)
{
	is_mutable_ = is_mutable;
}

void Camera::SetID(int id)
{
	id_ = id;
}

}  // namespace objectsfm
