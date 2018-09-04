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

#include "structure.h"

#include <Eigen/Cholesky>
#include <Eigen/Core>
#include <Eigen/Eigenvalues>
#include <Eigen/Geometry>
#include <Eigen/SVD>

#include "camera.h"
#include "utils/basic_funcs.h"

namespace objectsfm {

Point3D::Point3D()
{
	id_ = 0;
	weight = 1.0;
	is_mutable_ = true;
	is_bad_estimated_ = false;
	is_new_added_ = true;
}

Point3D::~Point3D()
{
}

void Point3D::Trianglate(Camera * cam1, Eigen::Vector2d pt1, Camera * cam2, Eigen::Vector2d pt2, Eigen::Vector3d & pt3d)
{
	std::vector<Eigen::Vector3d> ray_origins;
	std::vector<Eigen::Vector3d> ray_directions;

	Eigen::Vector3d origin1 = cam1->pos_ac_.c;
	Eigen::Vector3d dir_c1(pt1(0), pt1(1), cam1->cam_model_->f_);
	Eigen::Vector3d dir_w1 = cam1->pos_rt_.R.transpose()*dir_c1;
	dir_w1.normalize();
	ray_origins.push_back(origin1);
	ray_directions.push_back(dir_w1);

	Eigen::Vector3d origin2 = cam2->pos_ac_.c;
	Eigen::Vector3d dir_c2(pt2(0), pt2(1), cam2->cam_model_->f_);
	Eigen::Vector3d dir_w2 = cam2->pos_rt_.R.transpose()*dir_c2;
	dir_w2.normalize();
	ray_origins.push_back(origin2);
	ray_directions.push_back(dir_w2);

	//
	Eigen::Matrix4d A;
	A.setZero();
	Eigen::Vector4d b;
	b.setZero();
	for (int i = 0; i < ray_origins.size(); i++)
	{
		const Eigen::Vector4d ray_direction_homog(
			ray_directions[i].x(), ray_directions[i].y(), ray_directions[i].z(), 0);
		const Eigen::Matrix4d A_term =
			Eigen::Matrix4d::Identity() -
			ray_direction_homog * ray_direction_homog.transpose();
		A += A_term;
		b += A_term * ray_origins[i].homogeneous();
	}

	Eigen::LLT<Eigen::Matrix4d> linear_solver(A);
	Eigen::Vector4d triangulated_point = linear_solver.solve(b);
	if (linear_solver.info() != Eigen::Success)
	{
		return;
	}

	pt3d(0) = triangulated_point[0] / triangulated_point[3];
	pt3d(1) = triangulated_point[1] / triangulated_point[3];
	pt3d(2) = triangulated_point[2] / triangulated_point[3];
}

double Point3D::ReprojectToCam(Camera * cam, Eigen::Vector3d pt3d, Eigen::Vector2d pt_obs, Eigen::Vector2d &pt_est)
{
	Eigen::Vector4d pt_w_homo(pt3d(0), pt3d(1), pt3d(2), 1.0);

	double error_reproj = 0.0;
	Eigen::Vector3d pt_c = cam->M * pt_w_homo; // convert into the camera coordinate system
	if (pt_c(2) < 0)
	{
		return 1000.0;
	}

	double x = pt_c(0) / pt_c(2);
	double y = pt_c(1) / pt_c(2);
	double r2 = x * x + y * y;
	double distortion = 1.0 + r2 * (cam->cam_model_->k1_ + cam->cam_model_->k2_  * r2);
	pt_est(0) = cam->cam_model_->f_ * distortion * x; // the coordinate in pixel on the image
	pt_est(1) = cam->cam_model_->f_ * distortion * y;

	Eigen::Vector2d dev = pt_est - pt_obs;
	return dev.norm();
}

double Point3D::IntersectionAngle(Eigen::Vector3d pt3d, Camera * cam1, Camera * cam2)
{
	std::vector<Eigen::Vector3d> ray_directions(2);

	ray_directions[0] = pt3d - cam1->pos_ac_.c;
	ray_directions[0] /= ray_directions[0].norm();

	ray_directions[1] = pt3d - cam2->pos_ac_.c;
	ray_directions[1] /= ray_directions[1].norm();

	double angle = std::acos(ray_directions[0].dot(ray_directions[1]));

	return angle;
}

void Point3D::SetID(int id)
{
	id_ = id;
}

void Point3D::AddObservation(Camera * cam, double x, double y, int idx)
{
	cams_.insert(std::pair<int, Camera*>(idx, cam));

	Eigen::Vector2d pt(x, y);
	pts2d_.insert(std::pair<int,Eigen::Vector2d>(idx, pt));

	key_new_obs_ = idx;
}

void Point3D::RemoveObservation(int idx)
{
	std::map<int, Camera*>::iterator iter_cam = cams_.find(key_new_obs_);
	if (iter_cam != cams_.end())
	{
		cams_.erase(iter_cam);
	}

	std::map<int, Eigen::Vector2d>::iterator iter_pts = pts2d_.find(key_new_obs_);
	if (iter_pts != pts2d_.end())
	{
		pts2d_.erase(iter_pts);
	}
}

// x,y are centralized to the principle point of the image
// Xc = [R|t]Xw = M*Xw  and  Xc x [x,y,f]T = 0
// so, [0 -f y; f 0 -x; -y x 0] * M * Xw = 0
// (-f*M2+y*M3)*Xw = 0
// ( f*M1-x*M3)*Xw = 0
// (-y*M1+x*M2)*Xw = 0
bool Point3D::Trianglate(double th_error, double th_angle)
{
	if (cams_.size() < 2)
	{
		return false;
	}

	// 
	Eigen::MatrixXd A(int(2 * pts2d_.size()), 4);
	A.setZero();

	std::map<int, Camera*>::iterator iter_cams = cams_.begin();
	std::map<int, Eigen::Vector2d>::iterator iter_pts = pts2d_.begin();
	int i = 0;
	while (iter_cams != cams_.end())
	{
		A.block<1, 4>(2 * i + 0, 0) = -iter_cams->second->M.row(1)* iter_cams->second->cam_model_->f_
			+ iter_cams->second->M.row(2)*iter_pts->second(1);
		A.block<1, 4>(2 * i + 1, 0) =  iter_cams->second->M.row(0)* iter_cams->second->cam_model_->f_
			- iter_cams->second->M.row(2)*iter_pts->second(0);
		iter_cams++;
		iter_pts++;
		i++;
	}
	Eigen::Vector4d triangulated_point = A.jacobiSvd(Eigen::ComputeFullV).matrixV().rightCols<1>().head(4);
	data[0] = triangulated_point[0] / triangulated_point[3];
	data[1] = triangulated_point[1] / triangulated_point[3];
	data[2] = triangulated_point[2] / triangulated_point[3];

	// verify the quality of the triangulated 3d point
	Reprojection();
	if (std::sqrt(mse_) > th_error || !SufficientTriangulationAngle(th_angle))
	{
		return false;
	}
	return true;

	//Eigen::MatrixXd design_matrix(3 * num, 4 + num);
	//for (int i = 0; i <num; i++) 
	//{
	//	design_matrix.block<3, 4>(3 * i, 0) = -cams_[i]->P;
	//	design_matrix.block<3, 1>(3 * i, 4 + i) = pts2d_[i].homogeneous();
	//}

	//Eigen::Vector4d triangulated_point = design_matrix.jacobiSvd(Eigen::ComputeFullV).matrixV().rightCols<1>().head(4);
	//return true;
}

bool Point3D::Trianglate2(double th_error, double th_angle)
{
	std::vector<Eigen::Vector3d> ray_origins;
	std::vector<Eigen::Vector3d> ray_directions;

	std::map<int, Camera*>::iterator iter_cams = cams_.begin();
	std::map<int, Eigen::Vector2d>::iterator iter_pts = pts2d_.begin();
	while (iter_cams != cams_.end())
	{
		Eigen::Vector3d origin = iter_cams->second->pos_ac_.c;
		Eigen::Vector3d dir_c(iter_pts->second(0), iter_pts->second(1), iter_cams->second->cam_model_->f_);
		Eigen::Vector3d dir_w = iter_cams->second->pos_rt_.R.transpose()*dir_c;
		dir_w.normalize();

		ray_origins.push_back(origin);
		ray_directions.push_back(dir_w);

		iter_cams++;
		iter_pts++;
	}

	Eigen::Matrix4d A;
	A.setZero();
	Eigen::Vector4d b;
	b.setZero();
	for (int i = 0; i < ray_origins.size(); i++) 
	{
		const Eigen::Vector4d ray_direction_homog(
			ray_directions[i].x(), ray_directions[i].y(), ray_directions[i].z(), 0);
		const Eigen::Matrix4d A_term =
			Eigen::Matrix4d::Identity() -
			ray_direction_homog * ray_direction_homog.transpose();
		A += A_term;
		b += A_term * ray_origins[i].homogeneous();
	}

	Eigen::LLT<Eigen::Matrix4d> linear_solver(A);
	Eigen::Vector4d triangulated_point = linear_solver.solve(b);
	if (linear_solver.info() != Eigen::Success)
	{
		return false;
	}

	data[0] = triangulated_point[0] / triangulated_point[3];
	data[1] = triangulated_point[1] / triangulated_point[3];
	data[2] = triangulated_point[2] / triangulated_point[3];

	// verify the quality of the triangulated 3d point
	Reprojection();
	if (std::sqrt(mse_) > th_error || !SufficientTriangulationAngle(th_angle))
	{
		return false;
	}
	return true;
}

void Point3D::Reprojection()
{
	Eigen::Vector4d pt_w_homo(data[0], data[1], data[2], 1.0);

	mse_ = 0.0;
	int count_cam = 0;

	std::map<int, Camera*>::iterator iter_cams = cams_.begin();
	std::map<int, Eigen::Vector2d>::iterator iter_pts = pts2d_.begin();
	//std::map<int, bool>::iterator iter_inliers = is_inlier_.begin();
	while (iter_cams != cams_.end())
	{
		Eigen::Vector3d pt_c = iter_cams->second->M * pt_w_homo; // convert into the camera coordinate system
		if (pt_c(2) < 0)
		{
			mse_ = 100000.0;
			return;
		}

		double x = pt_c(0) / pt_c(2);
		double y = pt_c(1) / pt_c(2);
		double r2 = x * x + y * y;
		double distortion = 1.0 + r2 * (iter_cams->second->cam_model_->k1_ + iter_cams->second->cam_model_->k2_  * r2);
		double u = iter_cams->second->cam_model_->f_ * distortion * x; // the coordinate in pixel on the image
		double v = iter_cams->second->cam_model_->f_ * distortion * y;
		//errors_[i] = std::pow(u - iter_pts->second(0), 2) + std::pow(v - iter_pts->second(1), 2);
		mse_ += std::pow(u - iter_pts->second(0), 2) + std::pow(v - iter_pts->second(1), 2);
		count_cam++;

		iter_cams++;
		iter_pts++;
	}
	mse_ /= count_cam;
}

double Point3D::ReprojectToCam(Camera * cam, double obs_x, double obs_y)
{
	Eigen::Vector4d pt_w_homo(data[0], data[1], data[2], 1.0);

	double error_reproj = 0.0;
	Eigen::Vector3d pt_c = cam->M * pt_w_homo; // convert into the camera coordinate system
	if (pt_c(2) < 0)
	{
		error_reproj = 100000.0;
		return error_reproj;
	}

	double x = pt_c(0) / pt_c(2);
	double y = pt_c(1) / pt_c(2);
	double r2 = x * x + y * y;
	double distortion = 1.0 + r2 * (cam->cam_model_->k1_ + cam->cam_model_->k2_  * r2);
	double u = cam->cam_model_->f_ * distortion * x; // the coordinate in pixel on the image
	double v = cam->cam_model_->f_ * distortion * y;
	error_reproj = std::sqrt(std::pow(u - obs_x, 2) + std::pow(v - obs_y, 2));

	return error_reproj;
}

bool Point3D::SufficientTriangulationAngle(double th_angle_triangulation)
{
	std::vector<Eigen::Vector3d> ray_directions(cams_.size());

	int i = 0;
	std::map<int, Camera*>::iterator iter = cams_.begin();
	while (iter != cams_.end())
	{
		ray_directions[i](0) = data[0] - iter->second->pos_ac_.c(0);
		ray_directions[i](1) = data[1] - iter->second->pos_ac_.c(1);
		ray_directions[i](2) = data[2] - iter->second->pos_ac_.c(2);

		ray_directions[i] /= ray_directions[i].norm();
		iter++;
		i++;
	}

	double cos_of_min_angle = std::cos(th_angle_triangulation);
	for (size_t i = 0; i < ray_directions.size()-1; i++)
	{
		for (size_t j = i+1; j < ray_directions.size(); j++)
		{
			double angle = std::acos(ray_directions[i].dot(ray_directions[j]));
			if (ray_directions[i].dot(ray_directions[j]) < cos_of_min_angle)
			{
				return true;
			}
		}
	}
	return false;
}

void Point3D::SetMutable(bool is_mutable)
{
	is_mutable_ = is_mutable;
}

}  // namespace objectsfm
