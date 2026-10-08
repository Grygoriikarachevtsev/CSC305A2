// C++ include
#include <iostream>
#include <string>
#include <vector>
#include <cmath>

// Utilities for the Assignment
#include "utils.h"

// Image writing library
#define STB_IMAGE_WRITE_IMPLEMENTATION // Do not include this line twice in your project!
#include "stb_image_write.h"

// Shortcut to avoid Eigen:: everywhere, DO NOT USE IN .h
using namespace Eigen;






void raytrace_sphere()
{
    std::cout << "Simple ray tracer, one sphere with orthographic projection" << std::endl;

    const std::string filename("sphere_orthographic.png");
    MatrixXd C = MatrixXd::Zero(800, 800); // Store the color
    MatrixXd A = MatrixXd::Zero(800, 800); // Store the alpha mask

    const Vector3d camera_origin(0, 0, 3);
    const Vector3d camera_view_direction(0, 0, -1);

    // The camera is orthographic, pointing in the direction -z and covering the
    // unit square (-1,1) in x and y
    const Vector3d image_origin(-1, 1, 1);
    const Vector3d x_displacement(2.0 / C.cols(), 0, 0);
    const Vector3d y_displacement(0, -2.0 / C.rows(), 0);

    const double sphere_radius = 0.9;
    const Vector3d sphere_center(0, 0, 0);

    // Single light source
    const Vector3d light_position(-1, 1, 1);

    for (unsigned i = 0; i < C.cols(); ++i)
    {
        for (unsigned j = 0; j < C.rows(); ++j)
        {
            const Vector3d pixel_center = image_origin + double(i) * x_displacement + double(j) * y_displacement;

            // Prepare the ray
            const Vector3d ray_origin = pixel_center;
            const Vector3d ray_direction = camera_view_direction;

            // Intersect with the sphere
            // NOTE: this is a special case of a sphere centered in the origin and for orthographic rays aligned with the z axis
            const Vector3d center_to_ray_origin = ray_origin - sphere_center;

            const double a = ray_direction.dot(ray_direction);
            const double b = 2 * ray_direction.dot(center_to_ray_origin);
            const double c = center_to_ray_origin.dot(center_to_ray_origin) - sphere_radius * sphere_radius;

            // Determine the number of solutions present
            const double discriminant = b * b - 4.0 * a * c;


            if(discriminant >= 0){
                const double square_root = std::sqrt(discriminant);
                
                double t = -1.0; // if it remains -1.0 it means we have not found a valid root yet
                const double t1 = (-b - square_root) / (2.0 * a);
                const double t2 = (-b + square_root) / (2.0 * a);
                
                if(t1 >= 0.0) t = t1; // t1 is closer to the root
                else if(t2 >= 0.0) t = t2; // t1 could be behind us
                
                if(t >= 0.0){
                    const Vector3d ray_intersection = ray_origin + t * ray_direction;
                    const Vector3d ray_normal = (ray_intersection - sphere_center).normalized();
                    
                    C(i, j) = (light_position - ray_intersection).normalized().transpose() * ray_normal;

                    // Clamp to zero
                    C(i, j) = std::max(C(i, j), 0.0);

                    // Makes this hit pixel opaque
                    A(i, j) = 1.0;
                }
            
            }
        }
    }

    // Save to png
    write_matrix_to_png(C, C, C, A, filename);
}


// Helper function for parallelogram intersection
// The assignment specified only a check for a hit, but we can also get the ray intersection
// as well right away. A cleaner refactor.
bool intersect_parallelogram(
    const Vector3d &ray_origin,
    const Vector3d &ray_direction,
    const Vector3d &pgram_origin,
    const Vector3d &pgram_u,
    const Vector3d &pgram_v,
    Vector3d &ray_intersection)
{
    Matrix3d system;

    // [D, -U, -V]
    system.col(0) = ray_direction;
    system.col(1) = -pgram_u;
    system.col(2) = -pgram_v;

    
    const double eps = 1e-8;

    // We need to deal with edge cases, if the determinant is near zero, the
    // system cannot be solved reliably. 
    // Ray is parallel to the plane/degenerate parallelogram.
    if(std::abs(system.determinant()) < eps) return false;

    // [D -U -V] [t alpha beta]^T = P0 - O
    const Vector3d rhs = pgram_origin - ray_origin;
    const Vector3d sol = system.colPivHouseholderQr().solve(rhs);

    const double t = sol(0);
    const double alpha = sol(1);
    const double beta = sol(2);

    // Case if the ray intersection behind the ray origin
    if(t < 0.0) return false;

    // Covers cases where ray hits the infinite plane, but outside the finite parallelogram
    if(alpha < 0.0 || alpha > 1.0) return false;
    if(beta < 0.0 || beta > 1.0) return false;    

    ray_intersection = ray_origin + t * ray_direction;
    return true;
}


void raytrace_parallelogram()
{
    std::cout << "Simple ray tracer, one parallelogram with orthographic projection" << std::endl;

    const std::string filename("plane_orthographic.png");
    MatrixXd C = MatrixXd::Zero(800, 800); // Store the color
    MatrixXd A = MatrixXd::Zero(800, 800); // Store the alpha mask

    const Vector3d camera_origin(0, 0, 3);
    const Vector3d camera_view_direction(0, 0, -1);

    // The camera is orthographic, pointing in the direction -z and covering the unit square (-1,1) in x and y
    const Vector3d image_origin(-1, 1, 1);
    const Vector3d x_displacement(2.0 / C.cols(), 0, 0);
    const Vector3d y_displacement(0, -2.0 / C.rows(), 0);

    // Parameters of the parallelogram (position of the lower-left corner + two sides)
    const Vector3d pgram_origin(-0.5, -0.5, 0);
    const Vector3d pgram_u(1, 0.4, 0);
    const Vector3d pgram_v(0, 0.7, -10);

    // Single light source
    const Vector3d light_position(-1, 1, 1);

    // To avoid calculating plane normal over and over since it is constant
    const Vector3d pgram_normal = pgram_u.cross(pgram_v).normalized();  

    for (unsigned i = 0; i < C.cols(); ++i)
    {
        for (unsigned j = 0; j < C.rows(); ++j)
        {
            const Vector3d pixel_center = image_origin + double(i) * x_displacement + double(j) * y_displacement;

            // Prepare the ray
            const Vector3d ray_origin = pixel_center;
            const Vector3d ray_direction = camera_view_direction;

            // TODO: Check if the ray intersects with the parallelogram
            Vector3d ray_intersection;

            const bool hit = intersect_parallelogram(
                            ray_origin,
                            ray_direction,
                            pgram_origin,
                            pgram_u,
                            pgram_v,
                            ray_intersection);

            if (hit)
            {
                // TODO: The ray hit the parallelogram, compute the exact intersection
                // point

                // Simple diffuse model
                C(i, j) = (light_position - ray_intersection).normalized().transpose() * pgram_normal;

                // Clamp to zero
                C(i, j) = std::max(C(i, j), 0.0);

                // Disable the alpha mask for this pixel
                A(i, j) = 1.0;
            }
        }
    }

    // Save to png
    write_matrix_to_png(C, C, C, A, filename);
}

void raytrace_perspective()
{
    std::cout << "Simple ray tracer, one parallelogram with perspective projection" << std::endl;

    const std::string filename("plane_perspective.png");
    MatrixXd C = MatrixXd::Zero(800, 800); // Store the color
    MatrixXd A = MatrixXd::Zero(800, 800); // Store the alpha mask

    const Vector3d camera_origin(0, 0, 3);
    const Vector3d camera_view_direction(0, 0, -1);

    // The camera is perspective, pointing in the direction -z and covering the unit square (-1,1) in x and y
    const Vector3d image_origin(-1, 1, 1);
    const Vector3d x_displacement(2.0 / C.cols(), 0, 0);
    const Vector3d y_displacement(0, -2.0 / C.rows(), 0);

    // TODO: Parameters of the parallelogram (position of the lower-left corner + two sides)
    const Vector3d pgram_origin(-0.5, -0.5, 0);
    const Vector3d pgram_u(1, 0.4, 0);
    const Vector3d pgram_v(0, 0.7, -10);

    // Single light source
    const Vector3d light_position(-1, 1, 1);

    // To avoid calculating plane normal over and over since it is constant
    const Vector3d pgram_normal = pgram_u.cross(pgram_v).normalized();  

    for (unsigned i = 0; i < C.cols(); ++i)
    {
        for (unsigned j = 0; j < C.rows(); ++j)
        {
            const Vector3d pixel_center = image_origin + double(i) * x_displacement + double(j) * y_displacement;

            // TODO: Prepare the ray (origin point and direction)
            const Vector3d ray_origin = camera_origin;
            const Vector3d ray_direction = (pixel_center - camera_origin).normalized();
            
            // TODO: Check if the ray intersects with the parallelogram
            Vector3d ray_intersection;

            const bool hit = intersect_parallelogram(
                            ray_origin,
                            ray_direction,
                            pgram_origin,
                            pgram_u,
                            pgram_v,
                            ray_intersection);

            if (hit)
            {
                // TODO: The ray hit the parallelogram, compute the exact intersection point

                // Simple diffuse model
                C(i, j) = (light_position - ray_intersection).normalized().transpose() * pgram_normal;

                // Clamp to zero
                C(i, j) = std::max(C(i, j), 0.);

                // Disable the alpha mask for this pixel
                A(i, j) = 1;
            }
        }
    }

    // Save to png
    write_matrix_to_png(C, C, C, A, filename);
}

void raytrace_shading()
{
    std::cout << "Simple ray tracer, one sphere with different shading" << std::endl;

    const std::string filename("shading.png");
    // Have to do a different approach for storing pixel values, C would store only one value per pixel 
    // Having separate matricies is the easier approach
    MatrixXd R = MatrixXd::Zero(800, 800); // Red
    MatrixXd G = MatrixXd::Zero(800, 800); // Green
    MatrixXd B = MatrixXd::Zero(800, 800); // Blue
    MatrixXd A = MatrixXd::Zero(800, 800); // Store the alpha mask

    const Vector3d camera_origin(0, 0, 3);
    const Vector3d camera_view_direction(0, 0, -1);

    // The camera is perspective, pointing in the direction -z and covering the unit square (-1,1) in x and y
    const Vector3d image_origin(-1, 1, 1);
    const Vector3d x_displacement(2.0 / A.cols(), 0, 0);
    const Vector3d y_displacement(0, -2.0 / A.rows(), 0);

    //Sphere setup
    const Vector3d sphere_center(0, 0, 0);
    const double sphere_radius = 0.9;

    //material params
    const Vector3d diffuse_color(1, 0, 1);
    const double specular_exponent = 100;
    const Vector3d specular_color(0., 0, 1);

    // Single light source
    const Vector3d light_position(-1, 1, 1);
    const Vector3d light_intesity(1, 1, 1);
    
    const double ambient = 0.1;
    // grayscale RGB vector
    const Vector3d ambient_color = ambient * Vector3d::Ones();

    for (unsigned i = 0; i < A.cols(); ++i)
    {
        for (unsigned j = 0; j < A.rows(); ++j)
        {
            const Vector3d pixel_center = image_origin + double(i) * x_displacement + double(j) * y_displacement;

            // TODO: Prepare the ray (origin point and direction)
            const Vector3d ray_origin = camera_origin;
            const Vector3d ray_direction = (pixel_center - camera_origin).normalized();

            // Intersect with the sphere
            // TODO: implement the generic ray sphere intersection
            const Vector3d center_to_ray_origin = ray_origin - sphere_center;

            const double a = ray_direction.dot(ray_direction);
            const double b = 2 * ray_direction.dot(center_to_ray_origin);
            const double c = center_to_ray_origin.dot(center_to_ray_origin) - sphere_radius * sphere_radius;

            // Determine the number of solutions present
            const double discriminant = b * b - 4.0 * a * c;


            if(discriminant >= 0){
                const double square_root = std::sqrt(discriminant);
                
                double t = -1.0; // if it remains -1.0 it means we have not found a valid root yet
                const double t1 = (-b - square_root) / (2.0 * a);
                const double t2 = (-b + square_root) / (2.0 * a);
                
                if(t1 >= 0.0) t = t1; // t1 is closer to the root
                else if(t2 >= 0.0) t = t2; // t1 could be behind us
                
                if(t >= 0.0){
                    const Vector3d ray_intersection = ray_origin + t * ray_direction;
                    const Vector3d ray_normal = (ray_intersection - sphere_center).normalized();
                    const Vector3d light_direction = (light_position - ray_intersection).normalized();
                    const Vector3d view_direction = (camera_origin - ray_intersection).normalized();
                    // we will have a shiny highlight whenever the surface normal points close to the halfway vector
                    const Vector3d halfway_direction = (light_direction + view_direction).normalized();
                    
                    // d = max(0, N*L)
                    const double diffuse = std::max(0.0, ray_normal.dot(light_direction));
                    // s = (max(0, N * H))^p
                    double specular = std::pow(std::max(0.0, ray_normal.dot(halfway_direction)), specular_exponent);
                    
                    // C = C_ambient + d(C_diffuse * I) + s(C_specular * I)
                    const Vector3d color = ambient_color + diffuse * diffuse_color.cwiseProduct(light_intesity) 
                                                         + specular * specular_color.cwiseProduct(light_intesity);


                    R(i, j) = color(0);
                    G(i, j) = color(1);
                    B(i, j) = color(2);
                    A(i, j) = 1.0;
                }
            }
        }
    }

    // Save to png
    write_matrix_to_png(R, G, B, A, filename);
}

int main()
{
    raytrace_sphere();
    raytrace_parallelogram();
    raytrace_perspective();
    raytrace_shading();

    return 0;
}
