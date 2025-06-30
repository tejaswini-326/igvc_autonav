#include <iostream>
#include <cmath>


int main(int argc, char **argv){
	if(argc < 4){
		std::cerr << "Input: x y yaw" << std::endl;
		return 0;
	}
	double x = atof(argv[1]);
	double y = atof(argv[2]);
	double yaw = atof(argv[3]);
	// Use case: while defining initial_base_pos
	// in common_stereo, we found that
	// zed weirdly translates camera_link first and then rotates
	// about the map frame! 
	// This program will give you the correct xy yaw for this 
	// kind of transformation given the input xy yaw which is 
	// obtained by ROTATING the camera_link first and then 
	// translating about the map frame
	
	double r_fin = sqrt(x*x + y*y);
	double theta_fin;
	if(x == 0){
		if(y < 0){theta_fin = -M_PI/2;}
		else if(y == 0){theta_fin = 0;}
		else {theta_fin = M_PI/2;}
	}else{
		theta_fin = atan(y/x);
		if((x < 0 && y <= 0)||(x < 0 && y >= 0)){theta_fin += M_PI;}
	}
	double r_init = r_fin; 
	double theta_init = theta_fin - yaw;

	std::cout << "here are the respection transforms(x,y,yaw): " << r_init * cos(theta_init) << ' ' << r_init * sin(theta_init) << ' ' << yaw << std::endl;


}
