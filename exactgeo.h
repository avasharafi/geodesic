//
//  exactgeo.h
//  my-geodesic
//
//  Created by Ava on 09.09.21.
//

#ifndef exactgeo_h
#define exactgeo_h

#include <cassert>
#include <chrono>
#include <cmath>
#include <fstream>
#include <iostream>
#include <limits>
#include <sstream>
#include <string>
#include <unordered_map>
#include <vector>

#include <igl/opengl/glfw/Viewer.h>
#include <igl/exact_geodesic.h>
#include <igl/unproject_onto_mesh.h>
#include <igl/parula.h>
#include <igl/isolines_map.h>
#include <igl/PI.h>
#include <iostream>


using namespace Eigen;
using namespace std;

class Excatgeo{
    
private:
    void set_colormap(igl::opengl::glfw::Viewer & viewer)
    {
      const int num_intervals = 30;
      Eigen::MatrixXd CM(num_intervals,3);
      // Colormap texture
      for(int i = 0;i<num_intervals;i++)
      {
        double t = double(num_intervals - i - 1)/double(num_intervals-1);
        CM(i,0) = std::max(std::min(2.0*t-0.0,1.0),0.0);
        CM(i,1) = std::max(std::min(2.0*t-1.0,1.0),0.0);
        CM(i,2) = std::max(std::min(6.0*t-5.0,1.0),0.0);
      }
      igl::isolines_map(Eigen::MatrixXd(CM),CM);
      viewer.data().set_colormap(CM);
    }

public:
    int exact_geodesic(std::string file){
    
      Eigen::MatrixXd V;
      Eigen::MatrixXi F;
      igl::opengl::glfw::Viewer viewer;
        
      // Load a mesh
      igl::read_triangle_mesh(file, V, F);

      const auto update_distance = [&](const int vid)
      {
        
        Eigen::VectorXi VS,FS,VT,FT;
        // The selected vertex is the source
        VS.resize(1);
        VS << vid;
        // All vertices are the targets
        VT.setLinSpaced(V.rows(),0,V.rows()-1);
        Eigen::VectorXd d;
        std::cout<<"Computing geodesic distance to vertex "<<vid<<"..."<<std::endl;
          
      
          
        igl::exact_geodesic(V,F,VS,FS,VT,FT,d);
          

          
        // Plot the mesh
        set_colormap(viewer);
        viewer.data().set_data(d);
          
          
      };

      // Plot a distance when a vertex is picked
      viewer.callback_mouse_down =
      [&](igl::opengl::glfw::Viewer& viewer, int, int)->bool
      {
        int fid;
        Eigen::Vector3f bc;
        // Cast a ray in the view direction starting from the mouse position
        double x = viewer.current_mouse_x;
        double y = viewer.core().viewport(3) - viewer.current_mouse_y;
        if(igl::unproject_onto_mesh(
          Eigen::Vector2f(x,y),
          viewer.core().view,
          viewer.core().proj,
          viewer.core().viewport,
          V,
          F,
          fid,
          bc))
        {
          int max;
          bc.maxCoeff(&max);
          int vid = F(fid,max);
          update_distance(vid);
          return true;
        }
        return false;
      };
      viewer.data().set_mesh(V,F);
      viewer.data().show_lines = false;

      cout << "Click on mesh to define new source.\n" << std::endl;
      update_distance(0);
      return viewer.launch();
    }

};
#endif /* exactgeo_h */
