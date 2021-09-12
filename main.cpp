//
//  main.cpp
//  my-geodesic
//
//  Created by Ava on 08.08.21.
//

#include "geodesicDis.h"
#include "heatgeo.h"
//#include "exactgeo.h"

#include <chrono>
#include <iostream>

std::string file = "/Users/ava/Uni/BasicsLab/Projects/my-geodesic/my-geodesic/47984.obj";
//std::string file = "/Users/ava/Uni/BasicsLab/Projects/my-geodesic/my-geodesic/bunny.obj";


int main(int argc, char *argv[]){
    
    
    GeodesicDis geoDis;
    geoDis.compute_geodesic(file);
    

    /*
     Exact Discrete Geodesic Distances
     https://github.com/libigl/libigl/blob/main/tutorial/206_GeodesicDistance/main.cpp
     */
//    Excatgeo exactgeo;
//    exactgeo.exact_geodesic(file);
    

    /*
     Heat Method For Fast Geodesic Distance Approximation
     https://github.com/libigl/libigl/blob/main/tutorial/716_HeatGeodesics/main.cpp
     */
    
//    Heatgeo heatgeo;
//    heatgeo.heat_geo(file);
    
}

