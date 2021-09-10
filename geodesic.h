////
////  geodesic.h
////  my-geodesic
////
////  Created by Ava on 14.08.21.
////
//
//#ifndef geodesic_h
//#define geodesic_h
//
//#include <igl/readOFF.h>
//#include <igl/opengl/glfw/Viewer.h>
//#include <igl/read_triangle_mesh.h>
//
//#include <igl/unproject_onto_mesh.h>
//#include <igl/opengl/glfw/Viewer.h>
//#include <iostream>
//
//using namespace std;
//
//namespace Geodesic {
//
//    float pow2(float f) { return f * f; }
//    float sqrt_sat(float f) { return f <= 0.0f ? 0.0f : std::sqrt(f); }
//    float fast_abs(float f) { return f < 0 ? -f : f; }
//
//    struct pos3{
//        float x = 0;
//        float y = 0;
//        float z = 0;
//
//        pos3 operator+(pos3 const& r) const { return {x + r.x, y + r.y, z + r.z}; }
//        pos3 operator/(float f) const { return {x / f, y / f, z / f}; }
//    };
//    float distance(pos3 a, pos3 b){
//        auto dx = a.x - b.x;
//        auto dy = a.y - b.y;
//        auto dz = a.z - b.z;
//        return std::sqrt(dx * dx + dy * dy + dz * dz);
//    }
//
//    struct triangle{
//        int v0 = 0;
//        int v1 = 0;
//        int v2 = 0;
//    };
//
//    int compute_geodesic(std::string file,int src_idx, int target_idx){
//
//        // load mesh in ad hoc half-edge data structure
//        std::vector<pos3> vertex_pos;
//        std::vector<triangle> faces;
//        std::vector<int> halfedge_to_vertex;
//        std::vector<int> halfedge_to_face;
//        std::vector<int> halfedge_to_next;
//        std::vector<int> halfedge_to_prev;
//        std::vector<int> vertex_to_outgoing_halfedge;
//        std::vector<float> edge_lengths;
//
//        auto load_start = std::chrono::system_clock::now();
//
//        std::ifstream obj(file);
//
//        if (!obj.good())
//        {
//            cerr << "unable to open " << file;
//            return EXIT_FAILURE;
//        }
//
//        std::unordered_map<int64_t, int> edge_to_idx;
//        auto halfedge_of = [&](int v0, int v1) -> int { // from v0 -> v1
//            int o = 0;
//            if (v0 > v1)
//            {
//                std::swap(v0, v1);
//                o = 1;
//            }
//            auto edge_nr = (int64_t(v1) << 32) + v0;
//
//            // edge exists
//            if (edge_to_idx.count(edge_nr))
//                return edge_to_idx[edge_nr] * 2 + o;
//
//            // new edge
//            int e = int(halfedge_to_face.size()) / 2;
//            edge_to_idx[edge_nr] = e;
//
//            vertex_to_outgoing_halfedge[size_t(v0)] = e * 2 + 0;
//            vertex_to_outgoing_halfedge[size_t(v1)] = e * 2 + 1;
//
//            halfedge_to_vertex.push_back(v1);
//            halfedge_to_vertex.push_back(v0);
//
//            halfedge_to_face.push_back(-1);
//            halfedge_to_face.push_back(-1);
//
//            halfedge_to_next.push_back(-1);
//            halfedge_to_next.push_back(-1);
//
//            halfedge_to_prev.push_back(-1);
//            halfedge_to_prev.push_back(-1);
//
//            return e * 2 + o;
//        };
//
//        std::string line;
//        while (std::getline(obj, line))
//        {
//            if (line.size() <= 2 || line[0] == '#')
//                continue; // comments
//
//            // vertex position
//            if (line[0] == 'v' && line[1] == ' ')
//            {
//                std::istringstream ss(line.substr(2));
//                pos3 p;
//                ss >> p.x >> p.y >> p.z;
//                vertex_pos.push_back(p);
//                vertex_to_outgoing_halfedge.push_back(-1);
//            }
//
//            // face
//            if (line[0] == 'f')
//            {
//                std::string s0, s1, s2;
//                std::istringstream ss(line.substr(2));
//                ss >> s0 >> s1 >> s2;
//                for (auto& c : s0)
//                    if (c == '/')
//                        c = ' ';
//                for (auto& c : s1)
//                    if (c == '/')
//                        c = ' ';
//                for (auto& c : s2)
//                    if (c == '/')
//                        c = ' ';
//                int v0, v1, v2;
//                std::stringstream(s0) >> v0;
//                std::stringstream(s1) >> v1;
//                std::stringstream(s2) >> v2;
//                v0 -= 1;
//                v1 -= 1;
//                v2 -= 1;
//                auto f = int(faces.size());
//                faces.push_back({v0, v1, v2});
//
//                auto h01 = halfedge_of(v0, v1);
//                auto h12 = halfedge_of(v1, v2);
//                auto h20 = halfedge_of(v2, v0);
//
//                assert(0 <= h01 && h01 < int(halfedge_to_vertex.size()));
//                assert(0 <= h12 && h12 < int(halfedge_to_vertex.size()));
//                assert(0 <= h20 && h20 < int(halfedge_to_vertex.size()));
//
//                halfedge_to_face[size_t(h01)] = f;
//                halfedge_to_face[size_t(h12)] = f;
//                halfedge_to_face[size_t(h20)] = f;
//
//                halfedge_to_next[size_t(h01)] = h12;
//                halfedge_to_next[size_t(h12)] = h20;
//                halfedge_to_next[size_t(h20)] = h01;
//
//                halfedge_to_prev[size_t(h01)] = h20;
//                halfedge_to_prev[size_t(h12)] = h01;
//                halfedge_to_prev[size_t(h20)] = h12;
//            }
//        }
//
//        // sanity
//        for (int h = 0; h < int(halfedge_to_face.size()); ++h)
//        {
//            auto f = halfedge_to_face[size_t(h)];
//            if (f == -1)
//                continue;
//
//            auto h_next = halfedge_to_next[size_t(h)];
//            auto h_prev = halfedge_to_prev[size_t(h)];
//
//            assert(h == halfedge_to_next[size_t(h_prev)]);
//            assert(h == halfedge_to_prev[size_t(h_next)]);
//        }
//
//
//        // compute edge lengths
//        edge_lengths.resize(halfedge_to_face.size() / 2);
//        auto edge_sum = 0.0;
//        for (size_t e = 0; e < halfedge_to_face.size() / 2; ++e)
//        {
//           auto v0 = halfedge_to_vertex[e * 2 + 0];
//           auto v1 = halfedge_to_vertex[e * 2 + 1];
//
//           auto p0 = vertex_pos[size_t(v0)];
//           auto p1 = vertex_pos[size_t(v1)];
//
//           auto l = distance(p0, p1);
//           edge_lengths[e] = l;
//
//           edge_sum += double(l);
//        }
//
//
//        // normalize edge lengths
//        float avg_edge_length = 0;
//
//        avg_edge_length = float(edge_sum / edge_lengths.size());
//        auto inv_avg_edge_length = 1 / avg_edge_length;
//        for (auto& l : edge_lengths)
//            l *= inv_avg_edge_length;
//
//
//
//        std::cout<< "\n" << vertex_pos.size() << " vertices" << std::endl;
//        std::cout<< "\n" << faces.size() << " triangles" << std::endl;
//        std::cout<< "\n" << halfedge_to_face.size() << " halfedges" << std::endl;
//        std::cout<< "\n" << avg_edge_length << " average edge length \n" << std::endl;
//
//        auto f_cnt = faces.size();
//        auto v_cnt = vertex_pos.size();
//        auto h_cnt = halfedge_to_face.size();
//
//        // instant geodesics
//           {
//               auto ig_start = std::chrono::system_clock::now();
//
//               // Keep track of #iters and #expansions
//               float iteration = 0;
//               auto expansions = 0;
//               auto updates = 0;
//
//               // propagation data
//               std::vector<float> face_to_extra_distance(f_cnt, 0.0f);
//               std::vector<float> face_to_center_distance(f_cnt, std::numeric_limits<float>::max());
//               std::vector<float> halfedge_to_vertex_distance_sqr(h_cnt, -1.0f);
//
//               // queues with halfedge indices
//               std::vector<int> queue_from;
//               std::vector<int> queue_to;
//
//               auto h_begin = vertex_to_outgoing_halfedge[size_t(src_idx)];
//               auto h_end = vertex_to_outgoing_halfedge[size_t(target_idx)];
//
//               auto f_begin = halfedge_to_face[size_t(h_begin)];
//               auto f_end = halfedge_to_face[size_t(h_end)];
//
//               std::cout<<"h_begin:\t"<<h_begin<<"\t\th_end:\t"<<h_end<<std::endl;
//               std::cout<<"f_begin:\t"<<f_begin<<"\t\tf_end:\t"<<f_end<<std::endl;
//
//               auto v_begin = halfedge_to_vertex[size_t(src_idx)];
//               auto v_end = halfedge_to_vertex[size_t(target_idx)];
//
//               std::cout<<"v_begin:\t"<<v_begin<<"\t\tv_end:\t"<<v_end<<std::endl;
//
//               std::vector<int> passed_halfedge;
//               std::vector<int> passed_vector;
//               std::vector<int> passed_face;
//
//
//               // add source (currently only works for non-boundary vertices)
//               {
//                   // iterate over all faces
//                   auto h_begin = vertex_to_outgoing_halfedge[size_t(src_idx)];
//                   auto h = h_begin;
//                   do
//                   {
//                       auto f = halfedge_to_face[size_t(h)];
//
//                       // add source to face
//                       if (f != -1)
//                       {
//                           auto h0 = h;
//                           auto h1 = halfedge_to_next[size_t(h0)];
//                           auto h2 = halfedge_to_next[size_t(h1)];
//
//                           auto v0 = halfedge_to_vertex[size_t(h0)];
//                           auto v1 = halfedge_to_vertex[size_t(h1)];
//                           auto v2 = halfedge_to_vertex[size_t(h2)];
//
//    //                       std::cout<<"v0:\t"<<v0<<std::endl;
//    //                       std::cout<<"v1:\t"<<v1<<std::endl;
//    //                       std::cout<<"v2:\t"<<v2<<std::endl;
//
//                           // v2 is src vertex
//                           assert(v2 == src_idx);
//
//                           auto p0 = vertex_pos[size_t(v0)];
//                           auto p1 = vertex_pos[size_t(v1)];
//                           auto p2 = vertex_pos[size_t(v2)];
//                           auto centroid = (p0 + p1 + p2) / 3.0f;
//
//                           halfedge_to_vertex_distance_sqr[size_t(h2)] = 0.0f;
//                           halfedge_to_vertex_distance_sqr[size_t(h0)] = pow2(edge_lengths[size_t(h0 / 2)]);
//                           halfedge_to_vertex_distance_sqr[size_t(h1)] = pow2(edge_lengths[size_t(h2 / 2)]);
//
//                           face_to_extra_distance[size_t(f)] = 0.0f;
//                           face_to_center_distance[size_t(f)] = distance(centroid, vertex_pos[size_t(src_idx)]);
//
//                           // add to queue
//                           queue_from.push_back(h1);
//                       }
//
//                       h = halfedge_to_next[size_t(h ^ 1)]; // opposite -> next
//                   } while (h != h_begin);
//               }
//
//               // propagate
//               const int behind_mask = int(0x80000000);
//               while (!queue_from.empty())
//               {
//                   iteration += 1.0f;
//
//                   queue_to.clear();
//
//                   // no range-based for because queue is changing!
//                   for (auto qi = 0u; qi < queue_from.size(); ++qi)
//                   {
//                       auto idx = queue_from[qi];
//
//                       auto is_behind = idx & behind_mask;
//                       auto h_source = idx & ~behind_mask;
//
//    //                   std::cout<<"is_behind:\t"<<is_behind<<std::endl;
//    //                   std::cout<<"h_source:\t"<<h_source<<std::endl;
//
//
//                       auto f_source = halfedge_to_face[size_t(h_source)];
//
//                       //std::cout<<"f_source:\t"<<f_source<<std::endl;
//
//
//                       assert(f_source >= 0);
//    //                   assert(f_source != f_end);
//
//                       if(f_source==f_end){
//                           queue_from.clear();
//                           queue_to.clear();
//                           std::cout<<"reach to target"<<std::endl;
//                           break;
//
//                       }
//
//                       assert(face_to_center_distance[size_t(f_source)] < std::numeric_limits<float>::max());
//                       assert(face_to_extra_distance[size_t(f_source)] >= 0);
//
//                       // Expand
//                       ++expansions;
//
//                       // Input Data
//                       auto h1 = h_source ^ 1; // opposite
//                       auto h2 = halfedge_to_prev[size_t(h1)];
//                       auto h3 = halfedge_to_prev[size_t(h2)];
//
//
//                       passed_halfedge.push_back(h1);
//                       passed_vector.push_back(halfedge_to_vertex[h1]);
//
//    //                   std::cout<<"v:\t"<<halfedge_to_vertex[h1]<<std::endl;
//    //                   std::cout<<"h:\t"<<h1<<std::endl;
//    //                   std::cout<<"v_prev:\t"<<halfedge_to_vertex[h2]<<std::endl;
//    //                   std::cout<<"v2_prev:\t"<<halfedge_to_vertex[h3]<<std::endl;
//
//                       auto f_target = halfedge_to_face[size_t(h1)];
//
//    //                   std::cout<<"f:\t"<<f_target<<std::endl;
//
//                       passed_face.push_back(f_target);
//
//                       // skip border
//                       if (f_target == -1)
//                           continue;
//
//                       auto e1 = edge_lengths[size_t(h1 / 2)];
//                       auto e2 = edge_lengths[size_t(h2 / 2)];
//                       auto e3 = edge_lengths[size_t(h3 / 2)];
//
//                       auto d1_sqr = halfedge_to_vertex_distance_sqr[size_t(h_source)];
//                       auto d2_sqr = halfedge_to_vertex_distance_sqr[size_t(halfedge_to_prev[size_t(h_source)])];
//                       assert(d1_sqr >= 0);
//                       assert(d2_sqr >= 0);
//
//                       auto ref_d_t = face_to_center_distance[size_t(f_target)];
//                       auto prev_sigma_t = face_to_extra_distance[size_t(f_source)];
//
//                       // Reconstruct points
//                       auto px = (e1 * e1 + (e2 * e2 - e3 * e3)) / (e1 + e1);
//                       auto py = sqrt_sat(e2 * e2 - px * px);
//
//                       auto sx = (e1 * e1 + (d1_sqr - d2_sqr)) / (e1 + e1);
//                       auto sy_neg = sqrt_sat(d1_sqr - sx * sx);
//
//                       auto cx = (px + e1) * (1 / 3.0f);
//                       auto cy = py * (1 / 3.0f);
//
//                       // Prepare update
//                       float dA, dB, dC;
//                       float d_t;
//                       float sigma_t = prev_sigma_t;
//
//                       int next_h2 = h2;
//                       int next_h3 = h3;
//
//                       // Precalc
//                       auto d_s_1 = std::sqrt(sx * sx + sy_neg * sy_neg);
//                       auto d_s_2 = std::sqrt(pow2(sx - e1) + sy_neg * sy_neg);
//
//                       auto d_c_1 = std::sqrt(cx * cx + cy * cy);
//                       auto d_c_2 = std::sqrt(pow2(cx - e1) + cy * cy);
//
//                       // Source on same side
//                       if (is_behind)
//                       {
//                           auto dis1 = d_c_1 + d_s_1;
//                           auto dis2 = d_c_2 + d_s_2;
//
//                           if (dis1 < dis2) // turn left
//                           {
//                               dA = +0.0f;
//                               dB = e1 * e1;
//                               dC = e2 * e2;
//
//                               sigma_t += d_s_1;
//                               d_t = sigma_t + d_c_1;
//                           }
//                           else // turn right
//                           {
//                               dA = e1 * e1;
//                               dB = +0.0f;
//                               dC = e3 * e3;
//
//                               sigma_t += d_s_2;
//                               d_t = sigma_t + d_c_2;
//                           }
//                       }
//                       // Source on opposite side
//                       else
//                       {
//                           auto bend_left = false;
//                           auto bend_right = false;
//
//                           // data-driven bending heuristic
//                           {
//                               static constexpr float threshold_c = 5.1424f;
//                               static constexpr float threshold_g = 4.20638f;
//                               static constexpr float threshold_h = 0.504201f;
//                               static constexpr float threshold_hg = 2.84918f;
//                               static constexpr float lambda[16] = {0.320991f, 0.446887f, 0.595879f,  0.270094f,  0.236679f, 0.159685f,  0.0872932f, 0.434132f,
//                                                                    1.0f,      0.726262f, 0.0635997f, 0.0515979f, 0.56903f,  0.0447586f, 0.0612103f, 0.718198f};
//
//                               auto max_e = std::max(e1, std::max(e2, e3));
//                               auto min_e = std::min(e1, std::min(e2, e3));
//                               auto tc = max_e;
//                               auto tg = max_e;
//                               auto th = py;
//                               auto thg = py;
//                               auto b0 = tc > threshold_c * e1;
//                               auto b1 = tg > threshold_g * min_e;
//                               auto b2 = th < threshold_h * e1;
//                               auto b3 = thg < threshold_hg * max_e;
//                               auto idx = b0 + b1 * 2 + b2 * 4 + b3 * 8;
//                               auto l = lambda[idx];
//                               auto qx = px * (1 - l) + cx * l;
//                               auto qy = py * (1 - l) + cy * l;
//
//                               // intersection test
//                               auto ttx = qx * sy_neg + sx * qy;
//
//                               bend_left = ttx < 0;
//                               bend_right = ttx > e1 * (qy + sy_neg);
//                           }
//
//                           // case: left out
//                           if (bend_left)
//                           {
//                               dA = +0.0f;
//                               dB = e1 * e1;
//                               dC = e2 * e2;
//
//                               sigma_t += d_s_1;
//                               d_t = sigma_t + d_c_1;
//                           }
//                           // case: right out
//                           else if (bend_right)
//                           {
//                               dA = e1 * e1;
//                               dB = +0.0f;
//                               dC = e3 * e3;
//
//                               sigma_t += d_s_2;
//                               d_t = sigma_t + d_c_2;
//                           }
//                           // case: proper intersection
//                           else
//                           {
//                               dA = d1_sqr;
//                               dB = d2_sqr;
//                               dC = pow2(px - sx) + pow2(py + sy_neg);
//
//                               // fix signs
//                               assert(!(next_h2 & behind_mask));
//                               assert(!(next_h3 & behind_mask));
//
//                               if (py * sx + px * sy_neg < 0)
//                                   next_h2 |= behind_mask;
//                               if (py * (e1 - sx) - (px - e1) * sy_neg < 0)
//                                   next_h3 |= behind_mask;
//
//                               // sigma_t already correct
//                               d_t = sigma_t + std::sqrt(pow2(cx - sx) + pow2(cy + sy_neg));
//                           }
//                       }
//
//                       // Check if better
//                       if (d_t < ref_d_t)
//                       {
//                           ++updates;
//
//                           face_to_center_distance[size_t(f_target)] = d_t;
//                           face_to_extra_distance[size_t(f_target)] = sigma_t;
//
//                           // signs already fixed before
//                           halfedge_to_vertex_distance_sqr[size_t(h1)] = dB;
//                           halfedge_to_vertex_distance_sqr[size_t(h2)] = dA;
//                           halfedge_to_vertex_distance_sqr[size_t(h3)] = dC;
//
//                           // propagate
//                           auto insert_from = d_t < iteration;
//                           auto& next_queue = insert_from ? queue_from : queue_to;
//                           next_queue.push_back(h2);
//                           next_queue.push_back(h3);
//                       }
//                   }
//
//                   std::swap(queue_from, queue_to);
//               }
//
//               auto ig_end = std::chrono::system_clock::now();
//               cerr << "GSP computed geodesics in " << std::chrono::duration<double>(ig_end - ig_start).count() * 1000 << " ms" << endl;
//               cerr << "  .. " << iteration << " iterations" << endl;
//               cerr << "  .. " << expansions << " halfedge expansions" << endl;
//               cerr << "  .. " << updates << " triangle updates" << endl;
//
//               // output geodesics
//               std::vector<float> min_vertex_dis(v_cnt, std::numeric_limits<float>::max());
//
//              for (size_t index = 0; index < passed_halfedge.size(); ++index)
//              {
//                  auto h = passed_halfedge[index];
//                  auto v = halfedge_to_vertex[h];
//                  auto f = halfedge_to_face[h];
//
//    //              std::cout<<"v:\t"<<v<<std::endl;
//    //              std::cout<<"h:\t"<<h<<std::endl;
//    //              std::cout<<"f:\t"<<f<<std::endl;
//
//                  if (f == -1)
//                      continue; // border
//
//                  auto sigma = face_to_extra_distance[size_t(f)];
//                  auto v_sqr = halfedge_to_vertex_distance_sqr[size_t(h)];
//
//                  if (v_sqr < 0)
//                      continue; // not reached
//
//                  auto dis = sigma + std::sqrt(v_sqr);
//                  min_vertex_dis[size_t(v)] = std::min(dis, min_vertex_dis[size_t(v)]);
//              }
//
//               double sum = 0;
//
//               for (size_t index = 0; index < passed_vector.size(); ++index){
//                  cout << "f:\t"<<passed_face[index]<<"\tdis:\t"<<min_vertex_dis[passed_vector[index]] * avg_edge_length <<std::endl;
//                   sum+=min_vertex_dis[passed_vector[index]] * avg_edge_length;
//
//               }
//
//               std::cout<<"sum:\t"<<sum<<std::endl;
//
//            }
//
//           return 0;
//
//    }
//
//}
//
//
//#endif /* geodesic_h */
