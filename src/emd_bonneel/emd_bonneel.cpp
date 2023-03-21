//// Fast Network Simplex for optimal mass transport -- test unit
//// by Nicolas Bonneel (Nov. 2013)
///
//// Typical runtime: 10k source nodes + 10k destination nodes, OpenMP enabled, 6 cores, 12 threads :
///  -  double precision: 7.8 seconds using 1.6GB of RAM (or 7.2 seconds with 2.4 GB of RAM without #define SPARSE_FLOW)
///  -  single precision: 3.5 seconds using 1.2GB of RAM
///  -  int : 5 seconds using 1.2 GB of RAM


#include <iostream>
#include <vector>

#include "network_simplex_simple.h"

using namespace lemon;

// all types should be signed
typedef int64_t arc_id_type; // {short, int, int64_t} ; Should be able to handle (n1*n2+n1+n2) with n1 and n2 the number of nodes (INT_MAX = 46340^2, I64_MAX = 3037000500^2)
typedef double supply_type; // {float, double, int, int64_t} ; Should be able to handle the sum of supplies and *should be signed* (a demand is a negative supply)
typedef double cost_type;  // {float, double, int, int64_t} ; Should be able to handle (number of arcs * maximum cost) and *should be signed* 
                           
struct emd_result {
  cost_type total_cost;
  int status;
};


  struct TsFlow {
    int from, to;
    double amount;
  };


extern "C" emd_result emd_bonneel(supply_type *a, supply_type *b, cost_type *c, arc_id_type n1, arc_id_type n2, size_t max_iter) {

  typedef FullBipartiteDigraph Digraph;
  // DIGRAPH_TYPEDEFS(FullBipartiteDigraph);

  std::vector<supply_type> va(n1), vb(n2);

  Digraph di(n1, n2);
  NetworkSimplexSimple<Digraph, supply_type, cost_type, arc_id_type> net(di, true, n1 + n2, n1 * n2, max_iter);

  arc_id_type idarc = 0;
  for (int i = 0; i < n1; i++) {
    for (int j = 0; j < n2; j++) {
      cost_type d = c[i + n1 * j];
      net.setCost(di.arcFromId(idarc), d);
      idarc++;
    }
  }

  for (int i = 0; i < n1; i++)
    va[di.nodeFromId(i)] = a[i];

  for (int j = 0; j < n2; j++)
    vb[di.nodeFromId(j)] = -b[j];

  net.supplyMap(&va[0], n1, &vb[0], n2);

  // run the algorithm
  int ret = net.run();

  // some rudimentary error checking
  if (ret == (int) net.OPTIMAL) ret = 0;
  else if (ret == (int) net.UNBOUNDED) ret = 1;
  else if (ret == (int) net.INFEASIBLE) ret = 2;
  else if (ret == (int) net.MAX_ITER_REACHED) ret = 3; 
  else ret = -1;


//  emd_result r = ;
  return {net.totalCost(), ret};

}




extern "C" emd_result emd_bonneel_with_plan(supply_type *a, supply_type *b, cost_type *c, double *Tplan, arc_id_type n1, arc_id_type n2, size_t max_iter) {

  typedef FullBipartiteDigraph Digraph;
  // DIGRAPH_TYPEDEFS(FullBipartiteDigraph);

  std::vector<supply_type> va(n1), vb(n2);

  Digraph di(n1, n2);
  NetworkSimplexSimple<Digraph, supply_type, cost_type, arc_id_type> net(di, true, n1 + n2, n1 * n2, max_iter);

  arc_id_type idarc = 0;
  for (int i = 0; i < n1; i++) {
    for (int j = 0; j < n2; j++) {
      cost_type d = c[i + n1 * j];
      net.setCost(di.arcFromId(idarc), d);
      idarc++;
    }
  }

  for (int i = 0; i < n1; i++)
    va[di.nodeFromId(i)] = a[i];

  for (int j = 0; j < n2; j++)
    vb[di.nodeFromId(j)] = -b[j];

  net.supplyMap(&va[0], n1, &vb[0], n2);

  // run the algorithm
  int ret = net.run();

  std::vector<TsFlow> flow;
  flow.reserve(n1 + n2 - 1);
  
  int count=0;
  for (int64_t j = 0; j < n2; j++) {
    for (int64_t i = 0; i < n1; i++) 
 
    {
      TsFlow f;
      f.amount = net.flow(di.arcFromId(i*n2 + j));
      Tplan[count]=f.amount;
      count+=1;
    }
  }


  // some rudimentary error checking
  if (ret == (int) net.OPTIMAL) ret = 0;
  else if (ret == (int) net.UNBOUNDED) ret = 1;
  else if (ret == (int) net.INFEASIBLE) ret = 2;
  else if (ret == (int) net.MAX_ITER_REACHED) ret = 3; 
  else ret = -1;


//  emd_result r = ;
  return {net.totalCost(), ret};

}





