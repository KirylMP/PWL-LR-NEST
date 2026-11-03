#include <algorithm>
#include <array>
#include <cmath>
#include <vector>

namespace {
using V=std::array<double,4>;
using M=std::array<std::array<double,4>,4>;
M eye(){M a{}; for(int j=0;j<4;++j)a[j][j]=1.; return a;}
M mul(const M&a,const M&b){M c{};for(int i=0;i<4;++i)for(int k=0;k<4;++k)for(int j=0;j<4;++j)c[i][j]+=a[i][k]*b[k][j];return c;}
V mul(const M&a,const V&x){V y{};for(int i=0;i<4;++i)for(int j=0;j<4;++j)y[i]+=a[i][j]*x[j];return y;}
M expm(M a,double t){
 double norm=0.;for(int i=0;i<4;++i){double s=0.;for(int j=0;j<4;++j)s+=std::abs(a[i][j]*t);norm=std::max(norm,s);}
 int n=norm>.5?std::max(0,(int)std::ceil(std::log2(norm/.5))):0;
 double scale=t/std::ldexp(1.,n);for(auto&r:a)for(auto&v:r)v*=scale;
 M sum=eye(),term=eye();for(int k=1;k<=32;++k){term=mul(term,a);double largest=0.;for(int i=0;i<4;++i)for(int j=0;j<4;++j){term[i][j]/=k;sum[i][j]+=term[i][j];largest=std::max(largest,std::abs(term[i][j]));}if(largest<2e-17)break;}
 for(int j=0;j<n;++j)sum=mul(sum,sum);return sum;
}
struct Cache {const double* co;const double* knots;double gamma,dt;std::array<M,4> a,half;};
const Cache& get(const double* co,const double* knots,double g,double dt){
 static thread_local std::vector<Cache> cache;
 for(const auto& c:cache)if(c.co==co&&c.knots==knots&&c.gamma==g&&c.dt==dt)return c;
 Cache c{};c.co=co;c.knots=knots;c.gamma=g;c.dt=dt;
 double p=std::pow(g,.7);double b[3]={p/(p+std::pow(.02,.7)),p/(p+std::pow(.5,.7)),p/(p+std::pow(3.,.7))};
 for(int z=0;z<4;++z){M a{};for(int r=0;r<3;++r){
  a[r][0]=co[3+r];a[r][1]=co[6+r];a[r][2]=co[9+r];a[r][3]=co[r];
  for(int k=0;k<3;++k)a[r][3]+=co[(4+k)*3+r]*b[k];
  for(int k=0;k<z;++k){a[r][0]+=co[(7+k)*3+r];a[r][3]-=co[(7+k)*3+r]*knots[k];}
 }c.a[z]=a;c.half[z]=expm(a,dt/2.);}
 cache.push_back(c);return cache.back();
}
}

extern "C" int gchi_analytic_step(const double* co,const double* knots,double gamma,double dt,double* state){
 if(!std::isfinite(gamma)||gamma<0||!std::isfinite(dt)||dt<=0)return -1;
 const Cache& c=get(co,knots,gamma,dt);V y{state[0],state[1],state[2],1.};
 int z=std::upper_bound(knots,knots+3,y[0])-knots;
 if(z>0&&y[0]==knots[z-1]){double dc=0.;for(int k=0;k<4;++k)dc+=c.a[z][0][k]*y[k];if(dc<0)--z;}
 for(int part=0;part<2;++part){double left=dt/2.;int exits=0;
  while(left>1e-15){
   V end=mul(std::abs(left-dt/2.)<1e-15?c.half[z]:expm(c.a[z],left),y);
   int dir=0;double bound=0.;
   if(z<3&&end[0]>knots[z]){dir=1;bound=knots[z];}
   else if(z>0&&end[0]<knots[z-1]){dir=-1;bound=knots[z-1];}
   if(!dir){y=end;break;}
   double lo=0.,hi=left;
   for(int k=0;k<42&&hi-lo>1e-13;++k){double mid=(lo+hi)/2.;if((mul(expm(c.a[z],mid),y)[0]-bound)*dir>=0)hi=mid;else lo=mid;}
   double t=(lo+hi)/2.;y=mul(expm(c.a[z],t),y);y[0]=bound;left-=t;z+=dir;
   if(++exits>16)return -2;
  }
 }
 for(int i=0;i<3;++i){if(!std::isfinite(y[i]))return -3;state[i]=y[i];}return 0;
}
