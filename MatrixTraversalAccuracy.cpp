
// build: cl /O2 /openmp /std:c++17 /EHsc MatrixTraversalAccuracy.cpp /Fe:MatrixTraversalAccuracy.exe
#include <thread>
#include <cstdint>
#include <omp.h>
#include <stdio.h>
#include <time.h>
#include <vector>
#include <queue>
#include <iostream>

uint64_t traverseCost();
uint32_t N=1;
uint32_t A[10];
uint32_t B[10];
int T=1;

int main() {
    int threads=std::thread::hardware_concurrency()/4;
    uint64_t val = 102022661;
    for (int i = 0; i < 10; i++) {
        A[i] = val;
        val = (val * val) % 998388889; //population formula same as in the Euler Problem
        B[i] = val;
        val = (val * val) % 998388889;
    }
    printf("For N=1:  Expected:966774091 Got:%llu \n",traverseCost());
    if (threads>1) T=2;
    N=2;
    printf("For N=2:  Expected:2388327490 Got:%llu \n",traverseCost());
    if (threads>4) T=5;
    if (threads>9) T=10;
    N=10;
    printf("For N=10: Expected:13389278727 Got:%llu \n",traverseCost());
}

uint64_t traverseCost(){

    std::vector<std::vector<uint32_t>> mina(T), posa(T), minar(T), posar(T),minb(T), posb(T),minbr(T), posbr(T);
    uint64_t base=0;
    #pragma omp parallel for schedule(static)//compiler does not support collapse
    for (int qt = 0; qt < 4 * T; qt++) {
        int q = qt / T;
        int t = qt % T;
        std::vector<uint32_t>* min = nullptr;
        std::vector<uint32_t>* pos = nullptr;
        uint32_t m; uint64_t sum;
        switch (q) {
        ///////////////////////////////////////////////////////////////////
        case 0: min = &mina[t];  pos = &posa[t];  
        m=A[t*N/T]; sum=m; *min={m}; *pos={t*N/T};
        for (int i=t*N/T+1; i<(t+1)*N/T; i++){
            sum+=A[i];
            if(A[i]<m){ 
		        m=A[i];
		        min->push_back(m);
                pos->push_back(i);
        }}
        #pragma omp atomic
        base+=sum;
        break;
        ///////////////////////////////////////////////////////////////////
        case 1: min = &minar[t];  pos = &posar[t];
        m=A[N-1-t*N/T]; *min={m}; *pos={t*N/T};
        for (int i=t*N/T+1; i<(t+1)*N/T; i++){if(A[N-1-i]<m){ 
		        m=A[N-1-i];
		        min->push_back(m);
                pos->push_back(i);
        }}
        break;
        ///////////////////////////////////////////////////////////////////
        case 2: min = &minb[t];  pos = &posb[t];  
        m=B[t*N/T]; sum=m; *min={m}; *pos={t*N/T};
        for (int i=t*N/T+1; i<(t+1)*N/T; i++){
            sum+=B[i];
            if(B[i]<m){ 
		        m=B[i];
		        min->push_back(m);
                pos->push_back(i);
        }}
        #pragma omp atomic
        base+=sum;
        break;
        ///////////////////////////////////////////////////////////////////
        case 3: min = &minbr[t];  pos = &posbr[t];
        m=B[N-1-t*N/T]; *min={m}; *pos={t*N/T};
        for (int i=t*N/T+1; i<(t+1)*N/T; i++){if(B[N-1-i]<m){
		    m=B[N-1-i];
		    min->push_back(m);
            pos->push_back(i);
        }}
        break;
        ///////////////////////////////////////////////////////////////////
        }}

    std::vector<uint32_t> costa, dista, costar, distar, costb, distb, costbr, distbr;
    #pragma omp parallel for schedule(static)
    for (int q = 0; q < 4; q++) {
        std::vector<std::vector<uint32_t>*> mins(T), poss(T);
        std::vector<uint32_t> *cost, *dist, *pos;
        bool forward;
        switch (q) {
        case 0: for(int t=0;t<T;t++){mins[t]=&mina[t];  poss[t]=&posa[t];}  cost=&costa;  dist=&dista; forward=true; break;
        case 1: for(int t=0;t<T;t++){mins[t]=&minar[t]; poss[t]=&posar[t];} cost=&costar; dist=&distar; forward=false; break;
        case 2: for(int t=0;t<T;t++){mins[t]=&minb[t];  poss[t]=&posb[t];}  cost=&costb;  dist=&distb; forward=true; break;
        case 3: for(int t=0;t<T;t++){mins[t]=&minbr[t]; poss[t]=&posbr[t];} cost=&costbr; dist=&distbr; forward=false; break;
        }
        uint32_t m=(*mins[0])[0];
        uint32_t p=(*poss[0])[0];
        cost->push_back(m);
        int i=1;
        for (int t = 0; t < T; t++) {    
            int s=mins[t]->size();    
            while (i<s && (*mins[t])[i] >= m) i++;
            while (i<s){
                m = (*mins[t])[i];
                uint32_t P = (*poss[t])[i];
                cost->push_back(m);
                dist->push_back(P-p);
                p=P;
                i++;
            }
            i=0;
            mins[t]->clear(); mins[t]->shrink_to_fit(); poss[t]->clear(); poss[t]->shrink_to_fit();
        }
        //cost->pop_back();//maybe include this line. 
        for (auto& c : *cost) c-=m;
        if(forward){
            #pragma omp atomic
            base+=(uint64_t)m*(N-1);
    }}
    mina.clear(); mina.shrink_to_fit(); posa.clear(); posa.shrink_to_fit(); minar.clear(); minar.shrink_to_fit(); posar.clear(); posar.shrink_to_fit();
    minb.clear(); minb.shrink_to_fit(); posb.clear(); posb.shrink_to_fit(); minbr.clear(); minbr.shrink_to_fit(); posbr.clear(); posbr.shrink_to_fit();

    #pragma omp parallel for schedule(static)   //uncomment
    for (int q = 0; q < 2; q++) {
        std::vector<uint32_t> *ca,*cb,*da,*db;
        switch (q) {
        case 0: ca=&costa; cb=&costb; da=&dista; db=&distb;break; 
        case 1: ca=&costar; cb=&costbr; da=&distar; db=&distbr;break;
        }
        int a = 0, b = 0;
        const int A = ca->size(), B = cb->size();
        uint64_t cost = 0;
        using State = std::tuple<uint64_t, int, int>;
        std::priority_queue<State, std::vector<State>, std::greater<State>> pq;
        std::vector<std::vector<uint64_t>> dist(A, std::vector<uint64_t>(B, UINT64_MAX));
        dist[0][0] = 0;
        pq.emplace(0ULL, 0, 0);
        while (true) {
            auto [c, a, b] = pq.top(); pq.pop();;
            if (c > dist[a][b]) continue;
            if (a == A-1 || b == B-1) { cost = c; break; }

            if (a+1 < A) {
                uint64_t product = (uint64_t)(*da)[a] * (*cb)[b];
                if (product > UINT64_MAX - c) { printf("OVERFLOW!\n"); exit(1); }
                uint64_t nc = c + product;
                if (nc < dist[a+1][b]) { dist[a+1][b] = nc; pq.emplace(nc, a+1, b); }
            }
            if (b+1 < B) {
                uint64_t product = (uint64_t)(*db)[b] * (*ca)[a];
                if (product > UINT64_MAX - c) { printf("OVERFLOW!\n"); exit(1); }
                uint64_t nc = c + product;
                if (nc < dist[a][b+1]) { dist[a][b+1] = nc; pq.emplace(nc, a, b+1); }
        }}
        ca->clear(); ca->shrink_to_fit(); cb->clear(); cb->shrink_to_fit(); da->clear(); da->shrink_to_fit(); db->clear(); db->shrink_to_fit();
        if (cost > UINT64_MAX - base) { printf("OVERFLOW in base!\n"); exit(1); }
        #pragma omp atomic
        base+=cost;
    }    
    return base;
}