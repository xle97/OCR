#include<iostream>
#include"beam_search_decode.h"
#include <stdio.h>
#include <stdlib.h>
#include<unordered_map>
using namespace std;

// typedef struct {
//    int ans[10];
// }struct_test,*struct_pointertt;


extern "C"{
    void decode(double *log_prob){
        
        string n2s="$皖沪津渝冀晋蒙辽吉黑苏浙京闽赣鲁豫鄂湘粤桂琼川贵云藏陕甘青宁新ABCDEFGHJKLMNOPQRSTUVWXYZ0123456789";
        int tmp;
        vector<int>ans=beam_search_decode_test(log_prob);
        
        for(int i=0;i<ans.size();i++){
            tmp=ans[i];
            
            if(tmp>0&&tmp<32){  //解码排除了0，所以if不考虑0
                tmp=(tmp-1)*3+1;
                cout<<n2s.substr(tmp,3);
            }
            else{
                tmp=94+(tmp-32);
                cout<<n2s.substr(tmp,1);
            } 
            
        }
        cout<<endl;
           
    }
    // struct_pointertt decode(double *log_prob){
    //     struct_pointertt ret=(struct_pointertt)malloc(sizeof(struct_test));
    //     for (int i=0;i<10;i++){
    //         ret->ans[i]=i;
    //     }
    //     return ret;
    // }
}

