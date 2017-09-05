//
//  ViewController.m
//  RealTimeStyleTranfer
//
//  Created by yanailab on 2016/07/22.
//  Copyright © 2016年 RyosukeTanno. All rights reserved.
//

#import "ViewController.h"
#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>

#import <CoreVideo/CoreVideo.h>
#import <CoreImage/CoreImage.h>
#import <CoreGraphics/CoreGraphics.h>
#import <AVFoundation/AVFoundation.h>
#import <Accelerate/Accelerate.h>
#import <QuartzCore/QuartzCore.h>

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <pthread.h>

#define CODEBOOK 0
int pr=0;

const int max_memory=1e8;
const int max_memory_im2col  =1e9;

float *im2col_buf;

const double thre=1e-8;
const int max_core=2;
const int max_blas=4;
const int ratio[]={1,1,1,1};

int relu_avg=0;

struct tensor {
    int h;
    int w;
    int d;
    
    float *data;
    struct tensor *old;
    struct tensor *old2;
};

typedef struct tensor Ten;

//#define RELU_AVG 1

inline Ten *add(Ten *in1,Ten *in2)
{
    float *ptr1=in1->data;
    float *ptr2=in2->data;
    float *last=ptr1 + in1->w * in1->h * in1->d;
    while(ptr1<last)
        *(ptr1++) += *(ptr2++);
    return in1;
}

Ten *concat(Ten *in1,Ten *in2)
{
    int i=0,j=0;
    float *ptr1=in1->data + in1->w * in1->h * in1->d;
    float *ptr2=in2->data;
    for(j=0;j<in2->d;j++){
        for(i=0;i<in1->w * in1->h;i++)
            *(ptr1++) = *ptr2;
        ptr2++;
    }
    in1->d += in2->d;
    return in1;
}

Ten *unpool(Ten *in)
{
    Ten *out=in->old;
    out->w=in->w*2; out->h=in->h*2; out->d=in->d/4;
    out->old=in;
    float *ptr=out->data;
    int size=in->w*in->h*out->d,i,j;
    float *ptr1=in->data;
    float *ptr2=ptr1 + size;
    float *ptr3=ptr2 + size;
    float *ptr4=ptr3 + size;
    for(j=0;j<in->h*out->d;j++){
        for(i=0;i<in->w;i++){
            *ptr++ = *ptr1++;
            *ptr++ = *ptr2++;
        }
        for(i=0;i<in->w;i++){
            *ptr++ = *ptr3++;
            *ptr++ = *ptr4++;
        }
    }
    return out;
}

inline void relu_bn(float *ptr,int size, float *sc, int sz)
{
    int kn,sn=-1;
    
    for(kn=0;kn<size;kn++){
        if (kn % sz==0) { sn++; }
        *ptr *= sc[sn];
        //    if (pr && kn % sz==0) {
        //       printf("[%d] %.20f %.20f ",sn+off,sc[sn],*ptr);
        //    }
        if (*ptr<0) *ptr=0;
        ptr++;
    }
    //  if (pr) printf("\n");
}

inline void bn0(float *ptr,int size, float *sc, int sz)
{
    int kn,sn=-1;
    
    for(kn=0;kn<size;kn++){
        if (kn % sz==0) { sn++; }
        *ptr *= sc[sn];
        //    if (pr && kn % sz==0) {
        //       printf("[%d] %.20f %.20f ",sn+off,sc[sn],*ptr);
        //    }
        //    if (*ptr<0) *ptr=0;
        ptr++;
    }
    //  if (pr) printf("\n");
}

inline void relu0(float *ptr,int size)
{
    float *ptr0=ptr;
    int kn;
    
    if (relu_avg){
        float db=0;  //db2=0,th;
        for(kn=0;kn<size;kn++){
            if (*ptr<0) *ptr=0;
            db += *ptr;
            ptr++;
            // db2+=*ptr * *ptr++;
        }
        db/=size;
        //db2/=size; db2-=db*db;
        //th=db-sqrt(db2)*0.5;
        for(kn=0;kn<size;kn++){
            if (*ptr0<db) *ptr0=0;
            ptr0++;
        }
    }else{
        for(kn=0;kn<size;kn++){
            if (*ptr<0) *ptr=0;
            ptr++;
        }
    }
}

Ten *relu(Ten *in)
{
    int i;
    float *ptr=in->data;
    for(i=0;i<in->h*in->w*in->d;i++){
        if (*ptr<0) *ptr=0;
        ptr++;
    }
    return in;
}

inline void tanh0(float *ptr,int size)
{
    int kn;
    //    return;
    for(kn=0;kn<size;kn++)
        *ptr++ = (tanh(*ptr)+1)*127.5;
    
    // [63] Tanh 62
    // [64] _ + 1 63
    // [65] _ * 127.5 64
}

Ten *softmax(Ten *in)
{
    int i,idx=0;
    double sum=0,old,maxv=0;
    float *ptr=in->data;
    for(i=0;i<in->h*in->w*in->d;i++){
        if (*ptr>maxv) { maxv=*ptr; idx=i; }
        ptr++;
    }
    maxv-=30;
    ptr=in->data;
    for(i=0;i<in->h*in->w*in->d;i++){
        *ptr -= maxv;
        old=*ptr;
        *ptr=exp(*ptr);
        //     if (pr && *ptr>1e15) printf("[%d]:%e (%f)\n",i,*ptr,old);
        sum += *ptr++;
    }
    if (pr) printf("softmax maxv:%f (%d) sum:%e [0]:%e, [%d]:%e\n",maxv,idx,sum,in->data[0],idx,in->data[idx]);
    ptr=in->data;
    for(i=0;i<in->h*in->w*in->d;i++)
        *ptr++ /= sum;
    if (pr) printf("softmax sum:%e [0]:%e, [%d]:%e\n",sum,in->data[0],idx,in->data[idx]);
    return in;
}

typedef struct{
    int w1,h1,d1;
    int *size,*pad,*stride;
#if CODEBOOK
    unsigned char *filter;
    float *codebook;
#else
    float *filter;
#endif
    float *bias;
    Ten *in,*out;
    int start_k,end_k;
    int relu;
    float *scale;
    int bn;
} ARG;

void *conv3_thread(void *arg){ // for kernel over 2x2
    ARG *a=(ARG *)arg;
    int sw=a->size[2], sh=a->size[3], sd=a->size[1], k=a->size[0];
    int sss=sw*sh*sd, ss=sw*sh;
    int sz=a->w1*a->h1;
    int off=a->in->h * a->in->w;
    int d,yi,xi,y,x,j,i,kn;
    float *ptr_out[a->end_k],*ptr_in,*ptr_in0;
#if CODEBOOK
    unsigned char *fl[a->end_k];
#else
    float *fl[a->end_k];
#endif
    
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,a->end_k-a->start_k,a->w1*a->h1,sss,
                (float)1.0, a->filter+a->start_k*sss ,sss, im2col_buf, sz, (float)1,
                a->out->data+a->start_k*sz, sz);
    
    if (a->bn && a->relu){
        relu_bn(a->out->data + sz * a->start_k, (a->end_k-a->start_k)*sz, a->scale + a->start_k, sz);
    }else if (a->relu==1){
        relu0(a->out->data + sz * a->start_k, (a->end_k-a->start_k)*sz);
#if 0
        float *ptr=a->out->data + sz * a->start_k;
        for(kn=0;kn<(a->end_k-a->start_k)*sz;kn++){
            if (*ptr<0) *ptr=0;
            ptr++;
        }
#endif
    }else if (a->bn && ! a->relu){
        bn0(a->out->data + sz * a->start_k, (a->end_k-a->start_k)*sz, a->scale + a->start_k, sz);
    }else if (a->relu==2){
        tanh0(a->out->data + sz * a->start_k, (a->end_k-a->start_k)*sz);
    }
    return 0;
}

#if 0
----                     -----
|K   | 1             *   | im  | ss   =  1*(w1*h1)
|    |                   |     |
----                     -----
sss=(sw*sh*1)     (conv_out_spatial_dim)=(w1*h1)=sz
//  cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans,
//  3,3,2,1,A, 3, B, 3,2,C,3);
// C = \alpha A*B + \beta C
// A: m*p
// B: p*n
// C: m*n
//  cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans,m,n,p,alpha,A,p,B,n,beta,C,n);
#endif

void *conv3_s1_thread(void *arg){ // for kernel nxnx1
    ARG *a=(ARG *)arg;
    
    int sw=a->size[2], sh=a->size[3], sd=a->size[1], k=a->size[0];
    int sss=sw*sh*sd, ss=sw*sh;
    int sz=a->w1*a->h1;
    int off=a->in->h * a->in->w;
    int yi,xi,y,x,j,i,kn,g;
    float *ptr_out,*ptr_in,*ptr_in0;
#if CODEBOOK
    unsigned char *fl;
#else
    float *fl;
#endif
    for (g = a->start_k; g < a->end_k; ++g) {
        //    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, 1 , sz ,ss,
        //    (float)1.0, a->filter + g*ss ,ss, im2col_buf + g*sz*ss, sz, (float)0,
        //    a->out->data + g*sz, sz);
        cblas_sgemv(CblasRowMajor, CblasTrans, ss, sz, (float)1.0,
                    im2col_buf + g*sz*ss, sz, a->filter + g*ss , 1, (float)1.0,
                    a->out->data + g*sz, 1);
    }
    
    if (a->bn && a->relu){
        relu_bn(a->out->data + sz * a->start_k, (a->end_k-a->start_k)*sz, a->scale + a->start_k, sz);
    }else if (a->relu==1){
        relu0(a->out->data + sz * a->start_k, (a->end_k-a->start_k)*sz);
#if 0
        float *ptr=a->out->data + sz * a->start_k;
        for(kn=0;kn<(a->end_k-a->start_k)*sz;kn++){
            if (*ptr<0) *ptr=0;
            ptr++;
        }
#endif
    }else if (a->bn && ! a->relu){
        bn0(a->out->data + sz * a->start_k, (a->end_k-a->start_k)*sz, a->scale + a->start_k, sz);
    }else if (a->relu==2){
        tanh0(a->out->data + sz * a->start_k, (a->end_k-a->start_k)*sz);
    }
    return 0;
}

void *conv3_1x1_thread(void *arg){ // for only kernel 1x1
    ARG *a=(ARG *)arg;
    int sw=a->size[2], sh=a->size[3], sd=a->size[1], k=a->size[0];
    int sz=a->w1*a->h1;
    int off=a->in->h * a->in->w;
    int d,yi,xi,y,x,j,i,kn;
    float *ptr_out[a->end_k],*ptr_in,*ptr_in0;
#if CODEBOOK
    unsigned char *fl[a->end_k];
#else
    float *fl[a->end_k];
#endif
    
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,a->end_k-a->start_k,sz,sd,
                (float)1.0,   a->filter+a->start_k*sd ,sd, a->in->data, sz, (float)1.0,
                a->out->data+a->start_k*sz, sz);
    
    if (a->bn && a->relu){
        relu_bn(a->out->data + sz * a->start_k, (a->end_k-a->start_k)*sz, a->scale + a->start_k, sz);
    }else if (a->relu==1){
        relu0(a->out->data + sz * a->start_k, (a->end_k-a->start_k)*sz);
#if 0
        float *ptr=a->out->data + sz * a->start_k;
        for(kn=0;kn<(a->end_k-a->start_k)*sz;kn++){
            if (*ptr<0) *ptr=0;
            ptr++;
        }
#endif
    }else if (a->bn && ! a->relu){
        bn0(a->out->data + sz * a->start_k, (a->end_k-a->start_k)*sz, a->scale + a->start_k, sz);
    }else if (a->relu==2){
        tanh0(a->out->data + sz * a->start_k, (a->end_k-a->start_k)*sz);
    }
    return 0;
}

void *conv3_thread_old(void *arg){ // for kernel over 2x2
    ARG *a=(ARG *)arg;
    
    int sw=a->size[2], sh=a->size[3], sd=a->size[1], k=a->size[0];
    int sss=sw*sh*sd, ss=sw*sh;
    int sz=a->w1*a->h1;
    int off=a->in->h * a->in->w;
    int d,yi,xi,y,x,j,i,kn;
    float *ptr_out[a->end_k],*ptr_in,*ptr_in0;
#if CODEBOOK
    unsigned char *fl[a->end_k];
#else
    float *fl[a->end_k];
#endif
    int ok=0,ng=0;
    
    for(d=0;d<sd;d++){
        ptr_in = a->in->data + off * d;
        for(kn=a->start_k;kn<a->end_k;kn++){
            fl[kn]=&(a->filter[sss*kn + ss*d]);
            ptr_out[kn] = a->out->data + sz * kn - 1;
        }
        for(yi=0;yi<a->h1;yi++){
            y=a->stride[0]*yi-a->pad[0];
            for(xi=0;xi<a->w1;xi++){
                x=a->stride[1]*xi-a->pad[2];
                for(kn=a->start_k;kn<a->end_k;kn++){
                    ptr_out[kn]++;
                    if (!d) *ptr_out[kn] = a->bias[kn];
                }
                int of=0;
                for(j=0;j<sh;j++){
                    ptr_in0=&ptr_in[x+(y+j) * a->in->w];
                    for(i=0;i<sw;i++){
                        if (x+i>=0 && x+i<a->in->w && y+j>=0 && y+j<a->in->h){
                            if (*ptr_in0>thre) ok++; else ng++;
                            if (*ptr_in0>thre || *ptr_in0<-thre){
                                for(kn=a->start_k;kn<a->end_k;kn++)
#if CODEBOOK
                                    *ptr_out[kn] += *ptr_in0 * a->codebook[*(fl[kn]+of)];
#else
                                *ptr_out[kn] += *ptr_in0 * *(fl[kn]+of);
                                //           THDoubleVector_add(ptr_out[kn],fl[kn]+of,*ptr_in0);
                                // sz->kn  -- modify --> kn->sz..ptr_out§¨œ¢¬≥§À§ §Î°•
                                // fl[kn] --> kn -> ss*d -> fl[kn] §¨œ¢¬≥§À§ §Î°•
                                
#endif
                            }
                        }
                        ptr_in0++;
                        of++;
                    }
                }
            }
        }
    }
    
    //  float *pt = ptr_out[a->start_k];
    //  printf("%.5f %.5f (%d,%d)\n",*pt,pt[1],ok,ng);
    
    if (a->bn && a->relu){
        relu_bn(a->out->data + sz * a->start_k, (a->end_k-a->start_k)*sz, a->scale + a->start_k, sz);
    }else if (a->relu==1){
        relu0(a->out->data + sz * a->start_k, (a->end_k-a->start_k)*sz);
#if 0
        float *ptr=a->out->data + sz * a->start_k;
        for(kn=0;kn<(a->end_k-a->start_k)*sz;kn++){
            if (*ptr<0) *ptr=0;
            ptr++;
        }
#endif
    }else if (a->bn && ! a->relu){
        bn0(a->out->data + sz * a->start_k, (a->end_k-a->start_k)*sz, a->scale + a->start_k, sz);
    }else if (a->relu==2){
        tanh0(a->out->data + sz * a->start_k, (a->end_k-a->start_k)*sz);
    }
    return 0;
}

void *conv3_s1_thread_old(void *arg){ // for kernel nxnx1
    ARG *a=(ARG *)arg;
    
    int sw=a->size[2], sh=a->size[3], sd=a->size[1], k=a->size[0];
    int sss=sw*sh*sd, ss=sw*sh;
    int sz=a->w1*a->h1;
    int off=a->in->h * a->in->w;
    int yi,xi,y,x,j,i,kn;
    float *ptr_out,*ptr_in,*ptr_in0;
#if CODEBOOK
    unsigned char *fl;
#else
    float *fl;
#endif
    
    //for(d=0;d<sd;d++){  sd=1
    //  ptr_in = a->in->data + off * d;
    for(kn=a->start_k;kn<a->end_k;kn++){
        fl=&(a->filter[sss*kn]);
        ptr_in = a->in->data + off * kn;
        ptr_out= a->out->data + sz * kn - 1;
        for(yi=0;yi<a->h1;yi++){
            y=a->stride[0]*yi-a->pad[0];
            for(xi=0;xi<a->w1;xi++){
                x=a->stride[1]*xi-a->pad[2];
                ptr_out++;
                *ptr_out = a->bias[kn];
                //      if (!kn) printf("(%d,%d)[%d,%d]",xi,yi,x,y);
                int of=0;
                for(j=0;j<sh;j++){
                    ptr_in0=&ptr_in[x + (y+j)*a->in->w];
                    for(i=0;i<sw;i++){
                        if (x+i>=0 && x+i<a->in->w && y+j>=0 && y+j<a->in->h){
#if CODEBOOK
                            *ptr_out += *ptr_in0 * a->codebook[*(fl+of)];
#else
                            *ptr_out += *ptr_in0 * *(fl+of);
#endif
                        }
                        ptr_in0++;
                        of++;
                    }
                }
            }
        }
    }
    
    if (a->bn && a->relu){
        relu_bn(a->out->data + sz * a->start_k, (a->end_k-a->start_k)*sz, a->scale + a->start_k, sz);
    }else if (a->relu==1){
        relu0(a->out->data + sz * a->start_k, (a->end_k-a->start_k)*sz);
#if 0
        float *ptr=a->out->data + sz * a->start_k;
        for(kn=0;kn<(a->end_k-a->start_k)*sz;kn++){
            if (*ptr<0) *ptr=0;
            ptr++;
        }
#endif
    }else if (a->bn && ! a->relu){
        bn0(a->out->data + sz * a->start_k, (a->end_k-a->start_k)*sz, a->scale + a->start_k, sz);
    }else if (a->relu==2){
        tanh0(a->out->data + sz * a->start_k, (a->end_k-a->start_k)*sz);
    }
    return 0;
}

void *conv3_1x1_thread_old(void *arg){ // for only kernel 1x1
    ARG *a=(ARG *)arg;
    
    int sw=a->size[2], sh=a->size[3], sd=a->size[1], k=a->size[0];
    int sz=a->w1*a->h1;
    int off=a->in->h * a->in->w;
    int d,yi,xi,y,x,j,i,kn;
    float *ptr_out[a->end_k],*ptr_in,*ptr_in0;
#if CODEBOOK
    unsigned char *fl[a->end_k];
#else
    float *fl[a->end_k];
#endif
    int ok=0,ng=0;
    
    ptr_in = a->in->data;
    for(d=0;d<sd;d++){
        //ptr_in = a->in->data + off * d;
        for(kn=a->start_k;kn<a->end_k;kn++){
            fl[kn]=&(a->filter[sd*kn + d]);
            if (!d) ptr_out[kn] = a->out->data + sz * kn;
        }
        int of=0;
        for(y=0;y<a->h1;y++){
            for(x=0;x<a->w1;x++){
                if (!d)
                    for(kn=a->start_k;kn<a->end_k;kn++)
                        ptr_out[kn][of] = a->bias[kn];
                if (*ptr_in>thre) ok++; else ng++;
                if (*ptr_in>thre || *ptr_in<-thre) {
                    for(kn=a->start_k;kn<a->end_k;kn++)
#if CODEBOOK
                        ptr_out[kn][of] += *ptr_in * a->codebook[*fl[kn]];
#else
                    ptr_out[kn][of] += *ptr_in * *fl[kn];
#endif
                }
                of++;
                ptr_in++;
            }
        }
    }
    //  float *pt = ptr_out[a->start_k];
    //  printf("%.5f %.5f (%d,%d)\n",*pt,pt[1],ok,ng);
    
    if (a->bn && a->relu){
        relu_bn(a->out->data + sz * a->start_k, (a->end_k-a->start_k)*sz, a->scale + a->start_k, sz);
    }else if (a->relu==1){
        relu0(a->out->data + sz * a->start_k, (a->end_k-a->start_k)*sz);
#if 0
        float *ptr=a->out->data + sz * a->start_k;
        for(kn=0;kn<(a->end_k-a->start_k)*sz;kn++){
            if (*ptr<0) *ptr=0;
            ptr++;
        }
#endif
    }else if (a->bn && ! a->relu){
        bn0(a->out->data + sz * a->start_k, (a->end_k-a->start_k)*sz, a->scale + a->start_k, sz);
    }else if (a->relu==2){
        tanh0(a->out->data + sz * a->start_k, (a->end_k-a->start_k)*sz);
    }
    return 0;
}

void *conv3_gemm(void *arg){ // for only kernel 1x1
    ARG *a=(ARG *)arg;
    int sw=a->size[2], sh=a->size[3], sd=a->size[1], k=a->size[0];
    int sss=sw*sh*sd, ss=sw*sh;
    int sz=a->w1*a->h1;
    int off=a->in->h * a->in->w;
    int d,yi,xi,y,x,j,i;
    //  float *ptr_out[a->end_k],*ptr_in,*ptr_in0;
#if CODEBOOK
    unsigned char *fl[a->end_k];
#else
    float *fl[a->end_k];
#endif
    
#if 0
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,a->end_k-a->start_k,a->w1*a->h1,sss,
                (float)1.0, a->filter+a->start_k*sss ,sss, im2col_buf, sz, (float)0,
                a->out->data+a->start_k*sz, sz);
    ----                     -----
    |K   | #num_kernel   *   | im  | sss     =  k * (w1*h1)
    |    |                   |     |
    ----                     -----
    sss=(sw*sh*sd)     (conv_out_spatial_dim)=(w1*h1)=sz
#endif
    
    int kn=a->end_k - a->start_k;
    
    float *out=a->out->data + a->start_k*sz;
    float *ptr0 = a->filter + a->start_k*sss;
    float *ptr1 = im2col_buf;
    
    for(y=0;y<kn;y++){
        for(x=0;x<sz;x++){
            *out=0;
            for(i=0;i<sss;i++){
                *out += ptr0[i] * ptr1[sz*i];
            }
            out++;
        }
        ptr0+=sss;
        ptr1++;
    }
    
    if (a->bn && a->relu){
        relu_bn(a->out->data + sz * a->start_k, (a->end_k-a->start_k)*sz, a->scale + a->start_k, sz);
    }else if (a->relu==1){
        relu0(a->out->data + sz * a->start_k, (a->end_k-a->start_k)*sz);
#if 0
        float *ptr=a->out->data + sz * a->start_k;
        for(kn=0;kn<(a->end_k-a->start_k)*sz;kn++){
            if (*ptr<0) *ptr=0;
            ptr++;
        }
#endif
    }else if (a->bn && ! a->relu){
        bn0(a->out->data + sz * a->start_k, (a->end_k-a->start_k)*sz, a->scale + a->start_k, sz);
    }else if (a->relu==2){
        tanh0(a->out->data + sz * a->start_k, (a->end_k-a->start_k)*sz);
    }
    return 0;
}

#if 0
cblas_sgemm(CblasColMajor,CblasNoTrans, CblasNoTrans,
            conv_out_channels_, conv_out_spatial_dim_, kernel_dim_,(float)1.,
            weights, col_buff, (float)0., output);

//  cblas_dgemm(CblasColMajor, CblasNoTrans,CblasNoTrans,
//  3,3,2,1,A, 3, B, 3,2,C,3);
// C = \alpha A*B + \beta C
// A: m*p
// B: p*n
// C: m*n
//  cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans,m,n,p,alpha,A,p,B,n,beta,C,n);
cblas_sgemm(CblasColMajor, CblasNoTrans, CblasNoTrans,k,w1*h1,sss,(float)1.0,
            a->filter,sss, im2col_buf, w1*h1, (float)0, out->data, w1*h1);

//  cblas_dgemm(CblasColMajor, CblasNoTrans, CblasTrans,m,n,p,alpha,A,p,B,n,beta,C,n);
//  cblas_sgemm(CblasColMajor, CblasNoTrans, CblasTrans,m,n,p,alpha,A,p,B,n,beta,C,n);

----                     -----
| K  | #num_kernel   *   | im  | sss     =  k * (w1*h1)
|    |                   |     |
----                     -----
sss=(sw*sh*sd)     (conv_out_spatial_dim)=(w1*h1)
#endif


void im2col(Ten *in,int sw,int sh,int sd,int w1,int h1,int *stride,int *pad)
{
    int c,h,w;
    for (c = 0; c < sw*sh*sd; ++c) {
        int w_offset = c % sw;
        int h_offset = (c / sw) % sh;
        int c_im = c / sh / sw;
        for (h = 0; h < h1; ++h) {
            for (w = 0; w < w1; ++w) {
                int h_pad = h * stride[0] - pad[0] + h_offset;
                int w_pad = w * stride[1] - pad[2] + w_offset;
                if (h_pad >= 0 && h_pad < in->h && w_pad >= 0 && w_pad < in->w)
                    im2col_buf[(c * h1 + h) * w1 + w] =
                    in->data[(c_im * in->h + h_pad) * in->w + w_pad];
                else
                    im2col_buf[(c * h1 + h) * w1 + w] = 0;
            }
        }
    }
}


///////////////////////////////////////////////////////////////////////////////
/*
 conv.get_deconv_outsize(h, kh, self.sy, self.ph)
 def get_deconv_outsize(size, k, s, p, cover_all=False):
 if cover_all:
 return s * (size - 1) + k - s + 1 - 2 * p
 else:
 return s * (size - 1) + k - 2 * p
 
 void col2im(Ten *in, int sw,int sh,int sd,int w1,int h1,int *stride,int *pad)
 // Dtype* data_col, const int channels,
 //        const int height, const int width, const int kernel_h, const int kernel_w,
 //        const int pad_h, const int pad_w,
 //        const int stride_h, const int stride_w,
 //        const int dilation_h, const int dilation_w,
 //        Dtype* data_im) {
 const int output_h = (height + 2 * pad_h -
 (dilation_h * (kernel_h - 1) + 1)) / stride_h + 1;
 const int output_w = (width + 2 * pad_w -
 (dilation_w * (kernel_w - 1) + 1)) / stride_w + 1;
 
 const int channel_size = height * width;
 for (int channel = channels; channel--; data_im += channel_size) {
 for (int kernel_row = 0; kernel_row < kernel_h; kernel_row++) {
 for (int kernel_col = 0; kernel_col < kernel_w; kernel_col++) {
 int input_row = -pad_h + kernel_row * dilation_h;
 for (int output_rows = output_h; output_rows; output_rows--) {
 if (!is_a_ge_zero_and_a_lt_b(input_row, height)) {
 data_col += output_w;
 } else {
 int input_col = -pad_w + kernel_col * dilation_w;
 for (int output_col = output_w; output_col; output_col--) {
 if (is_a_ge_zero_and_a_lt_b(input_col, width)) {
 data_im[input_row * width + input_col] += *data_col;
 }
 data_col++;
 input_col += stride_w;
 }
 }
 input_row += stride_h;
 }
 }
 }
 }
 }
 
 template <typename Dtype>
 void BaseConvolutionLayer<Dtype>::backward_cpu_gemm(const Dtype* output,
 const Dtype* weights, Dtype* input) {
 Dtype* col_buff = col_buffer_.mutable_cpu_data();
 if (is_1x1_) {
 col_buff = input;
 }
 for (int g = 0; g < group_; ++g) {
 caffe_cpu_gemm<Dtype>(CblasTrans, CblasNoTrans, kernel_dim_,
 conv_out_spatial_dim_, conv_out_channels_ / group_,
 (Dtype)1., weights + weight_offset_ * g, output + output_offset_ * g,
 (Dtype)0., col_buff + col_offset_ * g);
 }
 if (!is_1x1_) {
 conv_col2im_cpu(col_buff, input);
 }
 }
 
 
 template <typename Dtype>
 void BaseConvolutionLayer<Dtype>::weight_cpu_gemm(const Dtype* input,
 const Dtype* output, Dtype* weights) {
 const Dtype* col_buff = input;
 if (!is_1x1_) {
 conv_im2col_cpu(input, col_buffer_.mutable_cpu_data());
 col_buff = col_buffer_.cpu_data();
 }
 for (int g = 0; g < group_; ++g) {
 caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasTrans, conv_out_channels_ / group_,
 kernel_dim_, conv_out_spatial_dim_,
 (Dtype)1., output + output_offset_ * g, col_buff + col_offset_ * g,
 (Dtype)1., weights + weight_offset_ * g);
 }
 }
 template <typename Dtype>
 void BaseConvolutionLayer<Dtype>::forward_cpu_bias(Dtype* output,
 const Dtype* bias) {
 caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, num_output_,
 out_spatial_dim_, 1, (Dtype)1., bias, bias_multiplier_.cpu_data(),
 (Dtype)1., output);
 }
 
 template <typename Dtype>
 void BaseConvolutionLayer<Dtype>::backward_cpu_bias(Dtype* bias,
 const Dtype* input) {
 caffe_cpu_gemv<Dtype>(CblasNoTrans, num_output_, out_spatial_dim_, 1.,
 input, bias_multiplier_.cpu_data(), 1., bias);
 }
 
 for (int g = 0; g < group_; ++g) {
 caffe_cpu_gemm<Dtype>(CblasTrans, CblasNoTrans, kernel_dim_,
 conv_out_spatial_dim_, conv_out_channels_ / group_,
 (Dtype)1., weights + weight_offset_ * g, output + output_offset_ * g,
 (Dtype)0., col_buff + col_offset_ * g);
 }
 ----------------
 
 if (!is_1x1_) {
 conv_col2im_cpu(col_buff, input);
 
 void DeconvolutionLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
 const vector<Blob<Dtype>*>& top) {
 const Dtype* weight = this->blobs_[0]->cpu_data();
 for (int i = 0; i < bottom.size(); ++i) {
 const Dtype* bottom_data = bottom[i]->cpu_data();
 Dtype* top_data = top[i]->mutable_cpu_data();
 for (int n = 0; n < this->num_; ++n) {
 this->backward_cpu_gemm(bottom_data + n * this->bottom_dim_, weight,
 top_data + n * this->top_dim_);
 if (this->bias_term_) {
 const Dtype* bias = this->blobs_[1]->cpu_data();
 this->forward_cpu_bias(top_data + n * this->top_dim_, bias);
 }
 }
 }
 }
 */
///////////////////////////////////////////////////////////////////////////////

#if CODEBOOK
Ten *conv2(Ten *in,int *size,int *stride,int *pad,unsigned char *filter,float *bias,float *codebook,int relu)
#else
Ten *conv2(Ten *in,int *size,int *stride,int *pad,float *filter,float *bias,float *scale,int relu,int bn)
//Ten *conv2(Ten *in,int *size,int *stride,int *pad,float *filter,float *bias,int relu)
#endif
{
    // stride: y,x
    // pad: top,bottom,left,right
    int sw=size[2], sh=size[3], sd=size[1], k=size[0];
    int w1= (int)((in->w + (pad[2]+pad[3]) - sw)/stride[1])+1;
    int h1= (int)((in->h + (pad[0]+pad[1]) - sh)/stride[0])+1;
    int sss=sw*sh*sd, ss=sw*sh;
    int d1=size[0];
    int off=in->h * in->w;
    int kn,i,j,x,y,d,yi,xi,h,w,c;
    pthread_t thread[max_core];
    
#if DEBUG
    static double num_comp=0, num_weights=0;
    double n_comp = sss*k*w1*h1;
    double n_weights=(sss+1)*k;
    num_comp += n_comp;
    num_weights += n_weights;
#endif
    
    //   int max_h= in->h -sh+1, max_w= in->w -sw+1;
    //   Ten *out=(Ten *)malloc(sizeof(Ten));
    Ten *out=in->old;
    out->w=w1; out->h=h1; out->d=size[0];
    out->old=in;
    //   out->data=(float *)malloc(sizeof(float)*w1*h1*d1);
    //   out->data=in->data_out;
    
    //   printf("in:%x out:%x\n",in->data,out->data);
    
    if (pr) printf("  ### in->data[0]=%.5f in->data[last]=%.5f\n", in->data[0], in->data[in->d*off-1]);
    if (pr) printf("  ### filter[0]=%.5f filter[last]=%.5f\n", filter[0], filter[sss*k-1]);
    
#if DEBUG
    if (max_memory<w1*h1*d1) {
        fprintf(stderr,"Reserved memory is not enough.\n");
        fprintf(stderr,"reserved memory: %dbyte required memory: %dbyte\n",
                max_memory,w1*h1*d1);
        exit(1);
    }
    char *str;
#endif
    bzero(out->data,sizeof(float)*w1*h1*d1);
    //    memset(out->data,  0, sizeof(float)*w1*h1*d1);
    
    if (max_blas){
        float *fp=out->data;
        for(d=0;d<d1;d++)
            for(w=0;w<w1;w++)
                for(h=0;h<h1;h++)
                    *fp++ = bias[d];
    }
    
    
#if CODEBOOK
    ARG arg={w1,h1,d1,size,pad,stride,filter,codebook,bias,in,out,0,0,relu};
#else
    ARG arg={w1,h1,d1,size,pad,stride,filter,bias,in,out,0,0,relu,scale,bn};
#endif
    
#if DEBUG
    //     if (pr) printf("      ");
    if (sw==1 && sh==1 && stride[0]==1 && stride[1]==1 && pad[0]==0 && pad[2]==0) {
        str="Thread_1x1";
    }else if (sd==1) {
        str="Thread_s1";
    }else{
        str="Thread";
    }
    if (pr) printf("comp: %.0f, weights: %.0f\n",n_comp,n_weights);
    if (pr) printf("acc. comp: %.0f, acc. weights: %.0f\n",num_comp,num_weights);
#endif
    
#if NO_THREAD
    im2col(in,sw,sh,sd,w1,h1,stride,pad);
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,k,w1*h1,sss,
                (float)1.0, filter ,sss, im2col_buf, w1*h1, (float)1, out->data, w1*h1);
    
    if (relu==1){
        float *ptr=out->data;
        for(kn=0;kn<k*w1*h1;kn++){
            if (*ptr<0) *ptr=0;
            ptr++;
        }
    }else if (relu==2){
        tanh0(out->data, k*w1*h1);
    }
    return out;
#endif
    int ncore=max_core;
    
    if(k<ncore){
        ncore=k;
    }
    
    ARG a[ncore];
    static int total=0;
    if (!total)
        for(d=0;d<ncore;d++)
            total+=ratio[d];
    for(d=0;d<ncore;d++){
        a[d]=arg;
        a[d].start_k= (d)?a[d-1].end_k:0;
        a[d].end_k  = a[d].start_k + (int)((double)k*ratio[d]/total+0.99);
        a[d].end_k  = (a[d].end_k<k)?a[d].end_k:k;
    }
    
    for(d=0;d<ncore;d++){
#if DEBUG
        if (pr) printf("%s[%d]:%d-%d",str,d+1,a[d].start_k,a[d].end_k);
        if (pr) if (d==ncore-1) printf("\n"); else printf(", ");
#endif
        if (sw==1 && sh==1 && stride[0]==1 && stride[1]==1 && pad[0]==0 && pad[2]==0) {
            if (d<max_blas) pthread_create(&thread[d],NULL,conv3_1x1_thread,(void *)&a[d]);
            else      pthread_create(&thread[d],NULL,conv3_1x1_thread_old,(void *)&a[d]);
        }else if (sd==1) {
            if (max_blas && !d) im2col(in,sw,sh,k,w1,h1,stride,pad);
            if (d<max_blas) pthread_create(&thread[d],NULL,conv3_s1_thread,(void *)&a[d]);
            else      pthread_create(&thread[d],NULL,conv3_s1_thread_old,(void *)&a[d]);
        }else{
            if (max_blas && !d) im2col(in,sw,sh,sd,w1,h1,stride,pad);
            if (d<max_blas) pthread_create(&thread[d],NULL,conv3_thread,(void *)&a[d]);
            else      pthread_create(&thread[d],NULL,conv3_thread_old,(void *)&a[d]);
            //else      pthread_create(&thread[d],NULL,conv3_gemm,(void *)&a[d]);
        }
    }
    
    for(d=0;d<ncore;d++){
        pthread_join(thread[d], NULL);
    }
    //   free(in->data);
    //   free(in);
    
#if 0
    float *ptr=in->data;
    for(i=0;i<in->h*in->w*in->d;i++){
        if (*ptr<0) *ptr=0;
        ptr++;
    }
#endif
    
    return out;
}

Ten *fc(Ten *in,int *size,float *weight,float *bias,float *scale,int relu,int bn)
{
    int sd=size[1], k=size[0], ss;
    int conv_size[4]={k,sd,1,1};
    int stride[2]={1, 1};
    int pad[4]={0,0,0,0};
    
    
    if (in->d != sd) {
        ss=(int)(sqrt(sd / in->d));
        conv_size[3]=ss;
        conv_size[2]=ss;
        conv_size[1]=in->d;
    }
    
    return conv2(in,conv_size,stride,pad,weight,bias,scale,relu,bn);
}

Ten *pool(Ten *in,const char *type,int *size,int *stride,int *pad)
{
    int sw=size[1], sh=size[0];
    // global pooling
    if (sw==0) { sw=in->w; }
    if (sh==0) { sh=in->h; if (pr) printf("global pooling (%d,%d)\n",sw,sh);}
    int w1= (int)((in->w +(pad[2]+pad[3]) -sw)/stride[1])+1;
    int h1= (int)((in->h +(pad[0]+pad[1]) -sh)/stride[0])+1;
    int d1=in->d;
    int d,i,j,x,y,xi,yi;
    int max_h= in->h -sh+1, max_w= in->w -sw+1;
    if (w1<=0) { w1=1; sw=in->w; }
    if (h1<=0) { h1=1; sh=in->h; }
    //   Ten *out=(Ten *)malloc(sizeof(Ten));
    Ten *out=in->old;
    out->w=w1; out->h=h1; out->d=d1;
    //   out->data=(float *)malloc(sizeof(float)*w1*h1*d1);
    //   out->data=in->data_out;
    out->old=in;
#if DEBUG
    if (max_memory<w1*h1*d1) {
        fprintf(stderr,"Reserved memory is not enough.\n");
        fprintf(stderr,"reserved memory: %dbyte required memory: %dbyte\n",
                max_memory,w1*h1*d1);
        exit(1);
    }
#endif
    bzero(out->data,sizeof(float)*w1*h1*d1);
    
    if (!strncasecmp(type,"AV",2)){  // average pooling
#if DEBUG
        if (pr) printf("     Average pooling\n");
#endif
        for(d=0;d<d1;d++){
            float *ptr=out->data + w1*h1*d;
            float *ptr_in=in->data + in->w * in->h * d;
            for(yi=0;yi<h1;yi++){
                y=stride[0]*yi-pad[0];
                for(xi=0;xi<w1;xi++){
                    x=stride[1]*xi-pad[2];
                    *ptr=0; int ct=0;
                    for(j=0;j<sh;j++)
                        for(i=0;i<sw;i++)
                            if (x+i>=0 && x+i<in->w && y+j>=0 && y+j<in->h){
                                *ptr+=ptr_in[x+i+(y+j)* in->w ]; ct++;
                            }
                    *ptr++/=ct;
                }
            }
        }
    }else{  // max pooling
#if DEBUG
        if (pr) printf("     Max pooling\n");
#endif
        for(d=0;d<d1;d++){
            float *ptr=out->data + w1*h1*d;
            float *ptr_in=in->data + in->w * in->h * d;
            for(yi=0;yi<h1;yi++){
                y=stride[0]*yi-pad[0];
                for(xi=0;xi<w1;xi++){
                    x=stride[1]*xi-pad[2];
                    *ptr=-1e30;
                    for(j=0;j<sh;j++)
                        for(i=0;i<sw;i++)
                            if (x+i>=0 && x+i<in->w && y+j>=0 && y+j<in->h)
                                if (*ptr<ptr_in[x+i+(y+j)* in -> w])
                                    *ptr=ptr_in[x+i+(y+j)* in -> w];
                    ptr++;
                }
            }
        }
    }
    //   free(in->data);
    //   free(in);
    return out;
}


//#include "param_nt_161026.h"
#include "080802_style.h" //300: 0.2
//#include "080801_style.h" //


typedef struct{
    int idx;
    float v;
} IDX;

int IDX_sort(const void *a,const void *b)
{
    if (((IDX *)a)->v < ((IDX *)b)->v) return 1;
    else return -1;
}

#if DEBUG
Ten *DCNN_dummy(Ten *in)
{
    Ten *out=(Ten *)malloc(sizeof(Ten));
    out->w=1; out->h=1; out->d=101;
    out->data=(float *)malloc(sizeof(float)*101);
    bzero(out->data,sizeof(float)*101);
    out->data[23]=1.000;
    free(in->data);
    free(in);
    return out;
}
#endif


@interface ViewController ()
@property (strong, nonatomic) AVCaptureDeviceInput *videoInput;
@property (strong, nonatomic) AVCaptureVideoDataOutput *videoDataOutput;
@property (strong, nonatomic) AVCaptureSession *session;
@property (strong, nonatomic) UIImageView *flashView;
@end

@implementation ViewController

- (void)viewDidLoad {
    [super viewDidLoad];

    [self setupAVCapture];
}


// UIImage -> IplImage変換
- (IplImage*)IplImageFromUIImage:(UIImage*)image {
    
    CGImageRef imageRef = image.CGImage;
    CGColorSpaceRef colorSpace = CGColorSpaceCreateDeviceRGB();
    IplImage *iplimage = cvCreateImage(cvSize(image.size.width,image.size.height), IPL_DEPTH_8U, 4);
    
    CGContextRef contextRef = CGBitmapContextCreate(
                                                    iplimage->imageData,
                                                    iplimage->width,
                                                    iplimage->height,
                                                    iplimage->depth,
                                                    iplimage->widthStep,
                                                    colorSpace,
                                                    kCGImageAlphaPremultipliedLast|kCGBitmapByteOrderDefault);
    CGContextDrawImage(contextRef,
                       CGRectMake(0, 0, image.size.width, image.size.height),
                       imageRef);
    
    CGContextRelease(contextRef);
    CGColorSpaceRelease(colorSpace);
    
    IplImage *ret = cvCreateImage(cvGetSize(iplimage), IPL_DEPTH_8U, 3);
    cvCvtColor(iplimage, ret, CV_RGB2BGR);
    cvReleaseImage(&iplimage);
    
    return ret;
}

// IplImage -> UIImage変換
- (UIImage*)UIImageFromIplImage:(IplImage*)image {
    
    CGColorSpaceRef colorSpace;
    if (image->nChannels == 1)
    {
        colorSpace = CGColorSpaceCreateDeviceGray();
    } else {
        colorSpace = CGColorSpaceCreateDeviceRGB();
        //        BGRになっているのでRGBに変換
        //        エンジン側でBGR->RGBしてるのでコメントアウト
        //        cvCvtColor(image, image, CV_BGR2RGB);
    }
    NSData *data = [NSData dataWithBytes:image->imageData length:image->imageSize];
    CGDataProviderRef provider = CGDataProviderCreateWithCFData((__bridge CFDataRef)data);
    CGImageRef imageRef = CGImageCreate(image->width,
                                        image->height,
                                        image->depth,
                                        image->depth * image->nChannels,
                                        image->widthStep,
                                        colorSpace,
                                        kCGImageAlphaNone|kCGBitmapByteOrderDefault,
                                        provider,
                                        NULL,
                                        false,
                                        kCGRenderingIntentDefault
                                        );
    UIImage *ret = [UIImage imageWithCGImage:imageRef];
    
    CGImageRelease(imageRef);
    CGDataProviderRelease(provider);
    CGColorSpaceRelease(colorSpace);
    
    return ret;
}

/*----カメラ切り替え(フロント)----*/
- (AVCaptureDevice *)frontFacingCameraIfAvailable
{
    //  look at all the video devices and get the first one that's on the front
    NSArray *videoDevices = [AVCaptureDevice devicesWithMediaType:AVMediaTypeVideo];
    AVCaptureDevice *captureDevice = nil;
    for (AVCaptureDevice *device in videoDevices)
    {
        if (device.position == AVCaptureDevicePositionFront)
        {
            captureDevice = device;
            break;
        }
    }
    
    //  couldn't find one on the front, so just get the default video device.
    if ( ! captureDevice)
    {
        captureDevice = [AVCaptureDevice defaultDeviceWithMediaType:AVMediaTypeVideo];
    }
    
    return captureDevice;
}
/*----カメラ切り替え(バック)----*/
- (AVCaptureDevice *)backCamera
{
    //  look at all the video devices and get the first one that's on the front
    NSArray *videoDevices = [AVCaptureDevice devicesWithMediaType:AVMediaTypeVideo];
    AVCaptureDevice *captureDevice = nil;
    for (AVCaptureDevice *device in videoDevices)
    {
        if (device.position == AVCaptureDevicePositionBack)
        {
            captureDevice = device;
            break;
        }
    }
    
    //  couldn't find one on the front, so just get the default video device.
    if ( ! captureDevice)
    {
        captureDevice = [AVCaptureDevice defaultDeviceWithMediaType:AVMediaTypeVideo];
    }
    
    return captureDevice;
}

- (void)setupAVCapture
{
    NSError *error = nil;
    
    // 入力と出力からキャプチャーセッションを作成
    self.session = [[AVCaptureSession alloc] init];
    
    self.session.sessionPreset = AVCaptureSessionPresetMedium;
    
    // カメラからの入力を作成
    AVCaptureDevice *camera = [AVCaptureDevice defaultDeviceWithMediaType:AVMediaTypeVideo];

    
    // カメラからの入力を作成し、セッションに追加
    self.videoInput = [AVCaptureDeviceInput deviceInputWithDevice:camera error:&error];
    [self.session addInput:self.videoInput];
    
    // 画像への出力を作成し、セッションに追加
    self.videoDataOutput = [[AVCaptureVideoDataOutput alloc] init];
    [self.session addOutput:self.videoDataOutput];
    
    // ビデオ出力のキャプチャの画像情報のキューを設定
    dispatch_queue_t queue = dispatch_queue_create("myQueue", NULL);
    [self.videoDataOutput setAlwaysDiscardsLateVideoFrames:TRUE];
    [self.videoDataOutput setSampleBufferDelegate:self queue:queue];
    
    // ビデオへの出力の画像は、BGRAで出力
    self.videoDataOutput.videoSettings = @{
                                           (id)kCVPixelBufferPixelFormatTypeKey : [NSNumber numberWithInt:kCVPixelFormatType_32BGRA]
                                           };
    
    // ビデオ入力のAVCaptureConnectionを取得
    AVCaptureConnection *videoConnection = [self.videoDataOutput connectionWithMediaType:AVMediaTypeVideo];
    
    videoConnection.videoMinFrameDuration = CMTimeMake(1,2);
    
    [self.session startRunning];
}

- (void)didReceiveMemoryWarning {
    [super didReceiveMemoryWarning];
    // Dispose of any resources that can be recreated.
}

- (void)captureOutput:(AVCaptureOutput *)captureOutput
didOutputSampleBuffer:(CMSampleBufferRef)sampleBuffer
       fromConnection:(AVCaptureConnection *)connection
{

    UIImage *image = [self imageFromSampleBuffer:sampleBuffer];

    dispatch_async(dispatch_get_main_queue(), ^{

        

        NSDate *start = [NSDate date];
        
        IplImage *im, *im2;
        IplImage *im0 = [self IplImageFromUIImage:image];
        
        //        if (im0 == NULL) {
        //            fprintf(stderr,"file open error!!\n");
        //            exit(1);
        //        }else{
        //            printf("file image OK\n");
        //        }
        
        int i,x,y,sx,sy,offx=0,offy=0,offx2=0,offy2=0,ss,aoff=1;
        int size=0,ssize=0;
        int top=10;
        //         char *wei=NULL;
        
        //         if(argc<2+aoff){
        //             fprintf(stderr,"%s {img-name} {out img-name} {-p} {-w (weights)} {-s (size of long-side)}\n",argv[0]);
        //             return 1;
        //         }
        //         if (argc>=3+aoff && !strcmp(argv[2+aoff],"-p")) { pr=1; aoff++; }
        //         if (argc>=4+aoff && !strcmp(argv[2+aoff],"-w")) {
        //             //    size=atoi(argv[3+aoff]);
        //             wei=argv[3+aoff]; aoff+=2;
        //             fprintf(stderr,"weights:%s\n",wei);
        //         }
        //         if (argc>=4+aoff && !strcmp(argv[2+aoff],"-s")) {
        //             size=atoi(argv[3+aoff]); aoff+=2;
        //             fprintf(stderr,"resize size:%d\n",size);
        //         }
        //
        //
        //         im0=cvLoadImage(argv[1],1);
        //         if (im0 == NULL) {
        //             fprintf(stderr,"file open error!! (%s)\n",argv[1]);
        //             exit(1);
        //         }
        size=400;
        
        //        if (size)
        //            if (im0->width < im0->height){
        //                // if (ssize) sy=ssize; else
        //                sy=size;
        //                sx=sy*im0->width/im0->height;
        //            }else{
        //                // if (ssize) sx=ssize; else
        //                sx=size;
        //                sy=sx*im0->height/im0->width;
        //            }
        //            else{
        //                sx=im0->width;
        //                sy=im0->height;
        //            }
        //        im2 = cvCreateImage(cvSize(sx,sy),IPL_DEPTH_8U, 3);
        //
        //
        //        fprintf(stderr,"%d %d -> %d %d\n",im0->width,im0->height,im->width,im->height);
        //        cvResize(im0,im2,CV_INTER_CUBIC);
        
        //        if (im2 == NULL) {
        //            fprintf(stderr,"file open error!!\n");
        //            exit(1);
        //        }else{
        //            printf("file name im OK\n");
        //        }
        
        //         UIImage *image01=[self UIImageFromIplImage:im];
        //         UIImageWriteToSavedPhotosAlbum(image01,self,nil,nil);
        static Ten *input=NULL;
        static Ten *input2=NULL;
        if (!input){
            input=(Ten *)malloc(sizeof(Ten));
            input->data=(float *)malloc(sizeof(float)*max_memory);
            input->old=(Ten *)malloc(sizeof(Ten));
            input->old->data=(float *)malloc(sizeof(float)*max_memory);
            input->old2=(Ten *)malloc(sizeof(Ten));
            input->old2->data=(float *)malloc(sizeof(float)*max_memory);
            input->old->old2=input->old2;
            im2col_buf=(float *)malloc(sizeof(float)*max_memory_im2col);
            input2=(Ten *)malloc(sizeof(Ten));
        }
        
        input->h=im0->height;
        input->w=im0->width;
        
        input->d=3;
        ssize=input->h * input->w;
        //        printf("ssize: %d\n",ssize);
        //        fprintf(stderr,"input image: %d x %d x %d\n",input->w,input->h,input->d);
        float *ptr_r=input->data;
        float *ptr_g=ptr_r + ssize;
        float *ptr_b=ptr_g + ssize;
        for(y=0;y<im0->height;y++){
            for(x=0;x<im0->width;x++){
                CvScalar s=cvGet2D(im0,y,x);
                *ptr_r++=(float)(s.val[2]); // red
                *ptr_g++=(float)(s.val[1]); // gree
                *ptr_b++=(float)(s.val[0]); // blue
            }
        }

        input2->w=1;
        input2->h=1;
        input2->d=32;
        input2->data=tmp;
        //        float *test=input2->data;
        //        for(int i=0;i<input2->d;i++){
        //            printf("データ: %f", *test);
        //            *test++;
        //        }
        
        char *p;

        
        
        Ten *out=DCNN(input,input2);
        //           Ten *out=DCNN_dummy(input);
        //         Ten *out = input;
        
        /*計測処理終了*/
        NSTimeInterval time = -[start timeIntervalSinceNow];
        NSLog(@"%lf\n", time);
        //out->w, out->hをinput2に設定
        fprintf(stderr,"output image: %d x %d x %d\n",out->w,out->h,out->d);
        //
        im=cvCreateImage(cvSize(out->w,out->h),IPL_DEPTH_8U, 3);
        
        ssize=out->h * out->w;
        ptr_r=out->data;
        ptr_g=ptr_r + ssize;
        ptr_b=ptr_g + ssize;
        for(y=0;y<im->height;y++){
            for(x=0;x<im->width;x++){
                //        CvScalar s=cvGet2D(im0,y+offy,x+offx);
                CvScalar s=cvGet2D(im,y,x);
                //       s.val[2]=(float)(s.val[2]-avg_image[ss*y+x]); // red
                //       s.val[1]=(float)(s.val[1]-avg_image[ss*y+x+sz]); // gree
                //       s.val[0]=(float)(s.val[0]-avg_image[ss*y+x+sz*2]); // blue
                s.val[0]=(float)(*ptr_r++); // red
                s.val[1]=(float)(*ptr_g++); // gree
                s.val[2]=(float)(*ptr_b++); // blue
                //        printf("%.3f ",*ptr_r);
                cvSet2D(im,y,x,s);
            }
        }
        
        printf("%d\n", color_prev);
        //        color preserving mode
        if (color_prev) {
            IplImage *y1,*y2,*cb1,*cb2,*cr1,*cr2,*org,*org2,*out0;
            //            元画像->変換画像にリサイズ
            org=cvCreateImage(cvSize(out->w,out->h),IPL_DEPTH_8U, 3);
            org2=cvCreateImage(cvSize(out->w,out->h),IPL_DEPTH_8U, 3);
            out0=cvCreateImage(cvSize(out->w,out->h),IPL_DEPTH_8U, 3);
            cvResize(im0,org,CV_INTER_CUBIC);
            //            元画像: bgr->YCrCb
            //            cvCvtColor(org,org2,CV_BGR2YCrCb);
            cvCvtColor(org,org2,CV_BGR2Lab);
            y1=cvCreateImage(cvSize(out->w,out->h),IPL_DEPTH_8U, 1);
            cb1=cvCreateImage(cvSize(out->w,out->h),IPL_DEPTH_8U, 1);
            cr1=cvCreateImage(cvSize(out->w,out->h),IPL_DEPTH_8U, 1);
            cvSplit(org2,y1,cb1,cr1,NULL);
            //            変換画像: rgb -> YCrCb
            //            cvCvtColor(im,out0,CV_RGB2YCrCb);
            cvCvtColor(im,out0,CV_RGB2Lab);
            y2=cvCreateImage(cvSize(out->w,out->h),IPL_DEPTH_8U, 1);
            cb2=cvCreateImage(cvSize(out->w,out->h),IPL_DEPTH_8U, 1);
            cr2=cvCreateImage(cvSize(out->w,out->h),IPL_DEPTH_8U, 1);
            cvSplit(out0,y2,cb2,cr2,NULL);
            
            cvMerge(y2,cb1,cr1,NULL,out0);
            //            CV_YCrCb -> RGB
            //            cvCvtColor(out0,im,CV_YCrCb2RGB);
            cvCvtColor(out0,im,CV_Lab2RGB);
        }else{
            printf("not color\n");
        }
        
        
        image_result1=[self UIImageFromIplImage:im];
        
        self.previewView.image = image_result1;

        
#ifdef DEBUG
        if (pr) printf("-------------------------\n");
#endif
        
#if 0
        int s=out->w * out->h * out->d;
        IDX *idx=(IDX *)malloc(sizeof(IDX)*s);
        float *ptr=out->data;
        for(i=0;i<s;i++){
            //    printf("%.5f ",out->data[i]);
            idx[i].idx=i;
            idx[i].v = *ptr++;
        }
#endif
#if 0
        qsort(idx,s,sizeof(IDX),IDX_sort);
        for(i=0;i<top;i++)
            printf("%d %s %.5f\n",idx[i].idx,foodname[idx[i].idx],idx[i].v);
        
        free(idx);
        
        free(out->old2->data);
        free(out->old2);
        free(out->old->data);
        free(out->old);
        free(out->data);
        free(out);
        free(im2col_buf);
#endif
    });
}

// サンプルバッファのデータからCGImageRefを生成する
- (UIImage *)imageFromSampleBuffer:(CMSampleBufferRef)sampleBuffer
{
    CVImageBufferRef imageBuffer = CMSampleBufferGetImageBuffer(sampleBuffer);
    
    // ピクセルバッファのベースアドレスをロックする
    CVPixelBufferLockBaseAddress(imageBuffer, 0);
    
    // Get information of the image
    uint8_t *baseAddress = (uint8_t *)CVPixelBufferGetBaseAddressOfPlane(imageBuffer, 0);
    
    size_t bytesPerRow = CVPixelBufferGetBytesPerRow(imageBuffer);
    size_t width = CVPixelBufferGetWidth(imageBuffer);
    size_t height = CVPixelBufferGetHeight(imageBuffer);
    
    // RGBの色空間
    CGColorSpaceRef colorSpace = CGColorSpaceCreateDeviceRGB();
    
    CGContextRef newContext = CGBitmapContextCreate(baseAddress,
                                                    width,
                                                    height,
                                                    8,
                                                    bytesPerRow,
                                                    colorSpace,
                                                    kCGBitmapByteOrder32Little | kCGImageAlphaPremultipliedFirst);
    
    CGImageRef cgImage = CGBitmapContextCreateImage(newContext);
    
    CGContextRelease(newContext);
    CGColorSpaceRelease(colorSpace);
    CVPixelBufferUnlockBaseAddress(imageBuffer, 0);
    
    UIImage *image = [UIImage imageWithCGImage:cgImage scale:1.0 orientation:UIImageOrientationUp];
    
    //300x300: 200ms
    //400x400: 350ms
    CGFloat width_1 = width_1_tmp;  // リサイズ後幅のサイズ
    CGFloat height_1 = height_1_tmp;  // リサイズ後高さのサイズ
    UIGraphicsBeginImageContext(CGSizeMake(width_1, height_1));
    [image drawInRect:CGRectMake(0, 0, width_1, height_1)];
    image = UIGraphicsGetImageFromCurrentImageContext();
    UIGraphicsEndImageContext();
    
    
    CGImageRelease(cgImage);
    
    return image;
}
UIImage *image_result1;
bool color_prev=false;

//入力の動的変換
int width_1_tmp=300;
int height_1_tmp=300;

//重みの初期値
//for MMM 14 styles
//float tmp[]= {1,1, 1, 1, 1,1, 1, 1,1, 1, 1, 1, 1, 1};
float tmp[]= { 0, 0, 0, 0 ,0, 0, 0,
    0, 0, 0, 0, 0, 0,0,0,0,0,0.6,0,0,0,0,0,0,0,0,0,0,0,0,0,0 };
float tmp0[]= { 0.8, 0, 0, 0 ,0, 0, 0,
    0, 0, 0, 0, 0, 0 };
//シンプソンズ0.3最適(1だと強すぎるイメージ)
float tmp1[]= { 0, 1, 0, 0 ,0, 0, 0,
    0, 0, 0, 0, 0, 0 };
//イカスミ0.5次郎0.5
float tmp2[]= { 0, 0, 0.8, 0,0, 0, 0,
    0, 0, 0, 0, 0, 0 };
//次郎
float tmp3[]= { 0, 0, 0, 0.8,0, 0, 0,
    0, 0, 0, 0, 0, 0 };
//ピカソ
float tmp4[]= { 0, 0, 0, 0,0.8, 0, 0,
    0, 0, 0, 0, 0, 0 };
//次郎0.5ナポリタン0.5
float tmp5[]= { 0, 0, 0, 0, 0, 0.8, 0,
    0, 0, 0, 0, 0, 0 };
//スターリーナイト
float tmp6[]= { 0, 0, 0, 0,0, 0, 1,
    0, 0, 0, 0, 0, 0 };
float tmp7[]= { 0, 0, 0, 0,0, 0, 0,
    0.8, 0, 0, 0, 0, 0 };
float tmp8[]= { 0, 0, 0, 0,0, 0, 0,
    0, 0.8, 0, 0, 0, 0 };
//シンプソンズ別0.(1だと強すぎる)あまり良くない
float tmp9[]= { 0, 0, 0, 0,0, 0, 0,
    0, 0, 0.3, 0, 0, 0 };
//スケッチ
float tmp10[]= { 0, 0, 0, 0,0, 0, 0,
    0, 0, 0, 1, 0, 0 };
//波変更0.6
float tmp11[]={0, 0, 0, 0,0, 0, 0,
    0, 0, 0, 0, 0.5, 0};
//次郎0.5焼きそば0.5
float tmp12[]={0, 0, 0, 0,0, 0, 0,
    0, 0, 0, 0, 0, 0.8};

#define STYLENUM 32

- (IBAction)weight_select1:(id)sender {
    //    int i=0;
    //    tmp[i]=tmp[i]*self.weight_select1.value;

    for(int i=0; i<STYLENUM; i++){
        if(i==0){
            tmp[i]=self.weight_select1.value;
        }
    }
    self.weight_value1.text=[NSString stringWithFormat:@"%.2f", self.weight_select1.value];
    /*
     for(int i=0; i<13;i++){
     printf("%f\n", tmp[i]);
     }
     printf("***test***\n");
     */
}
- (IBAction)weight_select2:(id)sender {
    for(int i=0; i<STYLENUM; i++){
        if(i==1){
            tmp[i]=self.weight_select2.value;
        }
    }
    self.weight_value2.text=[NSString stringWithFormat:@"%.2f", self.weight_select2.value];
}

- (IBAction)weight_select3:(id)sender {
    for(int i=0; i<STYLENUM; i++){
        if(i==2){
            tmp[i]=self.weight_select3.value;
        }
    }
    self.weight_value3.text=[NSString stringWithFormat:@"%.2f", self.weight_select3.value];
}

- (IBAction)weight_select4:(id)sender {
    for(int i=0; i<STYLENUM; i++){
        if(i==3){
            tmp[i]=self.weight_select4.value;
        }
    }
    self.weight_value4.text=[NSString stringWithFormat:@"%.2f", self.weight_select4.value];
}

- (IBAction)weight_select5:(id)sender {
    for(int i=0; i<STYLENUM; i++){
        if(i==4){
            tmp[i]=self.weight_select5.value;
        }
    }
    self.weight_value5.text=[NSString stringWithFormat:@"%.2f", self.weight_select5.value];
}

- (IBAction)weight_select6:(id)sender {
    for(int i=0; i<STYLENUM; i++){
        if(i==5){
            tmp[i]=self.weight_select6.value;
        }
    }
    self.weight_value6.text=[NSString stringWithFormat:@"%.2f", self.weight_select6.value];
}

- (IBAction)weight_select7:(id)sender {
    for(int i=0; i<STYLENUM; i++){
        if(i==6){
            tmp[i]=self.weight_select7.value;
        }
    }
    self.weight_value7.text=[NSString stringWithFormat:@"%.2f", self.weight_select7.value];
}

- (IBAction)weight_select8:(id)sender {
    for(int i=0; i<STYLENUM; i++){
        if(i==7){
            tmp[i]=self.weight_select8.value;
        }
    }
    self.weight_value8.text=[NSString stringWithFormat:@"%.2f", self.weight_select8.value];
}

- (IBAction)weight_select9:(id)sender {
    for(int i=0; i<STYLENUM; i++){
        if(i==8){
            tmp[i]=self.weight_select9.value;
        }
    }
    self.weight_value9.text=[NSString stringWithFormat:@"%.2f", self.weight_select9.value];
}

- (IBAction)weight_select10:(id)sender {
    for(int i=0; i<STYLENUM; i++){
        if(i==9){
            tmp[i]=self.weight_select10.value;
        }
    }
    self.weight_value10.text=[NSString stringWithFormat:@"%.2f", self.weight_select10.value];
}

- (IBAction)weight_select11:(id)sender {
    for(int i=0; i<STYLENUM; i++){
        if(i==10){
            tmp[i]=self.weight_select11.value;
        }
    }
    self.weight_value11.text=[NSString stringWithFormat:@"%.2f", self.weight_select11.value];
}

- (IBAction)weight_select12:(id)sender {
    for(int i=0; i<STYLENUM; i++){
        if(i==11){
            tmp[i]=self.weight_select12.value;
        }
    }
    self.weight_value12.text=[NSString stringWithFormat:@"%.2f", self.weight_select12.value];
}

- (IBAction)weight_select13:(id)sender {
    for(int i=0; i<STYLENUM; i++){
        if(i==12){
            tmp[i]=self.weight_select13.value;
        }
    }
    self.weight_value13.text=[NSString stringWithFormat:@"%.2f", self.weight_select13.value];
}


- (IBAction)color:(id)sender {
    color_prev=true;
}

- (IBAction)standard:(id)sender {
    color_prev=false;
}


//int change_count=1;
//
//bool frontCamera = false;
//- (IBAction)set_camera_device:(id)sender {
//    frontCamera=true;
//    //    if(change_count%2==0){
//    //        frontCamera = false;
//    //        printf("false\n");
//    //    }else{
//    //        frontCamera = true;
//    //                printf("true\n");
//    //    }
//    //    printf("%d \n", change_count%2);
//    //    ++change_count;
//}
//
- (IBAction)reset:(id)sender {
    //    スライド値(重み)リセット用
    self.weight_select1.value=0;
    self.weight_select2.value=0;
    self.weight_select3.value=0;
    self.weight_select4.value=0;
    self.weight_select5.value=0;
    self.weight_select6.value=0;
    self.weight_select7.value=0;
    self.weight_select8.value=0;
    self.weight_select9.value=0;
    self.weight_select10.value=0;
    self.weight_select11.value=0;
    self.weight_select12.value=0;
    self.weight_select13.value=0;
    self.weight_select14.value=0;
    self.weight_select15.value=0;
    self.weight_select16.value=0;
    self.weight_select17.value=0;
    self.weight_select18.value=0.6;
    self.weight_select19.value=0;
    self.weight_select20.value=0;
    self.weight_select21.value=0;
    self.weight_select22.value=0;
    self.weight_select23.value=0;
    self.weight_select24.value=0;
    self.weight_select25.value=0;
    self.weight_select26.value=0;
    self.weight_select27.value=0;
    self.weight_select28.value=0;
    self.weight_select29.value=0;
    self.weight_select30.value=0;
    self.weight_select31.value=0;
    self.weight_select32.value=0;
//    self.weight_select14.value=0;
//    self.weight_value1.text=[NSString stringWithFormat:@"%.2f", 0.00];
//    self.weight_value2.text=[NSString stringWithFormat:@"%.2f", 0.00];
//    self.weight_value3.text=[NSString stringWithFormat:@"%.2f", 0.00];
//    self.weight_value4.text=[NSString stringWithFormat:@"%.2f", 0.00];
//    self.weight_value5.text=[NSString stringWithFormat:@"%.2f", 0.00];
//    self.weight_value6.text=[NSString stringWithFormat:@"%.2f", 0.00];
//    self.weight_value7.text=[NSString stringWithFormat:@"%.2f", 1.00];
//    self.weight_value8.text=[NSString stringWithFormat:@"%.2f", 0.00];
//    self.weight_value9.text=[NSString stringWithFormat:@"%.2f", 0.00];
//    self.weight_value10.text=[NSString stringWithFormat:@"%.2f", 0.00];
//    self.weight_value11.text=[NSString stringWithFormat:@"%.2f", 0.00];
//    self.weight_value12.text=[NSString stringWithFormat:@"%.2f", 0.00];
//    self.weight_value13.text=[NSString stringWithFormat:@"%.2f", 0.00];
//    self.weight_value14.text=[NSString stringWithFormat:@"%.2f", 0.00];

    for(int i=0; i<STYLENUM; i++){
        if(i==17){
            tmp[i]=0.6;
        }else{
            tmp[i]=0;
        }
    }
    width_1_tmp=300;
    height_1_tmp=300;
    partial=false;
//    self.input_size_view.text =[[NSString alloc] initWithFormat:@"400x400"];
}


- (IBAction)image_quality_low:(id)sender {
    width_1_tmp=250;
    height_1_tmp=350;
//    self.input_size_view.text =[[NSString alloc] initWithFormat:@"350x350"];
}

- (IBAction)image_quality_medium:(id)sender {
    width_1_tmp=300;
    height_1_tmp=300;
//    self.input_size_view.text =[[NSString alloc] initWithFormat:@"400x400"];
}

- (IBAction)image_quality_high:(id)sender {
    width_1_tmp=350;
    height_1_tmp=350;
//    self.input_size_view.text =[[NSString alloc] initWithFormat:@"450x450"];
}

- (IBAction)Random:(id)sender {

    double d;
    srand((unsigned)time(NULL));

    for(int i=0;i<STYLENUM;i++){
        d=(float)rand()/RAND_MAX;
        tmp[i]=d;
        if(i==0){
            self.weight_value1.text=[NSString stringWithFormat:@"%.2f", d];
            self.weight_select1.value=d;
        }else if(i==1){
            self.weight_value2.text=[NSString stringWithFormat:@"%.2f", d];
            self.weight_select2.value=d;
        }else if(i==2){
            self.weight_value3.text=[NSString stringWithFormat:@"%.2f", d];
            self.weight_select3.value=d;
        }else if(i==3){
            self.weight_value4.text=[NSString stringWithFormat:@"%.2f", d];
            self.weight_select4.value=d;
        }else if(i==4){
            self.weight_value5.text=[NSString stringWithFormat:@"%.2f", d];
            self.weight_select5.value=d;
        }else if(i==5){
            self.weight_value6.text=[NSString stringWithFormat:@"%.2f", d];
            self.weight_select6.value=d;
        }else if(i==6){
            self.weight_value7.text=[NSString stringWithFormat:@"%.2f", d];
            self.weight_select7.value=d;
        }else if(i==7){
            self.weight_value8.text=[NSString stringWithFormat:@"%.2f", d];
            self.weight_select8.value=d;
        }else if(i==8){
            self.weight_value9.text=[NSString stringWithFormat:@"%.2f", d];
            self.weight_select9.value=d;
        }else if(i==9){
            self.weight_value10.text=[NSString stringWithFormat:@"%.2f", d];
            self.weight_select10.value=d;
        }else if(i==10){
            self.weight_value11.text=[NSString stringWithFormat:@"%.2f", d];
            self.weight_select11.value=d;
        }else if(i==11){
            self.weight_value12.text=[NSString stringWithFormat:@"%.2f", d];
            self.weight_select12.value=d;
        }else if(i==12){
            self.weight_value13.text=[NSString stringWithFormat:@"%.2f", d];
            self.weight_select13.value=d;
        }else if(i==13){
                        self.weight_value14.text=[NSString stringWithFormat:@"%.2f", d];
            self.weight_select14.value=d;
        }else if(i==14){
                        self.weight_value15.text=[NSString stringWithFormat:@"%.2f", d];
            self.weight_select15.value=d;
        }else if(i==15){
                        self.weight_value16.text=[NSString stringWithFormat:@"%.2f", d];
            self.weight_select16.value=d;
        }else if(i==16){
                        self.weight_value17.text=[NSString stringWithFormat:@"%.2f", d];
            self.weight_select17.value=d;
        }else if(i==17){
                        self.weight_value18.text=[NSString stringWithFormat:@"%.2f", d];
            self.weight_select18.value=d;
        }else if(i==18){
                        self.weight_value19.text=[NSString stringWithFormat:@"%.2f", d];
            self.weight_select19.value=d;
        }else if(i==19){
                        self.weight_value20.text=[NSString stringWithFormat:@"%.2f", d];
            self.weight_select20.value=d;
        }else if(i==20){
                        self.weight_value21.text=[NSString stringWithFormat:@"%.2f", d];
            self.weight_select21.value=d;
        }else if(i==21){
                        self.weight_value22.text=[NSString stringWithFormat:@"%.2f", d];
            self.weight_select22.value=d;
        }else if(i==22){
                        self.weight_value23.text=[NSString stringWithFormat:@"%.2f", d];
            self.weight_select23.value=d;
        }else if(i==23){
                        self.weight_value24.text=[NSString stringWithFormat:@"%.2f", d];
            self.weight_select24.value=d;
        }else if(i==24){
                        self.weight_value25.text=[NSString stringWithFormat:@"%.2f", d];
            self.weight_select25.value=d;
        }else if(i==25){
                        self.weight_value26.text=[NSString stringWithFormat:@"%.2f", d];
            self.weight_select26.value=d;
        }else if(i==26){
                        self.weight_value27.text=[NSString stringWithFormat:@"%.2f", d];
            self.weight_select27.value=d;
        }else if(i==27){
                        self.weight_value28.text=[NSString stringWithFormat:@"%.2f", d];
            self.weight_select28.value=d;
        }else if(i==28){
                        self.weight_value29.text=[NSString stringWithFormat:@"%.2f", d];
            self.weight_select29.value=d;
        }else if(i==29){
                        self.weight_value30.text=[NSString stringWithFormat:@"%.2f", d];
            self.weight_select30.value=d;
        }else if(i==30){
                        self.weight_value31.text=[NSString stringWithFormat:@"%.2f", d];
            self.weight_select31.value=d;
        }else if(i==31){
                        self.weight_value32.text=[NSString stringWithFormat:@"%.2f", d];
            self.weight_select32.value=d;
        }
        /*
         else if(i==13){
         self.weight_value14.text=[NSString stringWithFormat:@"%.2f", d];
         self.weight_select14.value=d;
         }*/
    }
}

//- (IBAction)Stripe:(id)sender {
//    partial=true;
//}


- (IBAction)weight_select14:(id)sender {
    for(int i=0; i<STYLENUM; i++){
        if(i==13){
            tmp[i]=self.weight_select14.value;
        }
    }
    self.weight_value14.text=[NSString stringWithFormat:@"%.2f", self.weight_select13.value];
}
- (IBAction)weight_select15:(id)sender {
    for(int i=0; i<STYLENUM; i++){
        if(i==14){
            tmp[i]=self.weight_select15.value;
        }
    }
    self.weight_value15.text=[NSString stringWithFormat:@"%.2f", self.weight_select13.value];
}
- (IBAction)weight_select16:(id)sender {
    for(int i=0; i<STYLENUM; i++){
        if(i==15){
            tmp[i]=self.weight_select16.value;
        }
    }
    self.weight_value16.text=[NSString stringWithFormat:@"%.2f", self.weight_select13.value];
}
- (IBAction)weight_select17:(id)sender {
    for(int i=0; i<STYLENUM; i++){
        if(i==16){
            tmp[i]=self.weight_select17.value;
        }
    }
    self.weight_value17.text=[NSString stringWithFormat:@"%.2f", self.weight_select13.value];
}
- (IBAction)weight_select18:(id)sender {
    for(int i=0; i<STYLENUM; i++){
        if(i==17){
            tmp[i]=self.weight_select18.value;
        }
    }
    self.weight_value18.text=[NSString stringWithFormat:@"%.2f", self.weight_select13.value];
}
- (IBAction)weight_select19:(id)sender {
    for(int i=0; i<STYLENUM; i++){
        if(i==18){
            tmp[i]=self.weight_select19.value;
        }
    }
    self.weight_value19.text=[NSString stringWithFormat:@"%.2f", self.weight_select13.value];
}
- (IBAction)weight_select20:(id)sender {
    for(int i=0; i<STYLENUM; i++){
        if(i==19){
            tmp[i]=self.weight_select20.value;
        }
    }
    self.weight_value20.text=[NSString stringWithFormat:@"%.2f", self.weight_select13.value];
}
- (IBAction)weight_select21:(id)sender {
    for(int i=0; i<STYLENUM; i++){
        if(i==20){
            tmp[i]=self.weight_select21.value;
        }
    }
    self.weight_value21.text=[NSString stringWithFormat:@"%.2f", self.weight_select13.value];
}
- (IBAction)weight_select22:(id)sender {
    for(int i=0; i<STYLENUM; i++){
        if(i==21){
            tmp[i]=self.weight_select22.value;
        }
    }
    self.weight_value22.text=[NSString stringWithFormat:@"%.2f", self.weight_select13.value];
}
- (IBAction)weight_select23:(id)sender {
    for(int i=0; i<STYLENUM; i++){
        if(i==22){
            tmp[i]=self.weight_select23.value;
        }
    }
    self.weight_value23.text=[NSString stringWithFormat:@"%.2f", self.weight_select13.value];
}
- (IBAction)weight_select24:(id)sender {
    for(int i=0; i<STYLENUM; i++){
        if(i==23){
            tmp[i]=self.weight_select24.value;
        }
    }
    self.weight_value24.text=[NSString stringWithFormat:@"%.2f", self.weight_select13.value];
}
- (IBAction)weight_select25:(id)sender {
    for(int i=0; i<STYLENUM; i++){
        if(i==24){
            tmp[i]=self.weight_select25.value;
        }
    }
    self.weight_value25.text=[NSString stringWithFormat:@"%.2f", self.weight_select13.value];
}
- (IBAction)weight_select26:(id)sender {
    for(int i=0; i<STYLENUM; i++){
        if(i==25){
            tmp[i]=self.weight_select26.value;
        }
    }
    self.weight_value26.text=[NSString stringWithFormat:@"%.2f", self.weight_select13.value];
}
- (IBAction)weight_select27:(id)sender {
    for(int i=0; i<STYLENUM; i++){
        if(i==26){
            tmp[i]=self.weight_select27.value;
        }
    }
    self.weight_value27.text=[NSString stringWithFormat:@"%.2f", self.weight_select13.value];
}
- (IBAction)weight_select28:(id)sender {
    for(int i=0; i<STYLENUM; i++){
        if(i==27){
            tmp[i]=self.weight_select28.value;
        }
    }
    self.weight_value28.text=[NSString stringWithFormat:@"%.2f", self.weight_select13.value];
}
- (IBAction)weight_select29:(id)sender {
    for(int i=0; i<STYLENUM; i++){
        if(i==28){
            tmp[i]=self.weight_select29.value;
        }
    }
    self.weight_value29.text=[NSString stringWithFormat:@"%.2f", self.weight_select13.value];
}
- (IBAction)weight_select30:(id)sender {
    for(int i=0; i<STYLENUM; i++){
        if(i==29){
            tmp[i]=self.weight_select30.value;
        }
    }
    self.weight_value30.text=[NSString stringWithFormat:@"%.2f", self.weight_select13.value];
}
- (IBAction)weight_select31:(id)sender {
    for(int i=0; i<STYLENUM; i++){
        if(i==30){
            tmp[i]=self.weight_select31.value;
        }
    }
    self.weight_value31.text=[NSString stringWithFormat:@"%.2f", self.weight_select13.value];
}
- (IBAction)weight_select32:(id)sender {
    for(int i=0; i<STYLENUM; i++){
        if(i==31){
            tmp[i]=self.weight_select32.value;
        }
    }
    self.weight_value32.text=[NSString stringWithFormat:@"%.2f", self.weight_select13.value];
}
//- (IBAction)color:(id)sender {
//}
@end

