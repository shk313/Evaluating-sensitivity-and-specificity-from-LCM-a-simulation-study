/// Model 4: Conditional dependence in disease negative individuals ///
data {
  
  int<lower=1> M; // Number of different tests
  int<lower=1> N; // Number of people in study
  int<lower=0,upper=1> t1[N]; // results of T1
  int<lower=0,upper=1> t2[N]; 
  int<lower=0,upper=1> t3[N];
  int<lower=0,upper=1> t4[N];
  int<lower=0,upper=1> t5[N];

}


parameters {
  
  real<lower=0,upper=1> prev;
  
  vector[N] RE;
  real<lower=0,upper=5> bneg;
  
  real<lower=0,upper=1> a12;
  real<lower=1-inv_logit(logit(a12)*2),upper=1> a11;
  real<lower=0,upper=1> a22;
  real<lower=1-inv_logit(logit(a22)-bneg*2),upper=1> a21;
  real<lower=0,upper=1> a32;
  real<lower=1-inv_logit(logit(a32)-bneg*2),upper=1> a31;
  real<lower=0,upper=1> a42;
  real<lower=1-inv_logit(logit(a42)-bneg*2),upper=1> a41;
  real<lower=0,upper=1> a52;
  real<lower=1-inv_logit(logit(a52)-bneg*2),upper=1> a51;


}

transformed parameters {
  
  simplex[2] theta; // prob infected or not infected
  vector[N] prob[M,2];   

  theta[1] = 1-prev;
  theta[2] = prev;

// dependence in negatives  
  prob[1,1] = rep_vector(1-a11, N); // independent test 1
  prob[1,2] = rep_vector(a12, N);
  prob[2,1] = inv_logit(logit(1-a21)+bneg*RE);
  prob[2,2] = rep_vector(a22, N);
  prob[3,1] = inv_logit(logit(1-a31)+bneg*RE);
  prob[3,2] = rep_vector(a32, N);
  prob[4,1] = inv_logit(logit(1-a41)+bneg*RE);
  prob[4,2] = rep_vector(a42, N);
  prob[5,1] = inv_logit(logit(1-a51)+bneg*RE);
  prob[5,2] = rep_vector(a52, N);
  // //

}


model {
  real ps[2];
  
  // priors
  a11~beta(10,1);
  a12~beta(1,1);
  a21~beta(5,1);
  a22~beta(1,1);
  a31~beta(5,1);
  a32~beta(1,1);
  a41~beta(5,1);
  a42~beta(1,1);
  a51~beta(5,1);
  a52~beta(1,1);

  prev~beta(1,1); 

  RE~normal(0,1);
  bneg~gamma(1,1);


  for(n in 1:N){
    for(k in 1:2){
      ps[k] = log(theta[k]) +  binomial_lpmf(t1[n]| 1, prob[1,k,n]) +  binomial_lpmf(t2[n]| 1, prob[2,k,n]) + binomial_lpmf(t3[n]| 1, prob[3,k,n]) + binomial_lpmf(t4[n]| 1, prob[4,k,n]) + binomial_lpmf(t5[n]| 1, prob[5,k,n]) ;
    }

  target += log_sum_exp(ps);
  }


}

generated quantities {
  real Se_mean[M];
  real Sp_mean[M];
  
    // for loo-cv
  //vector[N] log_lik;
  //real ll[2];
  // 
  // int<lower=0> y_pred[N,M];
  // int<lower=0,upper=1> inf[N];
  // 
  // real<lower=0,upper=1> p[N,M];

for(m in 1:M){
  Se_mean[m] = mean(prob[m,2,]);
  Sp_mean[m] = mean(1-prob[m,1,]);
}
  
// prediction  
// for(n in 1:N){
//    inf[n] = binomial_rng(1,theta[2]);
//  }
// 
// for(n in 1:N){
//   for(m in 1:M){
//       p[n,m] = (inf[n]*prob[m,2,n])+((1-inf[n])*(prob[m,1,n])); // probaility of person N being positive for test m
//    }
// }
// 
// for(n in 1:N){
//   for(m in 1:M){
//     y_pred[n,m] = binomial_rng(1, p[n,m]); // test result for person N on test M
//   }
// }
// 
// 
// //Likelihood for use in LOO-CV
//for(n in 1:N){
//  for(k in 1:2){
//    ll[k] = log(theta[k]) +  binomial_lpmf(t1[n]| 1, prob[1,k,n]) +  binomial_lpmf(t2[n]| 1, prob[2,k,n]) + binomial_lpmf(t3[n]| 1, prob[3,k,n]) + binomial_lpmf(t4[n]| 1, prob[4,k,n]) + binomial_lpmf(t5[n]| 1, prob[5,k,n]);
//  }
//
//log_lik[n] = log_sum_exp(ll);
//}
  
}
