/// Model 3: Conditional dependence in disease positive individuals ///

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
  real<lower=0,upper=5> bpos;
  
  real<lower=0,upper=1> a11;
  real<lower=1-inv_logit(logit(a11)*2),upper=1> a12;
  real<lower=0,upper=1> a21;
  real<lower=1-inv_logit(logit(a21)-bpos*2),upper=1> a22;
  real<lower=0,upper=1> a31;
  real<lower=1-inv_logit(logit(a31)-bpos*2),upper=1> a32;
  real<lower=0,upper=1> a41;
  real<lower=1-inv_logit(logit(a41)-bpos*2),upper=1> a42;
  real<lower=0,upper=1> a51;
  real<lower=1-inv_logit(logit(a51)-bpos*2),upper=1> a52;
}

transformed parameters {
  
  simplex[2] theta; // prob infected or not infected
  vector[N] prob[M,2];   

  theta[1] = 1-prev;
  theta[2] = prev;
  
// Test 1 
  prob[1,1] = rep_vector(1-a11, N); // Specificity
  prob[1,2] = rep_vector(a12, N); // Sensitivity
// Test 2
  prob[2,1] = rep_vector(1-a21, N);
  prob[2,2] = inv_logit(logit(a22)+bpos*RE);
// Test 3
  prob[3,1] = rep_vector(1-a31, N);
  prob[3,2] = inv_logit(logit(a32)+bpos*RE);
// Test 4
  prob[4,1] = rep_vector(1-a41, N);
  prob[4,2] = inv_logit(logit(a42)+bpos*RE);
// Test 5
  prob[5,1] = rep_vector(1-a51, N);
  prob[5,2] = inv_logit(logit(a52)+bpos*RE);



}


model {
  real ps[2];
  
// Priors  
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
  bpos~gamma(1,1);
//  
  
  for(n in 1:N){
    for(k in 1:2){
      ps[k] = log(theta[k]) +  binomial_lpmf(t1[n]| 1, prob[1,k,n]) +  binomial_lpmf(t2[n]| 1, prob[2,k,n]) + binomial_lpmf(t3[n]| 1, prob[3,k,n]) + binomial_lpmf(t4[n]| 1, prob[4,k,n]) + binomial_lpmf(t5[n]| 1, prob[5,k,n]);
    }

  target += log_sum_exp(ps);
  }


}

generated quantities {
  real Se_mean[M];
  real Sp_mean[M];
  
for(m in 1:M){
  Se_mean[m] = mean(prob[m,2,]);
  Sp_mean[m] = mean(1-prob[m,1,]);
}
  
}
