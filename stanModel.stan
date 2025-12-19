// hier_rl_ctx.stan
data {
  int<lower=1> S;               // number of subjects
  int<lower=1> T;               // total number of trials (stacked)
  int<lower=1> K;               // number of contexts (condition x block), e.g. 5*2 = 10
  array[T] int<lower=1, upper=S> subj;   // subject index per trial
  array[T] int<lower=1, upper=K> ctx;    // context index per trial (1..K)
  array[T] int<lower=1, upper=2> choice; // choice: 1 or 2
  array[T] int<lower=0, upper=1> reward; // reward indicator (0/1)
  array[T] int<lower=0, upper=1> reset;  // 1 if this trial should reset Q-values
}

parameters {
  // hyperparameters per context
  vector[K] mu_a;       // for alpha (on logit scale)
  vector<lower=0>[K] sigma_a;

  vector[K] mu_d;       // for decay (logit scale)
  vector<lower=0>[K] sigma_d;

  vector[K] mu_logbeta; // for log(beta)
  vector<lower=0>[K] sigma_logbeta;

  // subject x context raw effects (std normal)
  matrix[S, K] a_raw;   // for alpha
  matrix[S, K] d_raw;   // for decay
  matrix[S, K] b_raw;   // for log-beta
}
transformed parameters {
  // subject x context parameters on constrained scales
  matrix[S, K] alpha;   // in (0,1)
  matrix[S, K] decay;   // in (0,1)
  matrix[S, K] beta;    // >0

  for (s in 1:S) {
    for (k in 1:K) {
      alpha[s,k] = inv_logit(mu_a[k] + sigma_a[k] * a_raw[s,k]);
      decay[s,k] = inv_logit(mu_d[k] + sigma_d[k] * d_raw[s,k]);
      beta[s,k]  = exp(mu_logbeta[k] + sigma_logbeta[k] * b_raw[s,k]);
    }
  }
}
model {
  // Priors on hyperparameters (weakly informative)
  mu_a ~ normal(0, 1);
  sigma_a ~ normal(0, 1);

  mu_d ~ normal(0, 1);
  sigma_d ~ normal(0, 1);

  mu_logbeta ~ normal(0, 1);
  sigma_logbeta ~ normal(0, 1);

  // standard normal for raw subject deviations
  to_vector(a_raw) ~ normal(0, 1);
  to_vector(d_raw) ~ normal(0, 1);
  to_vector(b_raw) ~ normal(0, 1);

  // Likelihood: loop trials, maintain Q-values per subject
  {
    // We'll iterate through trials in order; data must be ordered by subject and trial.
    // We maintain per-subject Q-values as locals and reset them when reset==1
    // Note: because trials are stacked across subjects, we need to track current subject
    int cur_subj = -1;
    real Q1 = 0.0;
    real Q2 = 0.0;

    for (t in 1:T) {
      int s = subj[t];
      int k = ctx[t];

      if (s != cur_subj) {
        // new subject -> initialize Qs to zero
        cur_subj = s;
        Q1 = 0.0;
        Q2 = 0.0;
      }
      if (reset[t] == 1) {
        // reset within subject (start of block)
        Q1 = 0.0;
        Q2 = 0.0;
      }

      // Obtain subject-context params
      real a = alpha[s,k];
      real d = decay[s,k];
      real b = beta[s,k];

      // softmax choice probability for action 1
      real exp1 = exp(b * Q1);
      real exp2 = exp(b * Q2);
      real p1 = exp1 / (exp1 + exp2);
      p1 = fmin(fmax(p1, 1e-9), 1-1e-9);

      if (choice[t] == 1)
        target += log(p1);
      else
        target += log1m(p1);

      // reward is 0/1 (scale as needed before feeding)
      real r = reward[t]; // assume on same scale as Q (we use ±1 or 0/1; consistent scaling matters)

      // prediction error and updates
      if (choice[t] == 1) {
        real delta = r - Q1;
        Q1 = Q1 + a * delta - d * (Q1 - 0.0);
        Q2 = Q2 - d * (Q2 - 0.0); // decay for unchosen
      } else {
        real delta = r - Q2;
        Q2 = Q2 + a * delta - d * (Q2 - 0.0);
        Q1 = Q1 - d * (Q1 - 0.0);
      }
    }
  }
}
generated quantities {
  // Optionally compute subject/context-level posterior means or PPC - omitted for brevity
}
