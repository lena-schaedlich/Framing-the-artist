data {
  int<lower=1> N;
  int<lower=1> ArtistsN;
  int<lower=1> EmotionsN;
  int<lower=1> PresentationsN;
  int<lower=1> MaxResponse;
  
  array[N] int<lower=1,upper=MaxResponse> Response;
  array[N] int<lower=1,upper=ArtistsN> Artist;
  array[N] int<lower=1,upper=EmotionsN> Emotion;
  array[N] int<lower=1,upper=PresentationsN> Presentation;
}

parameters {
  simplex[MaxResponse] response_intervals;
  
  array[ArtistsN] matrix[EmotionsN, PresentationsN] eta;
}

transformed parameters {
   ordered[MaxResponse - 1] cutpoints = logit(cumulative_sum(response_intervals)[1:(MaxResponse - 1)]);
}

model {
  response_intervals ~ dirichlet(rep_vector(3, MaxResponse));
  for(iA in 1:ArtistsN) to_vector(eta[iA]) ~ normal(0, 1);

  for(i in 1:N) Response[i] ~  ordered_logistic(eta[Artist[i]][Emotion[i], Presentation[i]], cutpoints);
}
