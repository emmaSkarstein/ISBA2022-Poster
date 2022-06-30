![ISBAposter](https://github.com/emmaSkarstein/ISBA2022-Poster/blob/main/Poster_ISBA.png)


# ABSTRACT
Missing data and measurement error are common problems in most applied data sets. However, while the former is receiving considerable attention, many researchers are still not routinely accounting for varying types of measurement error in their variables. 

By viewing missing data as a limiting case of measurement error, we propose Bayesian hierarchical models to account for continuous covariate measurement error and missingness simultaneously. The investigated models encompass both the well-known classical measurement error, but also the less considered Berkson measurement error, which often occurs, among other places, in experimental setups and exposure studies. 

The Bayesian framework is very flexible, and allows us to incorporate any prior knowledge we may have about the nature of the measurement error. We illustrate how the respective methods can then be efficiently implemented via integrated nested Laplace approximations (INLA). 

# WHAT'S THE PROBLEM?
In the context of INLA, both missing data and measurement error in covariates become a problem, as the covariates are part of the latent field of the model, and INLA does not allow these variables to have unobserved values. This applies for both missing data and measurement error, as in the case of measurement error all the values are essentially unobserved, and the value we observe is just an approximation to the truth. Therefore it is not obvious how to deal with measurement error and missing data in INLA. 

# HOW DOES IT WORK?
Previously, Muff et al. (2015) have presented a model for covariate measurement error in INLA, whereas options for missing covariate values have been presented in Gómez-Rubio et al. (2019) and Gómez-Rubio (2020).

We show how the measurement error model presented by Muff et al. (2015) can be used directly to impute missing data in INLA as well. This results in a joint Bayesian model that is capable of simultaneously accounting for classical measurement error, Berkson measurement error, missing data, or any combination of these, occurring in the same covariate. Using the model reformulation introduced in Muff et al. (2015) we are able to formulate our model in an appropriate manner, opening the doors to a new world of measurement error and missing data models in INLA.

A particularly important special case of our presented model is the one for only missing data, with no measurement error occurring. This leads to a straight-forward way to do missing value imputation for the covariates in INLA, which, as mentioned, is something that has not been well covered in the INLA literature previously. Gómez-Rubio et al. (2019) show how to do missing data imputation in INLA in a more general case, when the missingness is not at random. But in practice, an imputation model may often be sufficient for our needs, and when we have reason to believe that our data is missing at random this will be a lot easier to implement. 

# MODEL STRUCTURE

![posterequation](https://github.com/emmaSkarstein/ISBA2022-Poster/blob/main/poster_equation.png)

**MODEL OF INTEREST** The first level of the model is the actual model of interest, a generalized linear (mixed) model where one or more covariates have measurement error or missing values (or both). 

**CLASSICAL MEASUREMENT ERROR MODEL** Next comes the model for the classical measurement error, which describes how our observed covariate has some level of noise associated with it. 

**IMPUTATION MODEL** The last level is the imputation model. In the measurement error context, this describes how the true value of the mis-observed covariate is allowed to depend on other variables of the model, thus taking advantage of potential correlations to inform the estimation of this covariate, along with the measurement error model. If the observation is completely missing, then the imputation model alone imputes the missing value. The imputation model is connected to the model of interest through the measurement error model. 


# WOULD YOU LIKE TO LEARN MORE?
The paper summarizing this work is still not completely finished as of ISBA 2022, but if you are interested, you are welcome to ask for a copy of the draft by e-mailing me at emma.s.skarstein@ntnu.no.


# REFERENCES

Gómez-Rubio, V., Cameletti, M., and Blangiardo, M. (2019). Missing data analysis and imputation via latent Gaussian Markov random fields. Preprint, arXiv:1912.10981.

Muff, S., Riebler, A., Held, L., Rue, H., and Saner, P. (2015). Bayesian analysis of measurement error models using integrated nested Laplace approximations. Journal of the Royal Statistical Society: Series C (Applied Statistics), 64(2):231-252.


# POSTER DESIGN
The poster design is copied from this [Psycho poster](https://www.goldposter.com/10058/). I have also been inspired by Mike Morrison's \#betterposter campaign, and would definitely recommend checking out his videos on how to make better academic posters in less time, here is one of them: https://www.youtube.com/watch?v=WBjhxjWDiHw. Of course, Mike Morrison's point is to help researchers spend less time on their posters. But I really like fiddling with the design of my poster, and so I tried to adapt his principles while simultaneously trying to copy the design of the movie poster. I think vintage movie posters or advertisements are actually great models for how we could design academic posters, as they grab your attention with impactful and attractive colors and fonts, while at the same time not being too over-crowded and often very minimalistic. I think this use of fonts, colors and simplistic figures or icons can be a great guide, and copying a movie poster makes it more of an interesting challenge rather than a frustration. 

