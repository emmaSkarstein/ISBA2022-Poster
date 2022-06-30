Simulation: Berkson and classical measurement error alongside missing
data
================

<!-- README.md is generated from README.Rmd. Please edit that file -->
<!-- badges: start -->
<!-- badges: end -->

``` r
library(INLA)
```

![
\\def\\na{\\texttt{NA}}
](https://latex.codecogs.com/png.image?%5Cdpi%7B110%7D&space;%5Cbg_white&space;%0A%5Cdef%5Cna%7B%5Ctexttt%7BNA%7D%7D%0A "
\def\na{\texttt{NA}}
")

We here provide a simulation example that illustrates the missing data
and measurement error model, with extensive comments.

For this example, we simulate a linear regression model with a
mismeasured covariate
![\\boldsymbol{x}](https://latex.codecogs.com/png.image?%5Cdpi%7B110%7D&space;%5Cbg_white&space;%5Cboldsymbol%7Bx%7D "\boldsymbol{x}"),
observed as
![\\boldsymbol{w}](https://latex.codecogs.com/png.image?%5Cdpi%7B110%7D&space;%5Cbg_white&space;%5Cboldsymbol%7Bw%7D "\boldsymbol{w}"),
as well as a covariate without measurement error,
![\\boldsymbol{z}](https://latex.codecogs.com/png.image?%5Cdpi%7B110%7D&space;%5Cbg_white&space;%5Cboldsymbol%7Bz%7D "\boldsymbol{z}").
The covariate
![\\boldsymbol{x}](https://latex.codecogs.com/png.image?%5Cdpi%7B110%7D&space;%5Cbg_white&space;%5Cboldsymbol%7Bx%7D "\boldsymbol{x}")
is constructed to have both Berkson and classical measurement error, and
it is also missing (completely at random) approximately 20% of the
observations.

# Data generation

The data is generated in the following code.

``` r
n <- 1000
# Covariate without error:
z <- rnorm(n, mean = 0, sd = 1)

# Berkson error:
u_b <- rnorm(n)
w_b <- rnorm(n, mean = 1 + 0.5*z, sd = 1)
x <- w_b + u_b

# Response:
y <- 1 + x + z + rnorm(n)

# Classical error:
u_c <- rnorm(n)
w_c <- x + u_c

# Missingness:
miss_index <- sample(1:n, 0.2*n, replace = FALSE)
w_c[miss_index] <- NA

simulated_data <- data.frame(y = y, w = w_c, z = z)
```

The simulated “observed” data then consists of three columns:

![
\\boldsymbol{y} \\quad \\boldsymbol{w} \\quad \\boldsymbol{z}
](https://latex.codecogs.com/png.image?%5Cdpi%7B110%7D&space;%5Cbg_white&space;%0A%5Cboldsymbol%7By%7D%20%5Cquad%20%5Cboldsymbol%7Bw%7D%20%5Cquad%20%5Cboldsymbol%7Bz%7D%0A "
\boldsymbol{y} \quad \boldsymbol{w} \quad \boldsymbol{z}
")

For
![n = 1000](https://latex.codecogs.com/png.image?%5Cdpi%7B110%7D&space;%5Cbg_white&space;n%20%3D%201000 "n = 1000")
simulated observations, they contain:

-   ![y_1, \\dots, y_n](https://latex.codecogs.com/png.image?%5Cdpi%7B110%7D&space;%5Cbg_white&space;y_1%2C%20%5Cdots%2C%20y_n "y_1, \dots, y_n"):
    The continuous response.
-   ![w_1, \\dots, w_n](https://latex.codecogs.com/png.image?%5Cdpi%7B110%7D&space;%5Cbg_white&space;w_1%2C%20%5Cdots%2C%20w_n "w_1, \dots, w_n"):
    A continuous covariate with classical and Berkson measurement error
    and missing values.
-   ![z_1, \\dots, z_n](https://latex.codecogs.com/png.image?%5Cdpi%7B110%7D&space;%5Cbg_white&space;z_1%2C%20%5Cdots%2C%20z_n "z_1, \dots, z_n"):
    A continuous covariate.

``` r
attach(simulated_data)
#> The following objects are masked _by_ .GlobalEnv:
#> 
#>     y, z
n <- nrow(simulated_data)
```

# Model

Our response for this model will be

![
y_i = \\beta_0 + \\beta_x x_i + \\beta_z z_i + \\varepsilon_i, \\quad \\varepsilon_i \\sim N(0, \\sigma_y^2).
](https://latex.codecogs.com/png.image?%5Cdpi%7B110%7D&space;%5Cbg_white&space;%0Ay_i%20%3D%20%5Cbeta_0%20%2B%20%5Cbeta_x%20x_i%20%2B%20%5Cbeta_z%20z_i%20%2B%20%5Cvarepsilon_i%2C%20%5Cquad%20%5Cvarepsilon_i%20%5Csim%20N%280%2C%20%5Csigma_y%5E2%29.%0A "
y_i = \beta_0 + \beta_x x_i + \beta_z z_i + \varepsilon_i, \quad \varepsilon_i \sim N(0, \sigma_y^2).
")

As described above, we specify the Berkson error model through a random
effect
![\\widetilde u\_{bi}](https://latex.codecogs.com/png.image?%5Cdpi%7B110%7D&space;%5Cbg_white&space;%5Cwidetilde%20u_%7Bbi%7D "\widetilde u_{bi}")
in the model of interest, so this becomes

![
\\begin{aligned}
  y_i &= \\beta_0 + \\beta_x r_i + \\beta_x u\_{bi} + \\beta_z z_i + \\varepsilon_i \\\\
  &= \\beta_0 + \\beta_x r_i + \\widetilde u\_{bi} + \\beta_z z_i + \\varepsilon_i.
\\end{aligned}
](https://latex.codecogs.com/png.image?%5Cdpi%7B110%7D&space;%5Cbg_white&space;%0A%5Cbegin%7Baligned%7D%0A%20%20y_i%20%26%3D%20%5Cbeta_0%20%2B%20%5Cbeta_x%20r_i%20%2B%20%5Cbeta_x%20u_%7Bbi%7D%20%2B%20%5Cbeta_z%20z_i%20%2B%20%5Cvarepsilon_i%20%5C%5C%0A%20%20%26%3D%20%5Cbeta_0%20%2B%20%5Cbeta_x%20r_i%20%2B%20%5Cwidetilde%20u_%7Bbi%7D%20%2B%20%5Cbeta_z%20z_i%20%2B%20%5Cvarepsilon_i.%0A%5Cend%7Baligned%7D%0A "
\begin{aligned}
  y_i &= \beta_0 + \beta_x r_i + \beta_x u_{bi} + \beta_z z_i + \varepsilon_i \\
  &= \beta_0 + \beta_x r_i + \widetilde u_{bi} + \beta_z z_i + \varepsilon_i.
\end{aligned}
")

The prior distributions are

-   ![\\boldsymbol{r} \\sim N(\\alpha_0 + \\alpha_z \\boldsymbol{z}, \\tau_x \\boldsymbol{I})](https://latex.codecogs.com/png.image?%5Cdpi%7B110%7D&space;%5Cbg_white&space;%5Cboldsymbol%7Br%7D%20%5Csim%20N%28%5Calpha_0%20%2B%20%5Calpha_z%20%5Cboldsymbol%7Bz%7D%2C%20%5Ctau_x%20%5Cboldsymbol%7BI%7D%29 "\boldsymbol{r} \sim N(\alpha_0 + \alpha_z \boldsymbol{z}, \tau_x \boldsymbol{I})"),
-   ![\\beta_0, \\beta_x, \\beta_z \\sim N(0, \\tau\_{\\beta})](https://latex.codecogs.com/png.image?%5Cdpi%7B110%7D&space;%5Cbg_white&space;%5Cbeta_0%2C%20%5Cbeta_x%2C%20%5Cbeta_z%20%5Csim%20N%280%2C%20%5Ctau_%7B%5Cbeta%7D%29 "\beta_0, \beta_x, \beta_z \sim N(0, \tau_{\beta})"),
    with
    ![\\tau\_{\\beta} = 0.001](https://latex.codecogs.com/png.image?%5Cdpi%7B110%7D&space;%5Cbg_white&space;%5Ctau_%7B%5Cbeta%7D%20%3D%200.001 "\tau_{\beta} = 0.001"),
-   ![\\alpha_0, \\alpha_z \\sim N(0, \\tau\_{\\alpha})](https://latex.codecogs.com/png.image?%5Cdpi%7B110%7D&space;%5Cbg_white&space;%5Calpha_0%2C%20%5Calpha_z%20%5Csim%20N%280%2C%20%5Ctau_%7B%5Calpha%7D%29 "\alpha_0, \alpha_z \sim N(0, \tau_{\alpha})"),
    with
    ![\\tau\_{\\alpha} = 0.0001](https://latex.codecogs.com/png.image?%5Cdpi%7B110%7D&space;%5Cbg_white&space;%5Ctau_%7B%5Calpha%7D%20%3D%200.0001 "\tau_{\alpha} = 0.0001")
-   ![\\tau\_{u_b} \\sim G(0.5, 0.5)](https://latex.codecogs.com/png.image?%5Cdpi%7B110%7D&space;%5Cbg_white&space;%5Ctau_%7Bu_b%7D%20%5Csim%20G%280.5%2C%200.5%29 "\tau_{u_b} \sim G(0.5, 0.5)"),
-   ![\\tau\_{u_c} \\sim G(0.5, 0.5)](https://latex.codecogs.com/png.image?%5Cdpi%7B110%7D&space;%5Cbg_white&space;%5Ctau_%7Bu_c%7D%20%5Csim%20G%280.5%2C%200.5%29 "\tau_{u_c} \sim G(0.5, 0.5)"),
-   ![\\tau\_{u_x} \\sim G(0.5, 0.5)](https://latex.codecogs.com/png.image?%5Cdpi%7B110%7D&space;%5Cbg_white&space;%5Ctau_%7Bu_x%7D%20%5Csim%20G%280.5%2C%200.5%29 "\tau_{u_x} \sim G(0.5, 0.5)").

We specify the priors in the code:

``` r
# Priors for model of interest coefficients
prior.beta = c(0, 1/1000) # N(0, 10^3)

# Priors for exposure model coefficients
prior.alpha <- c(0, 1/10000) # N(0, 10^4)

# Priors for measurement error variance and true x-value
prior.prec.y <- c(0.5, 0.5) # Gamma(0.5, 0.5)
prior.prec.u_b <- c(0.5, 0.5) # Gamma(0.5, 0.5)
prior.prec.u_c <- c(0.5, 0.5) # Gamma(0.5, 0.5)
prior.prec.x <- c(0.5, 0.5) # Gamma(0.5, 0.5)

# Initial values
prec.y <- 1
prec.u_b <- 1
prec.u_c <- 1
prec.x <- 1
```

The hierarchical model described in the above section is fit in INLA as
a joint model using the feature. We first specify the models in the
following matrices and vectors:

![matrixformulation](https://github.com/emmaSkarstein/ISBA2022-Poster/blob/main/Simulation%20example/matrix_equations.png)

We specify these matrices in our code:

``` r
Y <- matrix(NA, 3*n, 3)

Y[1:n, 1] <- y               # Regression model of interest response
Y[n+(1:n), 2] <- w           # Error model response
Y[2*n+(1:n), 3] <- rep(0, n) # Exposure model response

beta.0 <- c(rep(1, n), rep(NA, 2*n))
beta.x <- c(1:n, rep(NA, n), rep(NA, n))
u.b.tilde <- c(1:n, rep(NA, n), rep(NA, n))
beta.z <- c(z, rep(NA, 2*n))

id.r <- c(rep(NA, n), 1:n, 1:n)
weight.r <- c(rep(1, n), rep(1, n), rep(-1, n))

alpha.0 = c(rep(NA, n), rep(NA, n), rep(1, n))
alpha.z = c(rep(NA, n), rep(NA, n), z)
```

``` r
dd <- data.frame(Y = Y,
                 beta.0 = beta.0,
                 beta.x = beta.x,
                 u.b.tilde = u.b.tilde,
                 beta.z = beta.z,
                 id.r = id.r,
                 weight.r = weight.r,
                 alpha.0 = alpha.0,
                 alpha.z = alpha.z)
```

Next, we set up the INLA formula. There are four fixed effects
(![\\beta_0](https://latex.codecogs.com/png.image?%5Cdpi%7B110%7D&space;%5Cbg_white&space;%5Cbeta_0 "\beta_0"),
![\\beta_z](https://latex.codecogs.com/png.image?%5Cdpi%7B110%7D&space;%5Cbg_white&space;%5Cbeta_z "\beta_z"),
![\\alpha_0](https://latex.codecogs.com/png.image?%5Cdpi%7B110%7D&space;%5Cbg_white&space;%5Calpha_0 "\alpha_0"),
![\\alpha_z](https://latex.codecogs.com/png.image?%5Cdpi%7B110%7D&space;%5Cbg_white&space;%5Calpha_z "\alpha_z"))
and three random effects. Two of the random effects are necessary to
ensure that the values of
![\\boldsymbol{r}](https://latex.codecogs.com/png.image?%5Cdpi%7B110%7D&space;%5Cbg_white&space;%5Cboldsymbol%7Br%7D "\boldsymbol{r}")
are the same in the exposure model and error model are assigned the same
values as in the regression model, where
![\\beta_x \\boldsymbol{r}](https://latex.codecogs.com/png.image?%5Cdpi%7B110%7D&space;%5Cbg_white&space;%5Cbeta_x%20%5Cboldsymbol%7Br%7D "\beta_x \boldsymbol{r}")
is the product of two unknown quantities. The third random effect term
is for encoding the Berkson error model.

-   `f(beta.x, copy="id.r", ...)`: The `copy="id.r"` argument ensures
    that identical values are assigned to
    ![\\boldsymbol{r}](https://latex.codecogs.com/png.image?%5Cdpi%7B110%7D&space;%5Cbg_white&space;%5Cboldsymbol%7Br%7D "\boldsymbol{r}")
    in all components of the joint model.
    ![\\beta_x](https://latex.codecogs.com/png.image?%5Cdpi%7B110%7D&space;%5Cbg_white&space;%5Cbeta_x "\beta_x"),
    which is treated as a hyperparameter, is the scaling parameter of
    the copied process
    ![\\boldsymbol{r}^\*](https://latex.codecogs.com/png.image?%5Cdpi%7B110%7D&space;%5Cbg_white&space;%5Cboldsymbol%7Br%7D%5E%2A "\boldsymbol{r}^*").
-   `f(id.r, weight.r, ...)`: `id.r` contains the
    ![\\boldsymbol{c}](https://latex.codecogs.com/png.image?%5Cdpi%7B110%7D&space;%5Cbg_white&space;%5Cboldsymbol%7Bc%7D "\boldsymbol{c}")-values,
    encoded as an i.i.d. Gaussian random effect, and weighted with
    `weight.c` to ensure the correct signs in the joint model. \[The
    `values` option contains the vector of all values assumes by the
    covariate for which the effect is estimated. What does this mean?\]
    The precision `prec` of the random effect is fixed at
    ![\\exp(-15)](https://latex.codecogs.com/png.image?%5Cdpi%7B110%7D&space;%5Cbg_white&space;%5Cexp%28-15%29 "\exp(-15)"),
    which is necessary since the uncertainty in
    ![\\boldsymbol{c}](https://latex.codecogs.com/png.image?%5Cdpi%7B110%7D&space;%5Cbg_white&space;%5Cboldsymbol%7Bc%7D "\boldsymbol{c}")
    is already modeled in the second level (column 2 of `Y`) of the
    joint model, which defines the exposure component.
-   `f(u.b.tilde, ...)`: This is a Gaussian random effect that ensures
    that we capture the additional variance due to the Berkson
    measurement error. \[why do we use “values” here?\]

``` r
formula = Y ~ beta.0 - 1 +
  f(beta.x, copy="id.r",
    hyper = list(beta = list(param = prior.beta, fixed=FALSE))) +
  f(id.r, weight.r, model="iid", values = 1:n,
    hyper = list(prec = list(initial = -15, fixed=TRUE))) +
  f(u.b.tilde, model = "iid", values = 1:n,
    hyper = list(prec = list(initial = log(1), fixed=TRUE))) +
  beta.z + alpha.0 + alpha.z
```

We explicitly remove the intercept using `-1` since there is no common
intercept in the joint model, and the model specific intercepts
![\\beta_0](https://latex.codecogs.com/png.image?%5Cdpi%7B110%7D&space;%5Cbg_white&space;%5Cbeta_0 "\beta_0")
and
![\\alpha_0](https://latex.codecogs.com/png.image?%5Cdpi%7B110%7D&space;%5Cbg_white&space;%5Calpha_0 "\alpha_0")
are specified instead.

Next comes the call of the `inla` function. We explain further some of
the terms:

-   `family`: Here we need to specify one likelihood function for each
    of the model levels corresponding to each column in the matrix `Y`.
    In this case, they are all Gaussian, but if we for instance had a
    logistic regression model as our model of interest, then the list
    would be `c("binomial", "gaussian", "gaussian")`.
-   `control.family`: Here we specify the hyperparameters for each of
    the three likelihoods. In this case, we specify the precision for
    each Gaussian likelihood,
    ![\\tau_y](https://latex.codecogs.com/png.image?%5Cdpi%7B110%7D&space;%5Cbg_white&space;%5Ctau_y "\tau_y"),
    ![\\tau\_{u_c}](https://latex.codecogs.com/png.image?%5Cdpi%7B110%7D&space;%5Cbg_white&space;%5Ctau_%7Bu_c%7D "\tau_{u_c}")
    and
    ![\\tau\_{x}](https://latex.codecogs.com/png.image?%5Cdpi%7B110%7D&space;%5Cbg_white&space;%5Ctau_%7Bx%7D "\tau_{x}"),
    respectively.
-   `control.predictor`: Compute the predictive distribution of the
    missing observations in the response.
-   `control.fixed`: Prior specification for the fixed effects.

``` r
model1 <- inla(formula, data = dd, 
                  family = c("gaussian", "gaussian", "gaussian"),
                  control.family = list(
                    list(hyper = list(prec = list(initial = log(prec.y),
                                                  param = prior.prec.y,
                                                  fixed = FALSE))),
                    list(hyper = list(prec = list(initial = log(prec.u_c),
                                                  param = prior.prec.u_c,
                                                  fixed = TRUE))),
                    list(hyper = list(prec = list(initial = log(prec.x),
                                                  param = prior.prec.x,
                                                  fixed = FALSE)))
                  ),
                  control.predictor = list(compute = TRUE), 
                  control.fixed = list(
                    mean = list(beta.0 = prior.beta[1],
                                beta.z = prior.beta[1],
                                alpha.0 = prior.alpha[1],
                                alpha.z = prior.alpha[1]),
                    prec = list(beta.0 = prior.beta[2],
                                beta.z = prior.beta[2],
                                alpha.0 = prior.alpha[2],
                                alpha.z = prior.alpha[2]))
               )
```

# Results

``` r
summary(model1)
#> 
#> Call:
#>    c("inla.core(formula = formula, family = family, contrasts = contrasts, 
#>    ", " data = data, quantiles = quantiles, E = E, offset = offset, ", " 
#>    scale = scale, weights = weights, Ntrials = Ntrials, strata = strata, 
#>    ", " lp.scale = lp.scale, link.covariates = link.covariates, verbose = 
#>    verbose, ", " lincomb = lincomb, selection = selection, control.compute 
#>    = control.compute, ", " control.predictor = control.predictor, 
#>    control.family = control.family, ", " control.inla = control.inla, 
#>    control.fixed = control.fixed, ", " control.mode = control.mode, 
#>    control.expert = control.expert, ", " control.hazard = control.hazard, 
#>    control.lincomb = control.lincomb, ", " control.update = 
#>    control.update, control.lp.scale = control.lp.scale, ", " 
#>    control.pardiso = control.pardiso, only.hyperparam = only.hyperparam, 
#>    ", " inla.call = inla.call, inla.arg = inla.arg, num.threads = 
#>    num.threads, ", " blas.num.threads = blas.num.threads, keep = keep, 
#>    working.directory = working.directory, ", " silent = silent, inla.mode 
#>    = inla.mode, safe = FALSE, debug = debug, ", " .parent.frame = 
#>    .parent.frame)") 
#> Time used:
#>     Pre = 3.83, Running = 1.59, Post = 0.0842, Total = 5.5 
#> Fixed effects:
#>          mean    sd 0.025quant 0.5quant 0.975quant mode kld
#> beta.0  1.150 0.063      1.025    1.150      1.273   NA   0
#> beta.z  1.038 0.057      0.926    1.038      1.148   NA   0
#> alpha.0 0.982 0.059      0.867    0.982      1.097   NA   0
#> alpha.z 0.597 0.061      0.477    0.597      0.717   NA   0
#> 
#> Random effects:
#>   Name     Model
#>     id.r IID model
#>    u.b.tilde IID model
#>    beta.x Copy
#> 
#> Model hyperparameters:
#>                                             mean    sd 0.025quant 0.5quant
#> Precision for the Gaussian observations    7.935 2.926      4.010    7.338
#> Precision for the Gaussian observations[3] 0.500 0.036      0.430    0.499
#> Beta for beta.x                            0.944 0.041      0.863    0.944
#>                                            0.975quant mode
#> Precision for the Gaussian observations        15.296   NA
#> Precision for the Gaussian observations[3]      0.573   NA
#> Beta for beta.x                                 1.025   NA
#> 
#> Marginal log-Likelihood:  -11733.30 
#>  is computed 
#> Posterior summaries for the linear predictor and the fitted values are computed
#> (Posterior marginals needs also 'control.compute=list(return.marginals.predictor=TRUE)')
```
