require(nnet)
require(FNN)
require(randomForest)
library(R.utils)
library(party)
library(tree)
library(plotly)
require(gridExtra)
require(GGally)
require(smooth)
require(zoo)
require(iml)
require(rmutil)
library(latex2exp)

dnvgl <- c(
  rgb(153, 214, 240, maxColorValue = 255),
  rgb(0, 53, 145, maxColorValue = 255),
  rgb(63, 156, 53, maxColorValue = 255),
  rgb(15, 32, 75, maxColorValue = 255),
  rgb(0, 159, 218, maxColorValue = 255),
  rgb(254, 203, 0, maxColorValue = 255),
  rgb(233, 131, 0, maxColorValue = 255),
  rgb(110, 80, 145, maxColorValue = 255),
  rgb(196, 38, 46, maxColorValue = 255),
  rgb(152, 143, 134, maxColorValue = 255),
  rgb(0, 0, 0, maxColorValue = 255)
)
dnvgl <- rep(dnvgl, 5)
palette(dnvgl)

library(logger)
log_info("XAI Shapley Cluster")

# TODO: Read paper
# Auto Associative Kernel Regression
AAKR <- function(X_test, X_train, h = .2, d2_skip = 0, y_importance = 0) {
  K <- dim(X_train)[1]
  N <- dim(X_train)[2]
  M <- dim(X_test)[1]
  d2 <- matrix(NA, K, N)
  w <- array(NA, K)
  X_pred_n <- matrix(NA, M, N)
  X_test_n <- X_test # normalize(X_test, X_train)  ## IF WE ASSUME THE INPUT IS ALREADY NORMALIZED
  X_train_n <- X_train # normalize(X_train, X_train)
  for (m in 1:M) {
    for (k in 1:K) {
      d2[k, ] <- (X_test_n[m, ] - X_train_n[k, ])^2
      if (d2_skip > 0) {
        d2[k, d2_skip] <- 0
      }
      d2[k, N] <- d2[k, N] * y_importance
      w[k] <- 1 / (sqrt(2 * pi) * h) * exp(-1 / (2 * h * h) * sum(d2[k, ]))
    }
    X_pred_n[m, ] <- (w %*% X_train_n) / sum(w)
  }
  X_test_pred <- X_pred_n
  return(X_test_pred)
}


normalize <- function(X, X_base) {
  X_n <- X * 0
  if (dim(as.matrix(X))[2] == 1) {
    X_n <- (X - mean(X_base)) / sd(X_base)
  } else {
    for (j in 1:dim(as.matrix(X))[2]) {
      X_n[, j] <- (X[, j] - mean(X_base[, j])) / sd(X_base[, j])
    }
  }
  return(X_n)
}


fn_prediction <- function(data_train, data_test, method, ntree = 100, maxnodes = 30) {
  if (method == 'lm') {
    fit <- lm(formula = y ~ x, data = data.frame(y = data_train[, 1], x = I(as.matrix(data_train[, -1]))))
    pred <- predict.lm(fit, newdata = data.frame(x = I(as.matrix(data_test[, -1]))))
  }
  if (method == 'lm0') {
    fit <- lm(formula = y ~ x + 0, data = data.frame(y = data_train[, 1], x = I(as.matrix(data_train[, -1]))))
    pred <- predict.lm(fit, newdata = data.frame(x = I(as.matrix(data_test[, -1]))))
  }
  if (method == 'intercept') {
    fit <- mean(data_train[, 1]) #lm(formula = y~x,data=data.frame(y=data_train[,1],x=I(as.matrix(0*data_train[,-1]))))
    pred <- array(fit, dim(data_test)[1]) #predict.lm(fit,newdata =data.frame(x=I(as.matrix(data_test[,-1]))))
  }
  if (method == 'mybinom') {
    fit <- mean(data_train[, 1]) #lm(formula = y~x,data=data.frame(y=data_train[,1],x=I(as.matrix(0*data_train[,-1]))))
    pred <- rbinom(n = dim(data_test)[1], size = 1, prob = fit)
  }
  if (method == 'knn') {
    fit <- knn.reg(train = data_train[, -1], test = data_test[, -1], y = data_train[, 1], k = 1, algorithm = "kd_tree")
    pred <- fit$pred
  }
  if (method == 'knn10') {
    fit <- knn.reg(train = data_train[, -1], test = data_test[, -1], y = data_train[, 1], k = 10, algorithm = "kd_tree")
    pred <- fit$pred
  }
  if (method == 'nnet') {
    fit <- nnet(x = data_train[, -1], y = data_train[, 1], maxit = 500, size = 5, linout = T, trace = F)
    pred <- predict(fit, newdata = data_test[, -1])
  }
  if (method == 'rf') {
    fit <- randomForest(x = data_train[, -1], y = data_train[, 1], ntree = ntree, maxnodes = maxnodes)
    pred <- predict(fit, newdata = matrix(data_test[, -1], ncol = dim(data_train)[2] - 1))
  }
  return(pred)
}


fn_prediction_error <- function(y, y_hat, metric) {
  if (metric == 'RMSE') {
    PE <- sqrt(mean((y - y_hat)^2, na.rm = T))
  }
  if (metric == 'MSE') {
    PE <- mean((y - y_hat)^2, na.rm = T)
  }
  if (metric == 'MAE') {
    PE <- mean(abs(y - y_hat), na.rm = T)
  }
  if (metric == 'sumSE') {
    PE <- sum((y - y_hat)^2)
  }
  if (metric == 'maxSE') {
    PE <- max((y - y_hat)^2, na.rm = T)
  }
  return(PE)
}


# Init
mmetric <- 'MSE'
mmethod <- 'rf' # Which regression model to use

prediction_accuracy <- TRUE
global_classification <- FALSE

K <- 5 # Number of clusters
N <- 100 * K * 4 # Number of datapoints

N_train <- N / 4
N_test <- N / 4
N_eval <- N / 2

set.seed(2)


# Data generation
x1 <- sin(seq(1, 2 * 2.4 * pi, length.out = N)) + rnorm(N, 0, 0.1)
x2 <- sin(seq(1, 2 * 20.45 * pi, length.out = N)) + rnorm(N, 0, 0.1)
x3 <- sin(seq(1, 2 * 4.21 * pi, length.out = N)) + rnorm(N, 0, 0.1)
x4 <- sin(seq(1, 2 * 8.15 * pi, length.out = N)) + rnorm(N, 0, 0.1)

# Assign cluster labels to each datapoint
# N / 4 is number of datapoints per x1 -> x4
# [1, 1, ..., 5, 5, 1, 1, ..., 5, 5, 1, 1, ..., 5, 5, 1, 1, ..., 5, 5]
xS <- rep(floor(seq(1, K + 1, length.out = N / 4 + 1))[1:(N / 4)], 4)
# print(xS)

y <- x1 * x2 + x3 * x4 + rnorm(N, 0, 0.1)

mdata_full <- cbind(y, x1, x2, x3, x4, xS)
include_mdata <- c(1, 2, 3, 4, 5) # y, x1, x2, x3, x4
index_mdata_xS <- 6 # xS

# Copy data from cluster 4 to cluster 5
mdata_full[which(xS == 5), include_mdata] <- mdata_full[which(xS == 4), include_mdata]

# Prepare training data
mdata_train_full <- mdata_full[1:(floor(N_train)), ]
mdata_test_full <- mdata_full[(floor(N_train) + 1):(floor(N_train) + floor(N_test)), ]
mdata_eval_full <- mdata_full[(floor(N_train) + floor(N_test) + 1):(floor(N)), ]

# Sort training data by cluster labels
segment_rule <- order((mdata_train_full[, index_mdata_xS]))
mdata_train_full <- mdata_train_full[segment_rule, ]

# mdata is mdata_full without the cluster labels (xS)
mdata <- rbind(mdata_train_full, mdata_test_full, mdata_eval_full)
mdata <- mdata[, include_mdata]

mdata_train <- mdata[1:(floor(N_train)), ]
mdata_test <- mdata[(floor(N_train) + 1):(floor(N_train) + floor(N_test)), ]
mdata_eval <- mdata[(floor(N_train) + floor(N_test) + 1):(floor(N)), ]

# Ensure matching row counts
N_train <- dim(mdata_train)[1]
N_test <- dim(mdata_test)[1]
N_eval <- dim(mdata_eval)[1]
log_info("N_train: {N_train}")
log_info("N_test: {N_test}")
log_info("N_eval: {N_eval}")

# Per cluster
K_train <- floor(N_train / K)
K_test <- floor(N_test / K)
K_eval <- floor(N_eval / K)
log_info("K_train: {K_train}")
log_info("K_test: {K_test}")
log_info("K_eval: {K_eval}")

# dim(mdata)[2] is y, x1, x2, x3, x4
# J is 4 - the number of features (x1, x2, x3, x4)
J <- dim(mdata)[2] - 1

# Plot train data
# Assign colors to each cluster for training data
# [ 1, 1, ..., 1, 2, 2, ..., 2, 3, 3, ..., 3, 4, 4, ..., 4, 5, 5, ..., 5]
col_array_train <- array(NA, N_train)
for (k in 1:K) {
  col_array_train[((k - 1) * K_train + 1):(K_train * k)] <- rep(k, K_train)
}
# print(col_array_train)

par(mar = c(2, 5, 2, 2)) # margin bottom, left, top, right
par(mfrow = c(length(include_mdata), 1))

# Plot y, x1, x2, x3, x4
for (j in include_mdata) {
  plot(mdata_train[, j], col = col_array_train, ylab = colnames(mdata_full)[j])
}

mtext("Train data", side = 3, line = -1.5, outer = TRUE)

set.seed(21)

# Set test same as train
mdata_test <- mdata_train

# Insert anomalies into the test data
actual_states <- rep(0, dim(mdata_test)[1])

if (global_classification) {
  actual_states[200:249] <- rep(1, 50)
  actual_states[300:349] <- rep(1, 50)
  mdata_test[200:249, 2] <- mdata_test[200:249, 2] + rnorm(50, -0.5, 0.5) # x1
  mdata_test[300:349, 5] <- mdata_test[300:349, 5] + rnorm(50, -0.5, 0.5) # x4

  # Plot test data with anomalies
  par(mar = c(2, 5, 2, 2))
  par(mfrow = c(length(include_mdata), 1))

  # Plot y, x1, x2, x3, x4
  for (j in include_mdata) {
    plot(mdata_test[, j], col = col_array_train, ylab = colnames(mdata_full)[j])
    # Plot anomalies
    if (j == 2) {
      points(200:249, mdata_test[200:249, 2], col = 6, pch = 16)
    }
    if (j == 5) {
      points(300:349, mdata_test[300:349, 5], col = 6, pch = 16)
    }
  }

  mtext("Test data with anomalies", side = 3, line = -1.5, outer = TRUE)

  X_test_pred <- AAKR(X_test = mdata_test, X_train = mdata_train)

  # Plot AAKR
  par(mar = c(2, 5, 2, 2))
  par(mfrow = c(length(include_mdata), 1))

  # Plot y, x1, x2, x3, x4
  for (j in include_mdata) {
    plot(mdata_train[, j], col = col_array_train, ylab = colnames(mdata_full)[j], ylim = c(-1, 1))
    points(X_test_pred[, j], col = 6, pch = 16)
  }

  mtext("Compare train data with AAKR predictions", side = 3, line = -1.5, outer = TRUE)

  par(mar = c(2, 5, 2, 2))
  par(mfrow = c(length(include_mdata), 1))

  # Plot y, x1, x2, x3, x4
  for (j in include_mdata) {
    plot(mdata_test[, j], col = col_array_train, ylab = colnames(mdata_full)[j], ylim = c(-1, 1))
    points(X_test_pred[, j], col = 6, pch = 16)
  }

  mtext("Compare test data with AAKR predictions", side = 3, line = -1.5, outer = TRUE)
}


fn_accuracy <- function(actual_states, X_train, X_test) {
  X_test_pred <- AAKR(X_test = X_test, X_train = X_train)
  my_res <- (X_test_pred - X_test)
  my_res_max <- apply(abs(my_res), 1, sum)
  predicted_states <- abs(my_res_max) > .5

  TP <- sum(actual_states == 1 & predicted_states == 1)
  FP <- sum(actual_states == 0 & predicted_states == 1)
  FN <- sum(actual_states == 1 & predicted_states == 0)
  TN <- sum(actual_states == 0 & predicted_states == 0)

  acc <- (TP + TN) / (TP + FP + TN + FN)
  return(acc)
}

M <- 250

phim <- array(NA, dim = c(N_test, K, M))
phi <- array(NA, dim = c(N_test, K, M))
phi_pred_p <- array(NA, dim = c(N_test, K, M))
phi_pred_idx <- array(NA, dim = c(M, K))

# Split data into K clusters
mdata_train_k <- array(NA, dim = c(K_train, J + 1, K))
mdata_test_k <- array(NA, dim = c(K_test, J + 1, K))
mdata_eval_k <- array(NA, dim = c(K_eval, J + 1, K))

for (k in 1:K) {
  # y, x1, x2, x3, x4
  for (j in 1:(J + 1)) {
    mdata_train_k[, j, k] <- mdata_train[(K_train * (k - 1) + 1):(K_train * k), j]
    mdata_test_k[, j, k] <- mdata_test[(K_test * (k - 1) + 1):(K_test * k), j]
    mdata_eval_k[, j, k] <- mdata_eval[(K_eval * (k - 1) + 1):(K_eval * k), j]
  }
}

set.seed(7)

for (k in 1:K) {
  log_info("Cluster {k}")
  for (m in 1:M) {
    cluster_permutation <- matrix(sample(1:K), nrow = 1)
    # print(cluster_permutation)

    # D- contains clusters preceding k in the permutation; D+ also contains k.
    cluster_position <- which(cluster_permutation == k)
    data_train_p <- R.utils::wrap(
      mdata_train_k[,, cluster_permutation[1:cluster_position]],
      map = list(NA, 2)
    )
    if (cluster_position == 1) {
      # The paper defines f_empty(x) = 0.  No model is trained for D-.
      data_train_m <- NULL
    } else {
      data_train_m <- R.utils::wrap(
        mdata_train_k[,, cluster_permutation[1:(cluster_position - 1)]],
        map = list(NA, 2)
      )
    }

    if (global_classification) {
      # v = accuracy
      # accuracy(D+) - accuracy(D-)
      if (cluster_position == 1) {
        phim[, k, m] <- fn_accuracy(actual_states, X_train = data_train_p, mdata_test)
      } else {
        phim[, k, m] <- fn_accuracy(actual_states, X_train = data_train_p, mdata_test) -
          fn_accuracy(actual_states, X_train = data_train_m, mdata_test)
      }
    } else {
      if (prediction_accuracy) {
        # v = (y - f(x))^2 - y^2
        phi_pred_p[, k, m] <- (mdata_test[, 1] -
          (fn_prediction(data_train = data_train_p, data_test = mdata_test, method = mmethod)))^2
        phi_pred_idx[m, k] <- which(cluster_permutation == k)

        if (cluster_position == 1) {
          phim[, k, m] <- phi_pred_p[, k, m] - ((mdata_test[, 1] - 0))^2
        } else {
          phim[, k, m] <- phi_pred_p[, k, m] -
            (mdata_test[, 1] - fn_prediction(data_train = data_train_m, data_test = mdata_test, method = mmethod))^2
        }
      } else {
        # v = f(x)
        # prediction(D+) - prediction(D-)
        phi_pred_p[, k, m] <- fn_prediction(data_train = data_train_p, data_test = mdata_test, method = mmethod)
        phi_pred_idx[m, k] <- which(cluster_permutation == k)

        if (cluster_position == 1) {
          phim[, k, m] <- phi_pred_p[, k, m] - 0
        } else {
          phim[, k, m] <- phi_pred_p[, k, m] -
            fn_prediction(data_train = data_train_m, data_test = mdata_test, method = mmethod)
        }
      }
    }

    # Shapley value for cluster k up to m
    phi[, k, m] <- apply(phim[, k, ], MARGIN = 1, FUN = mean, na.rm = T)
  }
}


selected_points <- c(50, 150, 250, 350, 450)
full_prediction <- fn_prediction(data_train = mdata_train, data_test = mdata_test, method = mmethod)

par(mar = c(2, 3, 2, 2))
layout(
  matrix(
    c(
      rep(1, length(selected_points)),
      2:(length(selected_points) + 1),
      (length(selected_points) + 2):(2 * length(selected_points) + 1)
    ),
    nrow = 3,
    ncol = length(selected_points),
    byrow = TRUE
  ),
  respect = TRUE
)

if (!prediction_accuracy) {
  plotted_value <- full_prediction
  plot_title <- "Predictions"
  y_label <- "Prediction"
} else {
  plotted_value <- (full_prediction - mdata_test[, 1])^2
  plot_title <- "Squared Error"
  y_label <- "Squared error"
}

# Top row: full-model prediction or squared error
plot(seq_along(plotted_value), plotted_value, xaxs = "i", ylab = y_label, main = "", type = "l", col = 11, lwd = 3)
for (point_index in selected_points) {
  points(point_index, plotted_value[point_index], pch = 16, cex = 1.5, col = 9)
  abline(v = point_index, lty = 2)
}
title(plot_title, line = 0)

# Middle row: final local Shapley values for each selected test point.
for (point_index in selected_points) {
  barplot(phi[point_index, , M], horiz = TRUE, col = seq_len(K), main = "")
  box()
}

# Bottom row: convergence of the Monte-Carlo Shapley estimates.
for (point_index in selected_points) {
  point_phi <- phi[point_index, , , drop = FALSE]
  plot(NA, xlim = c(1, M), ylim = range(point_phi, na.rm = TRUE), xlab = "", ylab = "", axes = FALSE)
  axis(1, labels = FALSE)
  axis(2, labels = FALSE)
  abline(h = 0, lty = 3)
  for (cluster_index in seq_len(K)) {
    lines(seq_len(M), phi[point_index, cluster_index, ], col = cluster_index)
  }
}


# Efficiency

PP <- 100
P_empty <- array(NA, dim = c(N / 4, PP))
P_N <- array(NA, dim = c(N / 4, PP))
for (p in 1:PP) {
  Z <- mdata_train
  for (j in 2:(J + 1)) {
    oo <- sample(x = seq(from = 1, to = dim(mdata_train)[1], by = 1), size = N / 4, replace = TRUE)
    Z[, j] <- mdata_train[oo, j]
  }
  if (prediction_accuracy == FALSE) {
    P_empty[, p] <- 0 #(myPred(data_train = Z,data_test = mydata_test,method = mymethod))
    P_N[, p] <- (fn_prediction(data_train = mdata_train, data_test = mdata_test, method = mmethod))
  } else {
    P_empty[, p] <- 0 #(mydata_test[,1]^2)
    P_N[, p] <- ((mdata_test[, 1] -
      fn_prediction(data_train = mdata_train, data_test = mdata_test, method = mmethod))^2 -
      mdata_test[, 1]^2)
  }
}

par(mfrow = c(3, 1))
par(mar = c(2, 2, 2, 2))
plot(P_empty[, p], col = 0, ylim = c(-1.3, 1.8), main = TeX('$v(N)  =  E(f_N)  -  E(f_X)$'))
for (p in 1:PP) {
  lines(P_empty[, p], col = 10, lwd = 2)
  lines(P_N[, p], col = 9, lwd = 1)
}
#lines(rowMeans(P_N),col=8,lwd=1,lty=1)
#lines(rowMeans(P_empty),col=9,lwd=2,lty=3)
legend("topright", legend = c(TeX('f_N'), TeX('$f_X$')), col = c(9, 10), lty = c(1, 1), cex = 0.8)
plot(phi[, 1, M], col = 0, ylim = c(-1.3, 1.8), main = TeX('$\\varphi$'))
for (k in 1:K) {
  lines(phi[, k, M], col = k, lty = 1)
}
lines(apply(X = phi[,, M], MARGIN = 1, FUN = 'sum'), col = 11, lwd = 2)
legend(
  "topright",
  legend = c(
    TeX('$\\varphi_1$        '),
    (TeX('$\\varphi_2$')),
    TeX('$\\varphi_3   $ '),
    TeX('$\\varphi_4$'),
    TeX('$\\varphi_5$'),
    c(TeX('    $  \\sum   \\varphi_k    $   '), '        ')
  ),
  col = c(1:5, 11),
  lty = c(3, 3, 3, 3, 3, 1, 0),
  lwd = c(1, 1, 1, 1, 1, 2)
)
plot(rowMeans(P_N) - rowMeans(P_empty), type = 'l', lwd = 1, col = 9, main = TeX('$\\sum\\varphi_k$ =v(N)'))
for (p in 1:PP) {
  lines(P_N[, p], col = 9, lwd = 1)
}
lines(apply(X = phi[,, M], MARGIN = 1, FUN = 'sum'), col = 11, lwd = 2, type = 'l', lty = 1)
legend(
  "topright",
  legend = c((TeX('$v(N)$')), TeX('$\\sum\\varphi_k$        ')),
  col = c(9, 11),
  lty = c(1, 1),
  lwd = c(1, 2)
)


y_hat_test <- fn_prediction(data_train = mdata_train, data_test = mdata_test, method = mmethod)
#
# # sum phi
# colMeans(phi[,,M])
# sum(colMeans(phi[,,M]))
#
# sum(colMeans(phi[,,M]))/sum(abs(colMeans(phi[,,M])))*100
#

# SYMMETRY

par(mfrow = c(1, 1))

plot(phi[, 4, M], type = 'l', col = 4, lwd = 2)
lines(phi[, 5, M], col = 5, lty = 3, lwd = 2)
legend(
  "topleft",
  legend = c(TeX('$\\varphi_4$'), TeX('$\\varphi_5$'), '        '),
  col = 4:5,
  lty = c(1, 3, 0, 3, 3, 1, 0),
  lwd = c(2, 2, 1, 1, 1, 2)
)

#plot((phi[,4,M]-phi[,5,M])/(max(phi[,4:5,M])-min(phi[,4:5,M])))

par(mfrow = c(5, 1))
plot(phi[, 1, M], col = 1, type = 'o', ylim = c(min(phi[,, M]), max(phi[,, M])))
abline(h = 0)
for (k in 2:K) {
  points(phi[, k, M], col = k, type = 'o')
}
if (prediction_accuracy == TRUE) {
  plot((mdata_test[, 1] - y_hat_test)^2, type = 'o')
} else {
  plot((mdata_test[, 1]), type = 'o')
  lines(y_hat_test, col = K + 1, type = 'o')
}
#lines(y_hat_test,col='red')

plot(mdata_train[, 1], col = mdata_train_full[, 6], ylim = c(-3, 3), type = 'o')
plot(mdata_train[, 2], col = mdata_train_full[, 6], ylim = c(-3, 3), type = 'o')
plot(mdata_train[, 3], col = mdata_train_full[, 6], ylim = c(-3, 3), type = 'o')

############
par(mfrow = c(2, 1))
plot(phi[, 1, M], col = 1, type = 'l', ylim = c(min(phi[,, M]), max(phi[,, M])))
abline(h = 0)
for (k in 2:K) {
  points(phi[, k, M], col = k, type = 'l')
}

if (prediction_accuracy == TRUE) {
  plot((mdata_test[, 1] - phi_pred_p[, 1, 1])^2, type = 'l', lwd = 0.4, col = 0, lty = 1)
  for (k in 1:K) {
    for (m in round(seq(from = 1, to = M, length.out = 5))) {
      lines((mdata_test[, 1] - phi_pred_p[, k, m])^2, col = k, pch = '.', lty = 1)
    }
    # for( k in 1:K){
    #   lines((mydata_test[,1]-myPred(data_train = mydata_train_k[,,k],data_test=mydata_test,method=mymethod))^2,col=k,lty=1)
    # }
  }
  lines((mdata_test[, 1] - phi_pred_p[, 1, M])^2, type = 'l', col = 'red', lwd = 2)
  lines(mdata_test[, 1] - mdata_test[, 1], col = 1, lwd = 2)
} else {
  plot(phi_pred_p[, 1, 1], type = 'l', lwd = 0.4, col = 0, lty = 1, ylim = c(min(phi_pred_p), max(phi_pred_p)))
  for (k in 1:K) {
    for (m in round(seq(from = 1, to = M, length.out = 20))) {
      lines(phi_pred_p[, k, m], col = k, pch = '.', lty = 1)
    }
    Sys.sleep(2)
  }
  lines(phi_pred_p[, 1, M], type = 'l', col = 'red', lwd = 2)
  lines(mdata_test[, 1], col = 1, lwd = 2)
}
# for( k in 1:K){
#   lines(myPred(data_train = mydata_train_k[,,k],data_test=mydata_test,method=mymethod),col=k,lty=3)
# }

par(mar = c(2, 1, 2, 2) * .5)
par(mfrow = c(2, 1))
if (prediction_accuracy == FALSE) {
  plot(1:dim(mdata_temp)[1], y_hat_temp[], xaxs = "i", ylab = 'y test', main = '', type = 'l', col = 0)
  for (k in 1:K) {
    for (m in round(seq(from = 1, to = M, length.out = M))) {
      lines(phi_pred_p[, k, m], col = alpha(k, 0.1), pch = '.', lty = 1)
    }
  }
  lines(1:dim(mdata_temp)[1], y_hat_temp, xaxs = "i", col = 9, ylab = 'y test', main = '', lwd = 3)
  #abline(h=0,lty=3,lwd=0.5)
  title('Predictions', 2)
  lines(1:dim(mdata_temp)[1], mdata_temp[, 1], xaxs = "i", col = 11, ylab = 'y test', main = '', lwd = 3)

  plot(phi[, k, M], col = 0)
  for (k in 1:K) {
    for (m in 1:M) {
      lines(phi[, k, m], col = alpha(k, m / M))
    }
  }
}


# par(mfrow=c(1,1))
# par(mar=c(3,3,3,3))
# plot(mydata_test[,1],col=0)
# for (m in 1:M){
#   for (k in 1:K){
#     lines((phi_pred_p[,k,m]),col=alpha(k, 0.1))
#   }
# }

MSE_lc <- array(NA, dim = c(M, K))
for (m in 1:M) {
  for (k in 1:K) {
    MSE_lc[m, k] <- mean((phi_pred_p[, k, m]))
  }
}

par(mfrow = c(1, 1))
par(mar = c(2, 2, 1, 1) * 2)
plot(
  0:K,
  col = 0,
  ylim = c(min(MSE_lc), max(MSE_lc)),
  xlim = c(0.5, K),
  xlab = 'Number of subsets in the training data',
  ylab = 'MSE'
)
for (k in 1:K) {
  for (m in 150:250) {
    #points(c(0,phi_pred_idx[m,k]),c(empty,MSE_lc[m,k]),col=alpha(k, 0.05),pch='-',cex=8)
    points(c(phi_pred_idx[m, k]) + runif(1, -.2, .2), c(MSE_lc[m, k]), col = alpha(k, 1), pch = '.', cex = 8)
  }
}

MLC <- array(NA, dim = c(K, K))
empty <- mean(mdata_test[, 1]^2)
for (k in 1:K) {
  # look at each fold
  for (kk in 1:K) {
    # how many in phi_pred_idx
    MLC[kk, k] <- mean(MSE_lc[which(phi_pred_idx[, k] == kk), k])
  }
}
for (k in 1:K) {
  #lines(0:K,c(empty,MLC[,k]),col=k,type='o',pch=15,lwd=2)
  lines(1:K, c(MLC[, k]), col = k, type = 'o', pch = 15, lwd = 2)
}
legend("topright", legend = c(1:5), col = c(1:5, 11), lty = 1, lwd = c(1, 1, 1, 1, 1, 2), title = 'Subset included')
