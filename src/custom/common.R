library(logger)
library(progress)
require(FNN)
require(nnet)
require(randomForest)

fn_prediction <- function(data_train, data_test, method, ntree = 100, maxnodes = 30) {
  if (method == 'lm') {
    # Linear regression with intercept
    fit <- lm(formula = y ~ x, data = data.frame(y = data_train[, 1], x = I(as.matrix(data_train[, -1]))))
    pred <- predict.lm(fit, newdata = data.frame(x = I(as.matrix(data_test[, -1]))))
  }
  if (method == 'lm0') {
    # Linear regression without intercept
    fit <- lm(formula = y ~ x + 0, data = data.frame(y = data_train[, 1], x = I(as.matrix(data_train[, -1]))))
    pred <- predict.lm(fit, newdata = data.frame(x = I(as.matrix(data_test[, -1]))))
  }
  if (method == 'knn') {
    # 1-nearest neighbor
    fit <- knn.reg(train = data_train[, -1], test = data_test[, -1], y = data_train[, 1], k = 1, algorithm = "kd_tree")
    pred <- fit$pred
  }
  if (method == 'knn10') {
    # 10-nearest neighbor
    fit <- knn.reg(train = data_train[, -1], test = data_test[, -1], y = data_train[, 1], k = 10, algorithm = "kd_tree")
    pred <- fit$pred
  }
  if (method == 'nnet') {
    # Neural network with 5 hidden units
    fit <- nnet(x = data_train[, -1], y = data_train[, 1], maxit = 500, size = 5, linout = TRUE, trace = FALSE)
    pred <- predict(fit, newdata = data_test[, -1])
  }
  if (method == 'rf') {
    # Random forest
    fit <- randomForest(x = data_train[, -1], y = data_train[, 1], ntree = ntree, maxnodes = maxnodes)
    pred <- predict(fit, newdata = matrix(data_test[, -1], ncol = dim(data_train)[2] - 1))
  }
  return(pred)
}

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
    for (j in seq_len(dim(as.matrix(X))[2])) {
      X_n[, j] <- (X[, j] - mean(X_base[, j])) / sd(X_base[, j])
    }
  }
  return(X_n)
}

fn_shapley_cluster_global_classification <- function(
  K,
  M,
  data_train_k,
  data_test,
  actual_states
) {
  N_test <- dim(data_test)[1]
  phim <- array(NA, dim = c(N_test, K, M))
  phi <- array(NA, dim = c(N_test, K, M))

  log_info("Calculating Shapley values global classification for each cluster")
  for (k in 1:K) {
    log_info("Cluster {k}")
    pb <- progress_bar$new(
      format = "[:bar] :percent :elapsed",
      total = M,
      clear = FALSE,
      width = 50
    )
    for (m in 1:M) {
      cluster_permutation <- matrix(sample(1:K), nrow = 1)

      cluster_position <- which(cluster_permutation == k)
      # D+
      data_train_p <- R.utils::wrap(
        data_train_k[,, cluster_permutation[1:cluster_position]],
        map = list(NA, 2)
      )
      # D-
      if (cluster_position == 1) {
        data_train_m <- NULL
      } else {
        data_train_m <- R.utils::wrap(
          data_train_k[,, cluster_permutation[1:(cluster_position - 1)]],
          map = list(NA, 2)
        )
      }

      # v = accuracy
      # accuracy(D+) - accuracy(D-)
      if (cluster_position == 1) {
        phim[, k, m] <- fn_accuracy(actual_states, X_train = data_train_p, data_test) - 0
      } else {
        phim[, k, m] <- fn_accuracy(actual_states, X_train = data_train_p, data_test) -
          fn_accuracy(actual_states, X_train = data_train_m, data_test)
      }

      # Shapley value for cluster k up to m
      phi[, k, m] <- apply(phim[, k, ], MARGIN = 1, FUN = mean, na.rm = TRUE)
      pb$tick()
    }
  }

  return(phi)
}

fn_shapley_cluster <- function(
  K,
  M,
  data_train_k,
  data_test,
  prediction_accuracy,
  method
) {
  N_test <- dim(data_test)[1]
  phim <- array(NA, dim = c(N_test, K, M))
  phi <- array(NA, dim = c(N_test, K, M))

  log_info("Calculating Shapley values for each cluster")
  for (k in 1:K) {
    log_info("Cluster {k}")
    pb <- progress_bar$new(
      format = "[:bar] :percent :elapsed",
      total = M,
      clear = FALSE,
      width = 50
    )
    for (m in 1:M) {
      cluster_permutation <- matrix(sample(1:K), nrow = 1)

      cluster_position <- which(cluster_permutation == k)
      # D+
      data_train_p <- R.utils::wrap(
        data_train_k[,, cluster_permutation[1:cluster_position]],
        map = list(NA, 2)
      )
      # D-
      if (cluster_position == 1) {
        data_train_m <- NULL
      } else {
        data_train_m <- R.utils::wrap(
          data_train_k[,, cluster_permutation[1:(cluster_position - 1)]],
          map = list(NA, 2)
        )
      }

      if (prediction_accuracy) {
        # v = (y - f(x))^2 - y^2
        if (cluster_position == 1) {
          phim[, k, m] <- (data_test[, 1] -
            fn_prediction(data_train = data_train_p, data_test = data_test, method = method))^2 -
            ((data_test[, 1] - 0))^2
        } else {
          phim[, k, m] <- (data_test[, 1] -
            fn_prediction(data_train = data_train_p, data_test = data_test, method = method))^2 -
            (data_test[, 1] - fn_prediction(data_train = data_train_m, data_test = data_test, method = method))^2
        }
      } else {
        # v = f(x)
        # prediction(D+) - prediction(D-)
        if (cluster_position == 1) {
          phim[, k, m] <- fn_prediction(data_train = data_train_p, data_test = data_test, method = method) - 0
        } else {
          phim[, k, m] <- fn_prediction(data_train = data_train_p, data_test = data_test, method = method) -
            fn_prediction(data_train = data_train_m, data_test = data_test, method = method)
        }
      }

      # Shapley value for cluster k up to m
      phi[, k, m] <- apply(phim[, k, ], MARGIN = 1, FUN = mean, na.rm = TRUE)
      pb$tick()
    }
  }

  return(phi)
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
