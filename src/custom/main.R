library(R.utils)
library(logger)
library(optparse)

source("src/custom/common.R")

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

log_info("XAI Shapley Cluster")


# Init
mmethod <- "rf" # Which regression model to use

option_list <- list(
  make_option(
    c("--prediction-accuracy"),
    type = "logical",
    default = TRUE,
    help = "Use prediction accuracy [default %default]"
  ),
  make_option(
    c("--global-classification"),
    type = "logical",
    default = FALSE,
    help = "Use global classification [default %default]"
  )
)
opt <- parse_args(OptionParser(option_list = option_list))

prediction_accuracy <- opt$`prediction-accuracy`
log_info("prediction_accuracy: {prediction_accuracy}")

global_classification <- opt$`global-classification`
log_info("global_classification: {global_classification}")

K <- 5 # Number of clusters
N <- 100 * K * 4 # Number of datapoints

N_train <- N / 4
N_test <- N / 4
N_eval <- N / 2 # Do nothing for now

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

# Sort data by cluster labels
mdata_train_full <- mdata_train_full[order((mdata_train_full[, index_mdata_xS])), ]

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

# Per cluster
K_train <- floor(N_train / K)
K_test <- floor(N_test / K)
K_eval <- floor(N_eval / K)


# Plot train data
# Assign colors to each cluster for training data
# [ 1, 1, ..., 1, 2, 2, ..., 2, 3, 3, ..., 3, 4, 4, ..., 4, 5, 5, ..., 5]
col_array_train <- array(NA, N_train)
for (k in 1:K) {
  col_array_train[((k - 1) * K_train + 1):(K_train * k)] <- rep(k, K_train)
}

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
      points(200:249, mdata_test[200:249, 2], col = 3, pch = 16)
    }
    if (j == 5) {
      points(300:349, mdata_test[300:349, 5], col = 3, pch = 16)
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
    points(X_test_pred[, j], col = "red")
  }

  mtext("Compare train data with AAKR predictions", side = 3, line = -1.5, outer = TRUE)

  par(mar = c(2, 5, 2, 2))
  par(mfrow = c(length(include_mdata), 1))

  # Plot y, x1, x2, x3, x4
  for (j in include_mdata) {
    plot(mdata_test[, j], col = col_array_train, ylab = colnames(mdata_full)[j], ylim = c(-1, 1))
    points(X_test_pred[, j], col = "red")
  }

  mtext("Compare test data with AAKR predictions", side = 3, line = -1.5, outer = TRUE)
}


M <- 250

# Split data into K clusters
mdata_train_k <- list()
for (k in 1:K) {
  mdata_train_k[[k]] <- mdata_train[(K_train * (k - 1) + 1):(K_train * k), , drop = FALSE]
}

set.seed(7)

if (global_classification) {
  phi <- fn_shapley_cluster_global_classification(
    K = K,
    M = M,
    data_train_k = mdata_train_k,
    data_test = mdata_test,
    actual_states = actual_states
  )
} else {
  phi <- fn_shapley_cluster(
    K = K,
    M = M,
    data_train_k = mdata_train_k,
    data_test = mdata_test,
    prediction_accuracy = prediction_accuracy,
    method = mmethod
  )
}

# Global Shapley values for each cluster
# phi has dimensions (N_test, K, M)
global_phi <- apply(phi, MARGIN = c(2, 3), FUN = mean, na.rm = TRUE)


# Plot convergence of Shapley values for each cluster
par(mar = c(5, 5.5, 3, 1))
par(mfrow = c(1, 1))
plot(
  seq_len(M),
  global_phi[1, ],
  type = "l",
  col = 1,
  lwd = 2,
  xlim = c(1, M),
  ylim = range(global_phi, na.rm = TRUE),
  xlab = "Number of iterations (M)",
  ylab = "Global Shapley values for each cluster",
)
for (cluster_index in 2:K) {
  lines(seq_len(M), global_phi[cluster_index, ], col = cluster_index, lwd = 2)
}
legend(
  "topright",
  title = "Clusters included",
  legend = seq_len(K),
  col = seq_len(K),
  lty = 1,
  lwd = 2
)

if (!global_classification) {
  selected_points <- c(50, 150, 250, 350, 450)
  full_prediction <- fn_prediction(data_train = mdata_train, data_test = mdata_test, method = mmethod)

  par(mar = c(3, 3, 2, 2) * .7)
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
  } else {
    plotted_value <- (full_prediction - mdata_test[, 1])^2
    plot_title <- "Squared Error"
  }

  # Top: full prediction or squared error
  plot(seq_along(plotted_value), plotted_value, xaxs = "i", main = "", type = "l", col = 11, lwd = 3)
  for (point_index in selected_points) {
    points(point_index, plotted_value[point_index], pch = 16, cex = 1.5, col = 9)
    abline(v = point_index, lty = 2)
  }

  # Middle: local Shapley values for each selected point
  for (point_index in selected_points) {
    barplot(phi[point_index, , M], horiz = TRUE, col = seq_len(K), main = "")
    box()
  }

  # Bottom: convergence of Shapley values for each selected point
  for (point_index in selected_points) {
    point_phi <- phi[point_index, , , drop = FALSE]
    plot(NA, xlim = c(1, M), ylim = range(point_phi, na.rm = TRUE), xlab = "", ylab = "", axes = FALSE)
    axis(1, labels = FALSE)
    axis(2)
    abline(h = 0, lty = 3)
    for (cluster_index in seq_len(K)) {
      lines(seq_len(M), phi[point_index, cluster_index, ], col = cluster_index)
    }
    box()
  }

  mtext(plot_title, side = 3, line = -11.5, outer = TRUE)
}
