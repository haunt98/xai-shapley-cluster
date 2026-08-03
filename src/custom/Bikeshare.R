library(R.utils)
library(logger)
library(optparse)

library(khroma)

discreterainbow <- color("discreterainbow")
palette(discreterainbow(12))

source("src/custom/common.R")

log_info("XAI Shapley Cluster - Bikeshare Dataset")

# Init
option_list <- list(
  make_option(
    c("--prediction-accuracy"),
    type = "logical",
    default = TRUE,
    help = "Use prediction accuracy [default %default]"
  ),
  make_option(
    c("--method"),
    type = "character",
    default = "rf",
    help = "Regression model to use [default %default]"
  )
)
opt <- parse_args(OptionParser(option_list = option_list))

prediction_accuracy <- opt$`prediction-accuracy`
log_info("prediction_accuracy: {prediction_accuracy}")

mmethod <- opt$method
log_info("method: {mmethod}")

M <- 150 # Number of cluster permutations
log_info("M: {M}")

# Load Bikeshare dataset
data("Bikeshare", package = "ISLR2")

# Remove incomplete cases
Bikeshare <- Bikeshare[complete.cases(Bikeshare), ]

# Clusters: 1..12
K <- length(unique(Bikeshare$mnth)) # 12
log_info("Number of clusters: {K}")

# Build data matrix: y, x1..x8 (all predictors), xS (mnth)
mdata_full <- cbind(
  y = Bikeshare$bikers,
  x1 = Bikeshare$hr,
  x2 = Bikeshare$holiday,
  x3 = Bikeshare$weekday,
  x4 = Bikeshare$workingday,
  x5 = Bikeshare$weathersit,
  x6 = Bikeshare$temp,
  x7 = Bikeshare$hum,
  x8 = Bikeshare$windspeed,
  xS = as.integer(Bikeshare$mnth)
)
n_features <- 8
include_mdata <- seq_len(n_features + 1) # y + all predictors

mnth_sizes_full <- table(mdata_full[, "xS"])
log_info("mdata_full per cluster: {paste(names(mnth_sizes_full), mnth_sizes_full, sep = '=', collapse = ', ')}")
# 1=688, 2=649, 3=730, 4=719, 5=744, 6=720, 7=744, 8=731, 9=717, 10=743, 11=719, 12=741

N_eval_per_cluster <- 30 # Eval set for final evaluation
log_info("N_eval_per_cluster: {N_eval_per_cluster}")

N_shapley_test_per_cluster <- 200 # Used for training Shapley
log_info("N_shapley_test_per_cluster: {N_shapley_test_per_cluster}")

N_shapley_train_per_cluster <- 400 # Used for training Shapley
log_info("N_shapley_train_per_cluster: {N_shapley_train_per_cluster}")

N_strategy_train <- 4800 # Used for final evaluation = 12 (cluster) x 400
log_info("N_strategy_train: {N_strategy_train}")

set.seed(19)

# Split data by cluster
cluster_indices <- lapply(seq_len(K), function(k) {
  which(mdata_full[, "xS"] == k)
})

cluster_eval_idx <- lapply(cluster_indices, function(idx) {
  sample(idx, N_eval_per_cluster)
})

# Exclude eval
cluster_pool_idx <- lapply(seq_len(K), function(k) {
  setdiff(cluster_indices[[k]], cluster_eval_idx[[k]])
})

cluster_shapley_test_idx <- lapply(cluster_pool_idx, function(idx) {
  sample(idx, N_shapley_test_per_cluster)
})

cluster_pool_exclude_shapley_test <- lapply(seq_len(K), function(k) {
  setdiff(cluster_pool_idx[[k]], cluster_shapley_test_idx[[k]])
})

cluster_shapley_train_idx <- lapply(cluster_pool_exclude_shapley_test, function(idx) {
  sample(idx, N_shapley_train_per_cluster)
})

mdata_eval <- do.call(
  rbind,
  lapply(seq_len(K), function(k) {
    mdata_full[cluster_eval_idx[[k]], include_mdata, drop = FALSE]
  })
)

eval_cluster_labels <- do.call(
  c,
  lapply(seq_len(K), function(k) {
    rep(k, N_eval_per_cluster)
  })
)

mdata_shapley_test <- do.call(
  rbind,
  lapply(seq_len(K), function(k) {
    mdata_full[cluster_shapley_test_idx[[k]], include_mdata, drop = FALSE]
  })
)

mdata_shapley_train_k <- lapply(seq_len(K), function(k) {
  mdata_full[cluster_shapley_train_idx[[k]], include_mdata, drop = FALSE]
})

# Plot train data
feature_names <- c("bikers", "hr", "holiday", "weekday", "workingday", "weathersit", "temp", "hum", "windspeed")
plot_groups <- list(1:3, 4:6, 7:9)

fn_plot_train_data <- function(mdata, col_array, title) {
  for (pg in seq_along(plot_groups)) {
    par(mar = c(2, 5, 2, 2))
    par(mfrow = c(length(plot_groups[[pg]]), 1))
    for (j in plot_groups[[pg]]) {
      plot(mdata[, j], col = col_array, ylab = feature_names[j])
    }
    mtext(title, side = 3, line = -1.5, outer = TRUE)
  }
}

# mdata_shapley_train_k
mdata_shapley_train <- do.call(rbind, mdata_shapley_train_k)
col_array_shapley_train <- rep(seq_len(K), each = N_shapley_train_per_cluster)
fn_plot_train_data(mdata_shapley_train, col_array_shapley_train, "Shapley train data (Bikeshare)")

# mdata_shapley_test
col_array_shapley_test <- rep(seq_len(K), each = N_shapley_test_per_cluster)
fn_plot_train_data(mdata_shapley_test, col_array_shapley_test, "Shapley test data (Bikeshare)")

# mdata_eval
col_array_eval <- rep(seq_len(K), each = N_eval_per_cluster)
fn_plot_train_data(mdata_eval, col_array_eval, "Eval data (Bikeshare)")

# Phase 1: Calculate Shapley values for each cluster

set.seed(7)

phi <- fn_shapley_cluster(
  K = K,
  M = M,
  data_train_k = mdata_shapley_train_k,
  data_test = mdata_shapley_test,
  prediction_accuracy = prediction_accuracy,
  method = mmethod
)

# Global Shapley values for each cluster
global_phi <- apply(phi, MARGIN = c(2, 3), FUN = mean, na.rm = TRUE)
global_phi_M <- global_phi[, M]
log_info("global_phi_M: {paste(seq_len(K), round(global_phi_M, 4), sep = '=', collapse = ', ')}")

# Plot convergence of global Shapley values for each cluster
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
  ylab = "Global Shapley values (Bikeshare)",
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

# Local Shapley values for selected points, 4 representative months
selected_months <- c(1, 4, 8, 12)
selected_points <- (selected_months - 1) * N_shapley_test_per_cluster + 50

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

full_prediction <- fn_prediction(data_train = mdata_shapley_train, data_test = mdata_shapley_test, method = mmethod)
log_info("MSE: {mean((full_prediction - mdata_shapley_test[, 1])^2)}")

if (!prediction_accuracy) {
  plotted_value <- full_prediction
  plot_title <- "Predictions"
} else {
  plotted_value <- (full_prediction - mdata_shapley_test[, 1])^2
  plot_title <- "Squared Error"
}

# Top: full prediction or squared error
plot(seq_along(plotted_value), plotted_value, xaxs = "i", main = "", type = "l", col = 11)
for (point_index in selected_points) {
  points(point_index, plotted_value[point_index], pch = 16, cex = 1.5, col = "black")
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

mtext(plot_title, side = 3, line = -7.5, outer = TRUE)

# Phase 2: Build training data for 2 strategies: equal, max

set.seed(11)

# Strategy equal (Baseline): sample equal datapoints for each cluster
N_equal_per_cluster <- N_strategy_train / K

cluster_equal_idx <- lapply(seq_len(K), function(k) {
  sample(cluster_pool_idx[[k]], N_equal_per_cluster)
})

# Strategy max (Proposed)
tau <- max(2.5 * sd(global_phi_M), 1e-6)
w_k <- exp(-global_phi_M / tau)
quota <- N_strategy_train * w_k / sum(w_k)

# Allocate each cluster by quota, capped by pool size
pool_cap <- sapply(cluster_pool_idx, length)
N_min_k <- floor(N_equal_per_cluster / 2)
N_max_k <- pmin(pmax(floor(quota), N_min_k), pool_cap)

remaining <- N_strategy_train - sum(N_max_k)

while (remaining > 0) {
  # Under budget: add to the cluster with the largest deficit that still has room
  j <- which.max(ifelse(N_max_k < pool_cap, quota - N_max_k, -Inf))
  N_max_k[j] <- N_max_k[j] + 1
  remaining <- remaining - 1
}

while (remaining < 0) {
  # Over budget: trim the cluster furthest above its quota, but never below the minimum
  j <- which.max(ifelse(N_max_k > N_min_k, N_max_k - quota, -Inf))
  N_max_k[j] <- N_max_k[j] - 1
  remaining <- remaining + 1
}

stopifnot(sum(N_max_k) == N_strategy_train)
log_info("N_max_k: {paste(seq_len(K), N_max_k, sep = '=', collapse = ', ')}")

cluster_max_idx <- lapply(seq_len(K), function(k) {
  sample(cluster_pool_idx[[k]], N_max_k[k])
})

# Phase 3: Evaluation

# Strategy equal
mdata_equal_train <- do.call(
  rbind,
  lapply(seq_len(K), function(k) {
    mdata_full[cluster_equal_idx[[k]], include_mdata, drop = FALSE]
  })
)

pred_equal <- fn_prediction(data_train = mdata_equal_train, data_test = mdata_eval, method = mmethod)

# Strategy max
mdata_max_train <- do.call(
  rbind,
  lapply(seq_len(K), function(k) {
    mdata_full[cluster_max_idx[[k]], include_mdata, drop = FALSE]
  })
)

pred_max <- fn_prediction(data_train = mdata_max_train, data_test = mdata_eval, method = mmethod)

# MSE per cluster across 2 strategies
fn_mse_per_cluster <- function(pred) {
  sapply(seq_len(K), function(k) {
    test_k <- which(eval_cluster_labels == k)
    mean((pred[test_k] - mdata_eval[test_k, 1])^2)
  })
}

mse_equal <- fn_mse_per_cluster(pred_equal)
mse_max <- fn_mse_per_cluster(pred_max)

log_info("MSE equal: {paste(seq_len(K), round(mse_equal, 4), sep = '=', collapse = ', ')}")
log_info("MSE max: {paste(seq_len(K), round(mse_max, 4), sep = '=', collapse = ', ')}")
log_info("Global MSE equal: {mean((pred_equal - mdata_eval[, 1])^2)}")
log_info("Global MSE max: {mean((pred_max - mdata_eval[, 1])^2)}")

# Plot MSE per cluster
par(mar = c(5, 5.5, 3, 1))
par(mfrow = c(1, 1))
ymax <- max(mse_equal, mse_max)
plot(
  seq_len(K),
  mse_equal,
  type = "l",
  col = 2,
  lwd = 3,
  ylim = c(0, ymax * 1.1),
  xlab = "Month",
  ylab = "MSE",
  xaxt = "n"
)
axis(1, at = seq_len(K), labels = seq_len(K))
lines(seq_len(K), mse_max, col = 4, lwd = 3, lty = 3)
legend(
  "topleft",
  legend = c("equal", "max"),
  col = c(1, 3),
  lty = c(1, 3),
  lwd = 2
)
