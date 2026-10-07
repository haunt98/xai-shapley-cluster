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
  ),
  make_option(
    c("--output"),
    type = "character",
    default = "results",
    help = "Output directory to save results as CSV [default %default]"
  ),
  make_option(
    c("--pdf"),
    type = "character",
    default = "Rplots.pdf",
    help = "PDF file to save plots [default %default]"
  )
)
opt <- parse_args(OptionParser(option_list = option_list))

prediction_accuracy <- opt$`prediction-accuracy`
log_info("prediction_accuracy: {prediction_accuracy}")

mmethod <- opt$method
log_info("method: {mmethod}")

output_dir <- opt$output
log_info("output_dir: {output_dir}")

pdf_path <- opt$pdf
log_info("pdf: {pdf_path}")

dir.create(dirname(pdf_path), recursive = TRUE, showWarnings = FALSE)
pdf(file = pdf_path)

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

# Fail fast if the dataset changes
expected_mnth_sizes <- c(688, 649, 730, 719, 744, 720, 744, 731, 717, 743, 719, 741)
stopifnot(identical(as.integer(mnth_sizes_full), as.integer(expected_mnth_sizes)))

# Split config (per cluster)
N_eval_per_cluster <- 30 # Eval set for final evaluation
log_info("N_eval_per_cluster: {N_eval_per_cluster}")

N_shapley_test_per_cluster <- 200 # Used for testing Shapley
log_info("N_shapley_test_per_cluster: {N_shapley_test_per_cluster}")

N_shapley_train_per_cluster <- 400 # Used for training Shapley
log_info("N_shapley_train_per_cluster: {N_shapley_train_per_cluster}")

# Fail fast if any cluster is too small
stopifnot(all(mnth_sizes_full >= N_eval_per_cluster + N_shapley_test_per_cluster + N_shapley_train_per_cluster))

N_strategy_per_cluster <- 400 # Strategy equal: training points per cluster
log_info("N_strategy_per_cluster: {N_strategy_per_cluster}")

N_strategy_train <- N_strategy_per_cluster * K # Total training points for strategies equal/max
log_info("N_strategy_train: {N_strategy_train} = {K} x {N_strategy_per_cluster}")

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

# Sanity check: eval, test, train are disjoint per cluster
stopifnot(all(mapply(
  function(ev, te, tr) {
    length(intersect(ev, te)) == 0 && length(intersect(ev, tr)) == 0 && length(intersect(te, tr)) == 0
  },
  cluster_eval_idx,
  cluster_shapley_test_idx,
  cluster_shapley_train_idx
)))

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
    cluster_boundaries <- which(diff(col_array) != 0)
    for (j in plot_groups[[pg]]) {
      plot(mdata[, j], col = col_array, pch = 16, cex = 0.5, ylab = feature_names[j])
      abline(v = cluster_boundaries, lty = 3, col = "gray70")
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

# Save Shapley results to CSV
params <- data.frame(
  name = c(
    "method",
    "prediction_accuracy",
    "K",
    "M",
    "N_shapley_train_per_cluster",
    "N_shapley_test_per_cluster",
    "N_eval_per_cluster",
    "N_strategy_per_cluster"
  ),
  value = c(
    mmethod,
    prediction_accuracy,
    K,
    M,
    N_shapley_train_per_cluster,
    N_shapley_test_per_cluster,
    N_eval_per_cluster,
    N_strategy_per_cluster
  )
)
fn_write_csv(params, file.path(output_dir, "params.csv"))

# Local Shapley values at the final iteration M: one row per test point, one column per cluster
phi_local <- phi[,, M]
colnames(phi_local) <- paste0("cluster_", seq_len(K))
fn_write_csv(
  data.frame(point = seq_len(dim(mdata_shapley_test)[1]), phi_local),
  file.path(output_dir, "phi_local.csv")
)

# Global Shapley values at the final iteration M: one row per cluster
fn_write_csv(
  data.frame(cluster = paste0("cluster_", seq_len(K)), global_phi = global_phi[, M]),
  file.path(output_dir, "phi_global.csv")
)

# Plot convergence of global Shapley values for each cluster
par(mar = c(5, 5.5, 3, 1))
par(mfrow = c(1, 1))
plot(
  seq_len(M),
  global_phi[1, ],
  type = "l",
  col = adjustcolor(1, alpha.f = 0.85),
  lwd = 1.5,
  xlim = c(1, M),
  ylim = range(global_phi, na.rm = TRUE),
  xlab = "Number of iterations (M)",
  ylab = "Global Shapley values (Bikeshare)",
)
grid()
abline(h = 0, lty = 2, col = "gray50")
for (cluster_index in 2:K) {
  lines(seq_len(M), global_phi[cluster_index, ], col = adjustcolor(cluster_index, alpha.f = 0.85), lwd = 1.5)
}
legend(
  "topright",
  title = "Clusters included",
  legend = seq_len(K),
  col = seq_len(K),
  lty = 1,
  lwd = 1.5,
  bty = "n"
)

# Local Shapley values for selected points, 4 representative months
selected_months <- c(1, 4, 8, 12)
selected_points <- (selected_months - 1) * N_shapley_test_per_cluster + 50

# Save local Shapley values of selected points at the final iteration M
phi_selected <- phi[selected_points, , M]
colnames(phi_selected) <- paste0("cluster_", seq_len(K))
fn_write_csv(
  data.frame(point = selected_points, phi_selected),
  file.path(output_dir, "phi_selected.csv")
)

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
  heights = c(2, 2, 1.2)
)

full_prediction <- fn_prediction(data_train = mdata_shapley_train, data_test = mdata_shapley_test, method = mmethod)
mse_full <- mean((full_prediction - mdata_shapley_test[, 1])^2)
log_info("MSE: {mse_full}")

# Save full model MSE
fn_write_csv(data.frame(mse = mse_full), file.path(output_dir, "mse_full.csv"))

# Save full model prediction per test point
fn_write_csv(
  data.frame(
    point = seq_len(dim(mdata_shapley_test)[1]),
    actual = mdata_shapley_test[, 1],
    prediction = full_prediction,
    squared_error = (full_prediction - mdata_shapley_test[, 1])^2
  ),
  file.path(output_dir, "prediction.csv")
)

if (!prediction_accuracy) {
  plotted_value <- full_prediction
} else {
  plotted_value <- (full_prediction - mdata_shapley_test[, 1])^2
}

# Top: full prediction or squared error
plot(seq_along(plotted_value), plotted_value, xaxs = "i", main = "", type = "l", col = 11, cex = 0.5)
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

# Phase 2: Build training data for 2 strategies: equal, max

set.seed(11)

# Strategy equal (Baseline): sample equal datapoints for each cluster
cluster_equal_idx <- lapply(seq_len(K), function(k) {
  sample(cluster_pool_idx[[k]], N_strategy_per_cluster)
})

# Strategy max (Proposed)
tau <- max(2.5 * sd(global_phi_M), 1e-6)
w_k <- exp(-global_phi_M / tau)
quota <- N_strategy_train * w_k / sum(w_k)

# Allocate each cluster by quota, capped by pool size
pool_cap <- sapply(cluster_pool_idx, length)
N_min_k <- floor(N_strategy_per_cluster / 2)
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

# Save training data allocation per strategy
fn_write_csv(
  data.frame(
    cluster = paste0("cluster_", seq_len(K)),
    global_phi = global_phi_M,
    w = w_k,
    quota = quota,
    n_equal = rep(N_strategy_per_cluster, K),
    n_max = N_max_k
  ),
  file.path(output_dir, "strategy_allocation.csv")
)

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

# Save strategy predictions per eval point
fn_write_csv(
  data.frame(
    point = seq_len(dim(mdata_eval)[1]),
    cluster = eval_cluster_labels,
    actual = mdata_eval[, 1],
    pred_equal = pred_equal,
    pred_max = pred_max,
    squared_error_equal = (pred_equal - mdata_eval[, 1])^2,
    squared_error_max = (pred_max - mdata_eval[, 1])^2
  ),
  file.path(output_dir, "strategy_prediction.csv")
)

# MSE per cluster across 2 strategies
fn_mse_per_cluster <- function(pred) {
  sapply(seq_len(K), function(k) {
    test_k <- which(eval_cluster_labels == k)
    mean((pred[test_k] - mdata_eval[test_k, 1])^2)
  })
}

mse_equal <- fn_mse_per_cluster(pred_equal)
mse_max <- fn_mse_per_cluster(pred_max)
mse_equal_global <- mean((pred_equal - mdata_eval[, 1])^2)
mse_max_global <- mean((pred_max - mdata_eval[, 1])^2)

log_info("MSE equal: {paste(seq_len(K), round(mse_equal, 4), sep = '=', collapse = ', ')}")
log_info("MSE max: {paste(seq_len(K), round(mse_max, 4), sep = '=', collapse = ', ')}")
log_info("Global MSE equal: {mse_equal_global}")
log_info("Global MSE max: {mse_max_global}")

# Save MSE per cluster across 2 strategies and global MSE
fn_write_csv(
  data.frame(cluster = seq_len(K), mse_equal = mse_equal, mse_max = mse_max),
  file.path(output_dir, "mse_per_cluster.csv")
)
fn_write_csv(
  data.frame(strategy = c("equal", "max"), global_mse = c(mse_equal_global, mse_max_global)),
  file.path(output_dir, "mse_global.csv")
)

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
  col = c(2, 4),
  lty = c(1, 3),
  lwd = 2
)

dev.off()
