library(R.utils)
library(logger)
library(optparse)

library(khroma)

vibrant <- color("vibrant")
palette(vibrant(7))

source("src/custom/common.R")

log_info("XAI Shapley Cluster - Bikeshare Dataset")

# Init
mmethod <- "rf" # Which regression model to use
log_info("method: {mmethod}")

option_list <- list(
  make_option(
    c("--prediction-accuracy"),
    type = "logical",
    default = TRUE,
    help = "Use prediction accuracy [default %default]"
  )
)
opt <- parse_args(OptionParser(option_list = option_list))

prediction_accuracy <- opt$`prediction-accuracy`
log_info("prediction_accuracy: {prediction_accuracy}")

M <- 100 # Number of cluster permutations
log_info("M: {M}")

# Load Bikeshare dataset
data("Bikeshare", package = "ISLR2")

# Remove incomplete cases
Bikeshare <- Bikeshare[complete.cases(Bikeshare), ]

# Clusters by month
mnth_labels <- sort(unique(Bikeshare$mnth))
K <- length(mnth_labels) # 12
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
  xS = Bikeshare$mnth
)
n_features <- 8
include_mdata <- seq_len(n_features + 1) # y + all predictors

secret_test_per_month <- 15 # Secret test set for final evaluation
log_info("secret_test_per_month: {secret_test_per_month}")

shapley_test_per_month <- 30 # Used for training Shapley
log_info("shapley_test_per_month: {shapley_test_per_month}")

shapley_train_per_month <- 400 # Used for training Shapley
log_info("shapley_train_per_month: {shapley_train_per_month}")

N_budget <- 600 # Used for final evaluation = 12 (cluster) x 50
log_info("N_budget: {N_budget}")

set.seed(19)

# Split data by cluster month
month_indices <- lapply(seq_len(K), function(k) {
  which(Bikeshare$mnth == mnth_labels[k])
})

month_secret_test_idx <- lapply(month_indices, function(idx) {
  sample(idx, secret_test_per_month)
})

# Exclude secret test
month_pool_idx <- lapply(seq_len(K), function(k) {
  setdiff(month_indices[[k]], month_secret_test_idx[[k]])
})

month_shapley_test_idx <- lapply(month_pool_idx, function(idx) {
  sample(idx, shapley_test_per_month)
})

month_pool_exclude_shapley_test <- lapply(seq_len(K), function(k) {
  setdiff(month_pool_idx[[k]], month_shapley_test_idx[[k]])
})

month_shapley_train_idx <- lapply(month_pool_exclude_shapley_test, function(idx) {
  sample(idx, shapley_train_per_month)
})

mdata_secret_test <- do.call(
  rbind,
  lapply(seq_len(K), function(k) {
    mdata_full[month_secret_test_idx[[k]], include_mdata, drop = FALSE]
  })
)

N_secret_test <- nrow(mdata_secret_test)

secret_test_month_labels <- do.call(
  c,
  lapply(seq_len(K), function(k) {
    rep(mnth_labels[k], secret_test_per_month)
  })
)

mdata_shapley_test <- do.call(
  rbind,
  lapply(seq_len(K), function(k) {
    mdata_full[month_shapley_test_idx[[k]], include_mdata, drop = FALSE]
  })
)

mdata_shapley_train_k <- lapply(seq_len(K), function(k) {
  mdata_full[month_shapley_train_idx[[k]], include_mdata, drop = FALSE]
})

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
names(global_phi_M) <- mnth_labels

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

# Phase 2: Build training data for 3 strategies: equal, one, max

set.seed(11)

N_equal_per_month <- N_budget / K # 50

# Strategy equal (Baseline): sample equal datapoints for each cluster month
month_equal_idx <- lapply(seq_len(K), function(k) {
  sample(month_pool_exclude_shapley_test[[k]], N_equal_per_month)
})

# Strategy one (Baseline): sample N_budget datapoints exclusively for each cluster month
month_one_idx <- lapply(seq_len(K), function(k) {
  sample(month_pool_exclude_shapley_test[[k]], N_budget)
})

# Strategy max (Proposed)
tau <- max(sd(global_phi_M), 1e-6)
w_k <- exp(-global_phi_M / tau)
quota <- N_budget * w_k / sum(w_k)

# Allocate each cluster month by quota
N_max_k <- pmax(floor(quota), 1)
remaining <- N_budget - sum(N_max_k)

while (remaining > 0) {
  # Allocate remaining with largest different between quota and current allocation
  j <- which.max(quota - N_max_k)
  N_max_k[j] <- N_max_k[j] + 1
  remaining <- remaining - 1
}
names(N_max_k) <- mnth_labels
log_info("N_max_k: {paste(mnth_labels, N_max_k, sep = '=', collapse = ', ')}")

month_max_idx <- lapply(seq_len(K), function(k) {
  sample(month_pool_exclude_shapley_test[[k]], N_max_k[k])
})

# Phase 3: Evaluation

# Strategy equal
mdata_equal_train <- do.call(
  rbind,
  lapply(seq_len(K), function(k) {
    mdata_full[month_equal_idx[[k]], include_mdata, drop = FALSE]
  })
)

pred_equal <- fn_prediction(data_train = mdata_equal_train, data_test = mdata_secret_test, method = mmethod)

# Strategy one
pred_one <- numeric(N_secret_test)

for (k in seq_len(K)) {
  test_k <- which(secret_test_month_labels == mnth_labels[k])

  mdata_one_train_k <- mdata_full[month_one_idx[[k]], include_mdata, drop = FALSE]

  pred_one[test_k] <- fn_prediction(
    data_train = mdata_one_train_k,
    data_test = mdata_secret_test[test_k, , drop = FALSE],
    method = mmethod
  )
}

# Strategy max
mdata_max_train <- do.call(
  rbind,
  lapply(seq_len(K), function(k) {
    mdata_full[month_max_idx[[k]], include_mdata, drop = FALSE]
  })
)

pred_max <- fn_prediction(data_train = mdata_max_train, data_test = mdata_secret_test, method = mmethod)

# MSE per month across 3 strategies
fn_mse_per_month <- function(pred) {
  sapply(seq_len(K), function(k) {
    test_k <- which(secret_test_month_labels == mnth_labels[k])
    mean((pred[test_k] - mdata_secret_test[test_k, 1])^2)
  })
}

mse_equal <- fn_mse_per_month(pred_equal)
mse_one <- fn_mse_per_month(pred_one)
mse_max <- fn_mse_per_month(pred_max)

names(mse_equal) <- mnth_labels
names(mse_one) <- mnth_labels
names(mse_max) <- mnth_labels

log_info("MSE equal: {paste(mnth_labels, round(mse_equal, 4), sep = '=', collapse = ', ')}")
log_info("MSE one: {paste(mnth_labels, round(mse_one, 4), sep = '=', collapse = ', ')}")
log_info("MSE max: {paste(mnth_labels, round(mse_max, 4), sep = '=', collapse = ', ')}")
log_info("Global MSE equal: {mean((pred_equal - mdata_secret_test[, 1])^2)}")
log_info("Global MSE one: {mean((pred_one - mdata_secret_test[, 1])^2)}")
log_info("Global MSE max: {mean((pred_max - mdata_secret_test[, 1])^2)}")

# Plot MSE per month
par(mar = c(5, 5.5, 3, 1))
par(mfrow = c(1, 1))
ymax <- max(mse_equal, mse_one, mse_max)
plot(
  seq_len(K),
  mse_equal,
  type = "l",
  col = 1,
  lwd = 2,
  ylim = c(0, ymax * 1.1),
  xlab = "Month",
  ylab = "MSE",
  xaxt = "n"
)
axis(1, at = seq_len(K), labels = mnth_labels)
lines(seq_len(K), mse_one, col = 2, lwd = 2, lty = 2)
lines(seq_len(K), mse_max, col = 3, lwd = 2, lty = 3)
legend(
  "topleft",
  legend = c("equal", "one", "max"),
  col = c(1, 2, 3),
  lty = c(1, 2, 3),
  lwd = 2
)
